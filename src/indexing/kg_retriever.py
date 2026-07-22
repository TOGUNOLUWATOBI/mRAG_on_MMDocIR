# src/indexing/kg_retriever.py
# Graph-traversal retriever: given a natural-language query, find matching entity
# nodes in the graph, expand their neighborhood via BFS, then collect the source
# chunks attached to the visited nodes as context for generation.
import logging
import math
import re
from typing import Any, Dict, List, Set, Tuple

import networkx as nx

logger = logging.getLogger(__name__)


class KGRetriever:
    """
    Retrieval via entity matching + BFS neighborhood expansion.

    Steps:
    1. Scan graph nodes whose name appears as a whole-word/phrase match in the query.
    2. Expand from those seed nodes for `hops` BFS hops (both directions), capping
       fan-out through high-degree "hub" nodes to their best-evidenced neighbors.
    3. Collect source chunks attached to all visited nodes.
    4. Deduplicate and return up to top_k chunks.

    Fallback when no entity is found in the query: use the top-degree nodes
    (most-connected entities = most central concepts in the corpus).
    """

    # Per-hop score multiplier — seed-layer chunks outrank 1-hop, which outrank
    # 2-hop, but as a graded decay rather than the old hard layer partition.
    # Hops beyond this list decay geometrically from the last entry.
    HOP_DECAY = [1.0, 0.5, 0.25]

    def __init__(
        self,
        graph: nx.DiGraph,
        chunks: List[Dict[str, Any]],
        hops: int = 2,
        min_seed_length: int = 3,
        max_fanout: int = 15,
        use_word_boundary_seeds: bool = True,
    ):
        self.graph = graph
        self.hops = hops
        self.min_seed_length = min_seed_length
        self.max_fanout = max_fanout
        # Ablation switch (see KG_ROADMAP.md "Tier 1" follow-up): lets a caller
        # reproduce the pre-Epic-3 raw-substring seed matching to isolate its
        # individual contribution from KG_MAX_FANOUT's. Not meant to be disabled
        # in normal use — word-boundary matching fixes a real false-positive bug
        # (see the docstring below).
        self.use_word_boundary_seeds = use_word_boundary_seeds

        # UID → chunk dict for O(1) lookup
        self.chunk_index: Dict[str, Dict] = {
            f"{c['pdf_name']}::{c['chunk_id']}": c for c in chunks
        }

        # Sort entities longest-first so more specific names match before substrings.
        # Drop entities shorter than min_seed_length: single/double-character node
        # labels are near-always extraction noise ("N", "y", "RE"), not real entities,
        # and match as false-positive seeds in almost every query.
        entity_names = sorted(
            (e for e in graph.nodes() if len(e) >= min_seed_length),
            key=len,
            reverse=True,
        )
        self.entities: List[str] = entity_names

        # Precompiled word-boundary patterns, not plain substring checks: a raw
        # `"rope" in "...europe..."` substring test false-positives on short real
        # words hiding inside longer ones (the extracted ML-term node "RoPE" matched
        # every query mentioning "Europe"). `\b` anchors to word edges so "rope"
        # only matches "rope" as its own token, not as a fragment of "europe".
        self._entity_patterns: List[Tuple[str, "re.Pattern"]] = [
            (e, re.compile(r"\b" + re.escape(e.lower()) + r"\b"))
            for e in entity_names
        ]

    # ------------------------------------------------------------------
    def _find_seeds(self, query: str) -> List[str]:
        query_lower = query.lower()
        if not self.use_word_boundary_seeds:
            # Pre-Epic-3 behavior, kept only for the ablation grid.
            return [e for e in self.entities if e.lower() in query_lower]
        return [e for e, pattern in self._entity_patterns if pattern.search(query_lower)]

    def _weighted_neighbors(self, node: str) -> List[Tuple[str, int]]:
        """Neighbors of `node` paired with edge evidence strength (# source chunks)."""
        neighbors = []
        for succ in self.graph.successors(node):
            weight = len(self.graph.edges[node, succ].get("source_chunk_uids", []))
            neighbors.append((succ, weight))
        for pred in self.graph.predecessors(node):
            weight = len(self.graph.edges[pred, node].get("source_chunk_uids", []))
            neighbors.append((pred, weight))
        return neighbors

    def _bfs_expand(self, seeds: List[str]) -> List[Set[str]]:
        """Return a list of node sets per hop: [seed_nodes, hop1_nodes, hop2_nodes, ...]
        so callers can prioritise chunks from nodes closer to the query.

        Fan-out through any single node is capped at `max_fanout`, keeping only its
        best-evidenced neighbors (by # of source chunks on the connecting edge) —
        without this, generic high-degree "hub" nodes (e.g. "students", degree 200+)
        drown well-targeted seeds in unrelated neighbors within a single hop.
        """
        layers: List[Set[str]] = [set(seeds)]
        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)
        for _ in range(self.hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                neighbors = self._weighted_neighbors(node)
                if len(neighbors) > self.max_fanout:
                    neighbors.sort(key=lambda pair: pair[1], reverse=True)
                    neighbors = neighbors[: self.max_fanout]
                next_frontier.update(n for n, _ in neighbors if n not in visited)
            visited.update(next_frontier)
            frontier = next_frontier
            layers.append(set(next_frontier))
        return layers

    def _node_specificity(self, node: str) -> float:
        """Rare, specific entities should outrank generic hub matches — e.g. a
        query mentioning "NTU" (69 source chunks) is more informative than one
        that happens to also match "students" (115 source chunks, a generic hub
        that co-occurs with almost any education-related question). Reuses the
        same evidence-count statistic KG_MAX_FANOUT already uses for edges,
        applied here to a node's own total evidence (source_chunk_uids count).
        """
        evidence = len(self.graph.nodes[node].get("source_chunk_uids", []))
        return 1.0 / (1.0 + math.log1p(max(evidence, 1)))

    def _hop_weight(self, hop: int) -> float:
        if hop < len(self.HOP_DECAY):
            return self.HOP_DECAY[hop]
        # Geometric decay beyond the configured hops, halving each further hop.
        extra_hops = hop - len(self.HOP_DECAY) + 1
        return self.HOP_DECAY[-1] * (0.5 ** extra_hops)

    def _collect_chunks(self, layers: List[Set[str]], top_k: int) -> List[Dict[str, Any]]:
        # Score each chunk by (hop-distance decay) x (contributing node's
        # specificity), then sort once by score. Previously every chunk got a
        # hardcoded score of 1.0 and ordering was purely "seed layer, then
        # 1-hop, then 2-hop" with arbitrary order inside each layer — meaning a
        # chunk reached only via a generic hub node (e.g. "students") ranked
        # identically to one reached via a rare, on-topic entity (e.g. "NTU"),
        # as long as they were in the same layer. `sorted(nodes)` below is only
        # a deterministic tiebreak for nodes at the same score, not the ranking
        # signal itself (see KG_ROADMAP.md's Tier 1 determinism fix).
        seen_uids: Set[str] = set()
        candidates: List[Tuple[float, str, Dict[str, Any]]] = []
        for hop, nodes in enumerate(layers):
            hop_weight = self._hop_weight(hop)
            for node in sorted(nodes):
                if node not in self.graph:
                    continue
                node_score = hop_weight * self._node_specificity(node)
                for uid in self.graph.nodes[node].get("source_chunk_uids", []):
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    chunk = self.chunk_index.get(uid)
                    if chunk:
                        candidates.append((node_score, uid, chunk))

        # Sort by score descending; uid ascending as a deterministic tiebreak.
        candidates.sort(key=lambda c: (-c[0], c[1]))

        results = []
        for score, uid, chunk in candidates[:top_k]:
            results.append({
                "text": chunk["text"],
                "score": score,
                # "payload" matches the schema Qdrant-backed retrievers use
                # (vector_database.py) so evaluate_retrieval() works unmodified.
                "payload": {
                    "chunk_id": chunk["chunk_id"],
                    "pdf_name": chunk["pdf_name"],
                    "page_numbers": chunk.get("page_numbers"),
                    "uid": uid,
                },
            })
        return results

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        seeds = self._find_seeds(query)

        if not seeds:
            # Fallback: most-connected nodes carry the broadest context
            seeds = sorted(
                self.graph.nodes(), key=lambda n: self.graph.degree(n), reverse=True
            )[:5]
            logger.debug("No entity match — falling back to top-degree nodes: %s", seeds)
        else:
            logger.debug("Seed entities: %s", seeds[:5])

        layers = self._bfs_expand(seeds)
        results = self._collect_chunks(layers, top_k)
        logger.debug("Retrieved %d chunks for query: %.60s", len(results), query)
        return results
