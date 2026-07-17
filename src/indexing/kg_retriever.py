# src/indexing/kg_retriever.py
# Graph-traversal retriever: given a natural-language query, find matching entity
# nodes in the graph, expand their neighborhood via BFS, then collect the source
# chunks attached to the visited nodes as context for generation.
import logging
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

    def __init__(
        self,
        graph: nx.DiGraph,
        chunks: List[Dict[str, Any]],
        hops: int = 2,
        min_seed_length: int = 3,
        max_fanout: int = 15,
    ):
        self.graph = graph
        self.hops = hops
        self.min_seed_length = min_seed_length
        self.max_fanout = max_fanout

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

    def _collect_chunks(self, layers: List[Set[str]], top_k: int) -> List[Dict[str, Any]]:
        # Iterate layer-by-layer so seed chunks rank first, then 1-hop, then 2-hop
        seen_uids: Set[str] = set()
        results = []
        for nodes in layers:
            for node in nodes:
                if node not in self.graph:
                    continue
                for uid in self.graph.nodes[node].get("source_chunk_uids", []):
                    if uid in seen_uids:
                        continue
                    seen_uids.add(uid)
                    chunk = self.chunk_index.get(uid)
                    if chunk:
                        results.append({
                            "text": chunk["text"],
                            "score": 1.0,
                            # "payload" matches the schema Qdrant-backed retrievers use
                            # (vector_database.py) so evaluate_retrieval() works unmodified.
                            "payload": {
                                "chunk_id": chunk["chunk_id"],
                                "pdf_name": chunk["pdf_name"],
                                "page_numbers": chunk.get("page_numbers"),
                                "uid": uid,
                            },
                        })
        return results[:top_k]

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
