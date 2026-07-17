# src/indexing/kg_retriever.py
# Graph-traversal retriever: given a natural-language query, find matching entity
# nodes in the graph, expand their neighborhood via BFS, then collect the source
# chunks attached to the visited nodes as context for generation.
import logging
from typing import Any, Dict, List, Set

import networkx as nx

logger = logging.getLogger(__name__)


class KGRetriever:
    """
    Retrieval via entity matching + BFS neighborhood expansion.

    Steps:
    1. Scan graph nodes whose name appears as a substring of the query.
    2. Expand from those seed nodes for `hops` BFS hops (both directions).
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
    ):
        self.graph = graph
        self.hops = hops
        self.min_seed_length = min_seed_length

        # UID → chunk dict for O(1) lookup
        self.chunk_index: Dict[str, Dict] = {
            f"{c['pdf_name']}::{c['chunk_id']}": c for c in chunks
        }

        # Sort entities longest-first so more specific names match before substrings.
        # Drop entities shorter than min_seed_length: single/double-character node
        # labels are near-always extraction noise ("N", "y", "RE"), not real entities,
        # and match as false-positive seeds in almost every query.
        self.entities: List[str] = sorted(
            (e for e in graph.nodes() if len(e) >= min_seed_length),
            key=len,
            reverse=True,
        )

    # ------------------------------------------------------------------
    def _find_seeds(self, query: str) -> List[str]:
        query_lower = query.lower()
        return [e for e in self.entities if e.lower() in query_lower]

    def _bfs_expand(self, seeds: List[str]) -> List[Set[str]]:
        """Return a list of node sets per hop: [seed_nodes, hop1_nodes, hop2_nodes, ...]
        so callers can prioritise chunks from nodes closer to the query."""
        layers: List[Set[str]] = [set(seeds)]
        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)
        for _ in range(self.hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                if node not in self.graph:
                    continue
                neighbors = (
                    set(self.graph.successors(node))
                    | set(self.graph.predecessors(node))
                )
                next_frontier.update(neighbors - visited)
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
