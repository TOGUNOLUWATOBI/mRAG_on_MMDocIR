# src/indexing/kg_store.py
# Persist and reload the knowledge graph + raw chunks as a single JSON file.
# The file can be copied to / from Google Drive for the Colab workflow.
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
from networkx.readwrite import json_graph

logger = logging.getLogger(__name__)


class KGStore:
    """Save and load a (graph, chunks) pair to/from a JSON file."""

    @staticmethod
    def save(
        graph: nx.DiGraph,
        chunks: List[Dict[str, Any]],
        path,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "graph": json_graph.node_link_data(graph),
            "chunks": chunks,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        size_kb = path.stat().st_size / 1024
        logger.info("Knowledge graph saved → %s (%.1f KB)", path, size_kb)

    @staticmethod
    def load(path) -> Tuple[nx.DiGraph, List[Dict[str, Any]]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge graph file not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # node_link_graph changed its default edge-key name between NX 2.x and 3.x;
        # try both to stay compatible across environments (local vs Colab).
        try:
            graph = json_graph.node_link_graph(data["graph"], directed=True)
        except Exception:
            graph = json_graph.node_link_graph(
                data["graph"], directed=True, edges="links"
            )

        chunks = data["chunks"]
        logger.info(
            "Knowledge graph loaded: %d nodes, %d edges, %d chunks",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(chunks),
        )
        return graph, chunks
