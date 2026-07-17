# src/pipelines/kg_preprocessing_pipeline.py
# Standalone script to build the knowledge graph from preprocessed chunks.
# Designed to run either locally or in Google Colab:
#
#   Local:
#     DAT560project/src> python -m pipelines.kg_preprocessing_pipeline
#
#   Colab:
#     1. Mount Google Drive
#     2. Set DRIVE_OUTPUT_PATH to your Drive folder
#     3. Run the cells — the graph JSON is written to Drive automatically
#
# The output file (knowledge_graph.json) contains both the NetworkX graph and
# the raw chunks, so the KGPipeline can load it on any machine without re-building.
import logging
import os
import shutil
from pathlib import Path

from config.config import KGConfig
from data.chunk_loader import load_preprocessed_chunks, print_chunk_statistics
from indexing.kg_builder import KGBuilder
from indexing.kg_store import KGStore
from utils.timer import MetricsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Colab: set this to your mounted Drive path to auto-copy the output ──────
# Example: "/content/drive/MyDrive/DAT560/knowledge_graph.json"
DRIVE_OUTPUT_PATH = os.environ.get("KG_DRIVE_OUTPUT_PATH", "")


def build_kg_pipeline(force_rebuild: bool = True):
    """
    Load preprocessed chunks → extract triples → build graph → save JSON.

    Args:
        force_rebuild: If False and the graph file already exists, skip building.
    """
    config = KGConfig()
    tracker = MetricsTracker(logger)
    log_and_time = tracker.log_and_time

    kg_path = Path(config.KG_GRAPH_FILE)

    if not force_rebuild and kg_path.exists():
        logger.info("Graph already exists at %s — skipping build (use force_rebuild=True to override)", kg_path)
        return {"status": "skipped", "path": str(kg_path)}

    with log_and_time("Total KG Build Runtime"):

        with log_and_time("Loading preprocessed chunks"):
            chunks = load_preprocessed_chunks(Path(config.PREPROCESSED_CHUNKS_FILE))
            print_chunk_statistics(chunks)

        with log_and_time("Extracting triples + building graph"):
            builder = KGBuilder(
                base_url=config.OLLAMA_BASE_URL,
                model=config.KG_EXTRACTION_MODEL,
                api_key=config.OLLAMA_API_KEY,
            )
            graph = builder.build_graph(chunks)

        with log_and_time(f"Saving graph to {kg_path}"):
            KGStore.save(graph, chunks, kg_path)

    tracker.print_timing_summary()

    # ── Optional: copy to Google Drive ──────────────────────────────────────
    if DRIVE_OUTPUT_PATH:
        drive_path = Path(DRIVE_OUTPUT_PATH)
        drive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(kg_path, drive_path)
        logger.info("Graph copied to Google Drive: %s", drive_path)

    return {
        "status": "success",
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "chunks": len(chunks),
        "path": str(kg_path),
    }


if __name__ == "__main__":
    result = build_kg_pipeline(force_rebuild=True)
    print(f"\nKG build complete: {result['status']}")
    print(f"  Nodes : {result.get('nodes', 'N/A')}")
    print(f"  Edges : {result.get('edges', 'N/A')}")
    print(f"  Chunks: {result.get('chunks', 'N/A')}")
    print(f"  File  : {result.get('path', 'N/A')}")
