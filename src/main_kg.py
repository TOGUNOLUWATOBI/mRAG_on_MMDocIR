"""
System 4: Knowledge-Graph RAG Pipeline Main Entry Point

Retrieval is entity-substring seed matching + BFS graph traversal over a
NetworkX knowledge graph (built once via KGBuilder from chunks_fixed_size.json),
in place of BM25/dense vector search.

Usage:
    python src/main_kg.py --test-query "What is X?"
    python src/main_kg.py --eval --eval-size 20
    python src/main_kg.py --eval --eval-size 20 --rebuild-index
"""

import sys
import os
import logging
import json
import time
from pathlib import Path

import argparse

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config.config import KGConfig
from data.data_loader import load_train_data
from pipelines.kg_pipeline import KGPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for System 4 KG pipeline."""

    parser = argparse.ArgumentParser(
        description="Run System 4 Knowledge-Graph RAG Pipeline"
    )

    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Test with a single query"
    )

    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run full evaluation"
    )

    parser.add_argument(
        "--eval-size",
        type=int,
        default=5,
        help="Number of test queries to evaluate (default: 5)"
    )

    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the knowledge graph (slow — re-extracts triples via LLM)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )

    args = parser.parse_args()

    print("="*80)
    print("SYSTEM 4: KNOWLEDGE-GRAPH RAG PIPELINE")
    print("="*80)

    print("Loading configuration...")
    config = KGConfig()
    config.EVAL_SUBSET_SIZE = args.eval_size

    print("Initializing KG pipeline...")
    pipeline = KGPipeline(config)

    print("Building/loading index...")
    start = time.time()
    pipeline.build_index(force_rebuild=args.rebuild_index)
    index_time = time.time() - start
    print(f"Index ready in {index_time:.2f}s")

    print("Initializing components...")
    pipeline.initialize_components()

    results = None

    if args.test_query:
        print(f"\nTesting with query: {args.test_query}")
        result = pipeline.run_query(args.test_query)
        results = {"single_query": result}

        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")
        print(f"\nDocs Retrieved: {len(result['retrieved_docs'])}")
        print(f"Timing: {result['timing']}")

    elif args.eval:
        print(f"\nRunning evaluation on {args.eval_size} queries...")

        test_data = load_train_data(config.TEST_JSONL)
        eval_summary = pipeline.evaluate(test_data)

        results = eval_summary

        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(json.dumps(eval_summary, indent=2))

    else:
        print("\nNo query or eval specified. Running example query...")
        sample_query = "How many students of NTU would recommend studying at NTU?"
        result = pipeline.run_query(sample_query)
        results = {"sample_query": result}

        print("\n" + "="*80)
        print("SAMPLE RESULT")
        print("="*80)
        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")

    if args.output and results:
        print(f"Saving results to {args.output}")
        with open(args.output, 'w') as f:
            def default_handler(obj):
                if hasattr(obj, '__dict__'):
                    return obj.__dict__
                return str(obj)

            json.dump(results, f, indent=2, default=default_handler)

    print("\nPipeline completed!")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
