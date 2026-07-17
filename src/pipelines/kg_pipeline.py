# src/pipelines/kg_pipeline.py
# RAG pipeline backed by a NetworkX knowledge graph instead of a vector database.
# Drop-in replacement for BaselineRAGPipeline / AdvancedRAGPipeline:
# same build_index / retrieve / run_query / evaluate interface.
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.config import KGConfig
from data.chunk_loader import load_preprocessed_chunks
from evaluation.generation_metrics import evaluate_generation
from evaluation.retrieval_metrics import evaluate_retrieval
from generation.answer_validator import AnswerValidator
from generation.generator import BaselineGenerator
from generation.prompts.standard import StandardPromptStrategy
from indexing.kg_builder import KGBuilder
from indexing.kg_retriever import KGRetriever
from indexing.kg_store import KGStore

logger = logging.getLogger(__name__)


class KGPipeline:
    """
    Knowledge-graph RAG pipeline.

    Index building:
      - On first run (or force_rebuild=True): loads preprocessed chunks,
        calls LLM to extract triples, builds a NetworkX DiGraph, saves to JSON.
      - On subsequent runs: loads the saved JSON directly (fast).

    Retrieval:
      - Entity-substring matching + BFS graph traversal → source chunks.
    """

    def __init__(self, config=None):
        self.config = config or KGConfig()
        self.graph = None
        self.chunks: Optional[List[Dict]] = None
        self.kg_retriever: Optional[KGRetriever] = None
        self.generator: Optional[BaselineGenerator] = None
        self.prompt_strategy = None

    # ------------------------------------------------------------------
    def initialize_components(self):
        self.generator = BaselineGenerator(
            base_url=self.config.OLLAMA_BASE_URL,
            model=self.config.LLM_MODEL,
            api_key=self.config.OLLAMA_API_KEY,
            temperature=self.config.LLM_TEMPERATURE,
            top_p=self.config.LLM_TOP_P,
            max_tokens=self.config.LLM_MAX_TOKENS,
            max_retries=self.config.LLM_MAX_RETRIES,
            retry_delay=self.config.LLM_RETRY_DELAY,
        )
        self.prompt_strategy = StandardPromptStrategy(generator=self.generator)

    # ------------------------------------------------------------------
    def build_index(self, force_rebuild: bool = False):
        """Load an existing graph JSON or build a new one from preprocessed chunks."""
        kg_path = Path(self.config.KG_GRAPH_FILE)

        if not force_rebuild and kg_path.exists():
            logger.info("Loading existing knowledge graph from %s", kg_path)
            self.graph, self.chunks = KGStore.load(kg_path)
        else:
            logger.info("Building knowledge graph from preprocessed chunks...")
            chunks_path = Path(self.config.PREPROCESSED_CHUNKS_FILE)
            self.chunks = load_preprocessed_chunks(chunks_path)

            builder = KGBuilder(
                base_url=self.config.OLLAMA_BASE_URL,
                model=self.config.KG_EXTRACTION_MODEL,
                api_key=self.config.OLLAMA_API_KEY,
            )
            self.graph = builder.build_graph(self.chunks)
            KGStore.save(self.graph, self.chunks, kg_path)

        self.kg_retriever = KGRetriever(
            graph=self.graph,
            chunks=self.chunks,
            hops=self.config.KG_HOPS,
            min_seed_length=self.config.KG_MIN_SEED_LENGTH,
            max_fanout=self.config.KG_MAX_FANOUT,
        )
        logger.info("Knowledge graph index ready.")

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if self.kg_retriever is None:
            raise RuntimeError("Call build_index() before retrieve().")
        return self.kg_retriever.retrieve(question, top_k or self.config.TOP_K)

    # ------------------------------------------------------------------
    def run_query(self, question: str, top_k: int = None, **kwargs) -> Dict[str, Any]:
        top_k = top_k or self.config.TOP_K
        query_start = time.time()

        retrieval_start = time.time()
        retrieved = self.retrieve(question, top_k)
        retrieval_time = time.time() - retrieval_start

        context = "\n\n".join(
            f"[Document {i+1}]:\n{r['text']}" for i, r in enumerate(retrieved)
        )

        generation_start = time.time()
        system_prompt = self.prompt_strategy.get_system_prompt()
        answer = self.generator.generate(question, context, system_prompt)
        generation_time = time.time() - generation_start

        return {
            "question": question,
            "retrieved_docs": retrieved,
            "context": context,
            "answer": answer,
            "timing": {
                "retrieval": retrieval_time,
                "generation": generation_time,
                "total": time.time() - query_start,
            },
        }

    # ------------------------------------------------------------------
    def evaluate(self, test_data: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Parallel evaluation — mirrors BaseRAGPipeline.evaluate()."""
        subset_size = getattr(self.config, "EVAL_SUBSET_SIZE", len(test_data))
        test_subset = test_data[:subset_size]
        logger.info("Evaluating on %d queries...", len(test_subset))

        eval_start = time.time()
        all_retrieved = [None] * len(test_subset)
        all_predictions = [None] * len(test_subset)
        all_ground_truths = [None] * len(test_subset)

        workers = getattr(self.config, "GENERATION_WORKERS", 2)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(self.run_query, record["question"], record=record, **kwargs): (i, record)
                for i, record in enumerate(test_subset)
            }
            for future in as_completed(futures):
                i, record = futures[future]
                result = future.result()
                all_retrieved[i] = result["retrieved_docs"]
                all_predictions[i] = result["answer"]
                all_ground_truths[i] = record["answer"]

        validate_enabled = getattr(self.config, "VALIDATE_ANSWER_FORMAT", False)
        if validate_enabled:
            validated = [
                AnswerValidator.validate(pred, rec["question"], gt)[1]
                for pred, rec, gt in zip(all_predictions, test_subset, all_ground_truths)
            ]
            raw_preds = all_predictions
            preds_for_metrics = validated
        else:
            raw_preds = None
            preds_for_metrics = all_predictions

        retrieval_metrics = evaluate_retrieval(all_retrieved, test_subset)
        generation_metrics = evaluate_generation(
            predictions=preds_for_metrics,
            ground_truths=all_ground_truths,
            embedder=None,
            raw_predictions=raw_preds,
        )

        return {
            "Pipeline": f"{kwargs.get('experiment_name', 'KG')} ({self.__class__.__name__})",
            "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Test Size": len(test_subset),
            "LLM Model": self.config.LLM_MODEL,
            "Retrieval": "KG-traversal",
            "Total time (s)": round(time.time() - eval_start, 2),
            **{k: round(v, 4) for k, v in retrieval_metrics.items()},
            **{k: round(v, 4) for k, v in generation_metrics.items()},
        }
