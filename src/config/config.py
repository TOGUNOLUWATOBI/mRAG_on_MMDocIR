# src/config/config.py
# Configuration for the all systems to share a single source of truth
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List
import os
from dotenv import load_dotenv

# Load environment variables from .env file for API keys
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Define paths outside the dataclass for cleaner referencing
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env", override=True)

SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = SRC_DIR / "data"
RESULTS_DIR = SRC_DIR / "results"
PREPROCESSING_TIME_CSV = RESULTS_DIR / "time_preprocessing.csv"
PIPELINE_TIME_CSV = RESULTS_DIR / "pipeline_time.csv"
RESULTS_CSV = RESULTS_DIR / "pipeline_results.csv"

PDFS_DIR = DATA_DIR / "train" / "pdfs_train"
PREPROCESSED_DATA_DIR = DATA_DIR / "preprocessed"
PREPROCESSED_DOCUMENTS_FILE = DATA_DIR / "preprocessed" / "all_documents.json"

TRAIN_JSONL = DATA_DIR / "train" / "train.jsonl"
TEST_JSONL = DATA_DIR / "test" / "test.jsonl"

PAGE_IMAGES_TRAIN_DIR = DATA_DIR / "train" / "page_images_train"
IMAGES_TRAIN_DIR = DATA_DIR / "train" / "images_train"
PAGE_IMAGES_TEST_DIR = DATA_DIR / "test" / "page_images_test"
IMAGES_TEST_DIR = DATA_DIR / "test" / "images_test"

CACHE_DIR = SRC_DIR / "cache"
CACHE_DB_PATH = CACHE_DIR / "query_cache.db"
PREPROCESSED_CHUNKS_FILE = SRC_DIR / "data" / "preprocessed" / "chunks_fixed_size.json"


@dataclass
class BaselineConfig:
    """Configuration matching the baseline pipeline."""

    HF_TOKEN: str = os.getenv("HF_TOKEN")

    # ===== PATHS =====
    PROJECT_ROOT: Path = PROJECT_ROOT
    SRC_DIR: Path = SRC_DIR
    DATA_DIR: Path = DATA_DIR

    PDFS_DIR: Path = PDFS_DIR
    PREPROCESSED_DATA_DIR: Path = PREPROCESSED_DATA_DIR
    PREPROCESSED_DOCUMENTS_FILE: str = str(PREPROCESSED_DOCUMENTS_FILE)

    TRAIN_JSONL: Path = TRAIN_JSONL
    TEST_JSONL: Path = TEST_JSONL

    CACHE_DIR: Path = CACHE_DIR
    CACHE_DB_PATH: Path = CACHE_DB_PATH
    RESULTS_CSV: Path = RESULTS_CSV
    PREPROCESSING_TIME_CSV: Path = PREPROCESSING_TIME_CSV

    PREPROCESSED_CHUNKS_FILE: str = str(PREPROCESSED_CHUNKS_FILE)

    # ===== EMBEDDING SETTINGS =====
    EMBEDDING_MODEL: str = "jinaai/jina-clip-v2"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 64

    # ===== LLM / GENERATOR SETTINGS =====
    LLM_MODEL: str = "qwen3-vl:8b-instruct"
    OLLAMA_BASE_URL: str = "https://ollama.ux.uis.no"
    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "")
    """API key for Ollama authentication (loaded from .env, empty string if not found)"""
    LLM_TEMPERATURE: float = 0.0
    LLM_TOP_P: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_MAX_RETRIES: int = 4
    LLM_RETRY_DELAY: int = 30

    # ===== VECTOR DATABASE SETTINGS =====
    VECTOR_DB_MODE: str = "local"
    VECTOR_DB_PATH: str = str(SRC_DIR / "local_qdrant")
    VECTOR_DB_COLLECTION: str = "advanced_fixed_size_2000"
    VECTOR_DB_DISTANCE: str = "COSINE"

    # ===== RETRIEVAL SETTINGS =====
    TOP_K: int = 5
    USE_HYBRID_RETRIEVAL: bool = True

    # ===== CHUNKING SETTINGS =====
    USE_PREPROCESSED_CHUNKS: bool = True
    CHUNKING_STRATEGY: str = "fixed_size"
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 200
    CONTEXT_WINDOW: int = 0  # adjacent chunks to prepend/append at retrieval time

    # ===== EVALUATION SETTINGS =====
    EVAL_SUBSET_SIZE: int = 150
    RETRIEVAL_WORKERS: int = 4
    GENERATION_WORKERS: int = 2

    RANDOM_SEED: int = 42

    def __post_init__(self):
        """Validate paths after initialization."""
        if isinstance(self.VECTOR_DB_PATH, Path):
            self.VECTOR_DB_PATH = str(self.VECTOR_DB_PATH)
        if isinstance(self.PREPROCESSED_CHUNKS_FILE, str):
            self.PREPROCESSED_CHUNKS_FILE = Path(self.PREPROCESSED_CHUNKS_FILE)

    @classmethod
    def load_from_dict(cls, config_dict: Dict[str, Any]):
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_fields}
        return cls(**filtered_dict)


@dataclass
class AdvancedConfig(BaselineConfig):
    """Configuration for the Advanced RAG Pipeline."""

    # ===== ADVANCED APP OVERRIDES =====
    # override the chunk file to use the semantic chunks instead of the fixed-size ones
    PREPROCESSED_CHUNKS_FILE: str = str(
        SRC_DIR / "data" / "preprocessed" / "chunks_fixed_size_2000.json"
    )

    VECTOR_DB_COLLECTION: str = "advanced_fixed_size_2000"
    """Separate collection from baseline so the two don't interfere"""

    # Use one model for everything — no server model swapping = no OOM crashes
    LLM_MODEL: str = "qwen3-vl:8b-instruct"  # "qwen3-vl:8b"

    # ===== MULTIMODAL SETTINGS =====
    USE_MULTIMODAL: bool = True

    MAX_VLM_IMAGES: int = 2
    """Cap images sent to VLM to avoid OOM"""

    VLM_MODEL: str = "qwen3-vl:8b-instruct"
    """Vision-language model used when image chunks are retrieved"""

    # Sequential — keeps logs readable and avoids concurrent calls on shared GPU
    GENERATION_WORKERS: int = 1
    RETRIEVAL_WORKERS: int = 1

    # ===== IMAGE RESIZING SETTINGS =====
    MAX_IMAGE_WIDTH: int = 1024
    """Maximum width in pixels for image resizing (0 to disable resizing)"""
    MAX_IMAGE_HEIGHT: int = 1024
    """Maximum height in pixels for image resizing (0 to disable resizing)"""
    IMAGE_RESIZE_QUALITY: int = 85
    """JPEG compression quality (0-100, higher = better quality but larger file size)"""
    VLM_USE_RAW_CHATML: bool = True
    """Use raw ChatML with /no_think to bypass qwen3-vl:8b thinking bug (ignores think=false in API)"""

    PAGE_IMAGES_TRAIN_DIR: Path = PAGE_IMAGES_TRAIN_DIR
    IMAGES_TRAIN_DIR: Path = IMAGES_TRAIN_DIR
    PAGE_IMAGES_TEST_DIR: Path = PAGE_IMAGES_TEST_DIR
    IMAGES_TEST_DIR: Path = IMAGES_TEST_DIR

    FIGURES_TRAIN_DIR: Path = DATA_DIR / "train" / "figures_train"

    # ===== RETRIEVAL FILTER =====
    ALLOWED_CHUNK_TYPES: List[str] = field(
        default_factory=lambda: ["text", "page_image", "figure", "evidence"]
    )

    # ===== QUERY TECHNIQUE SETTINGS =====
    QUERY_TECHNIQUE: str = "standard"
    QUERY_TECHNIQUE_CONFIG: Dict[str, Any] = field(
        default_factory=lambda: {
            "num_variants": 3,
            "max_page_images": 1,
        }
    )

    # ===== PROMPTING STRATEGY SETTINGS =====
    PROMPTING_STRATEGY: str = "standard"
    """
    Prompting strategy for answer generation:
    - 'standard': Direct extraction without special prompting
    - 'few_shot': Provide multiple examples (2-5), then ask question
    - 'role': Assign expert role to LLM (financial_analyst, researcher, etc.)
    - 'cot': Chain-of-Thought - explicit step-by-step reasoning
    - 'ensemble': Multiple strategies with voting/consensus
    """

    PROMPTING_STRATEGY_CONFIG: Dict[str, Any] = field(
        default_factory=lambda: {
            # Role strategy
            "role_type": "financial_analyst_role",
            # CoT strategy
            "show_reasoning": False,  # set to False to hide reasoning
            # Ensemble strategy
            "mode": "multi_prompt",  # 'multi_prompt' or 'self_consistency'
            "ensemble_size": 3,
            "aggregation_method": "embedding_similarity",  # 'judge', 'combine', 'embedding_similarity'
            "strategies": ["standard", "cot", "few_shot", "financial_analyst_role"],
            "include_strategy_metadata": False,
            "verbose_logging": False,  # Enable detailed ensemble logging
            "temperatures": {
                "standard": 0.5,
                "cot": 0.6,
                "few_shot": 0.5,
                "financial_analyst_role": 0.6,
            },
        }
    )
    """Configuration dict for the selected prompting strategy"""

    # ===== ANSWER VALIDATION SETTINGS =====
    VALIDATE_ANSWER_FORMAT: bool = True
    """Enable answer format validation for string-comparison metrics (exact_match, contains_match, token_f1)"""

@dataclass
class KGConfig(BaselineConfig):
    """Configuration for the Knowledge Graph RAG Pipeline."""

    # Path where the serialized graph + chunks JSON will be saved/loaded
    KG_GRAPH_FILE: str = str(DATA_DIR / "preprocessed" / "knowledge_graph.json")

    # Number of BFS hops to expand from seed entities during retrieval
    KG_HOPS: int = 2

    # Minimum character length for an entity to be usable as a seed match.
    # Extraction produces some single/double-character node labels ("N", "RE") that
    # are noise, not real entities — they false-positive-match almost every query.
    KG_MIN_SEED_LENGTH: int = 3

    # Cap on neighbors expanded per node per BFS hop, keeping only the best-evidenced
    # ones. Without this, generic high-degree hub nodes ("students", degree 200+)
    # flood retrieval results with unrelated chunks within a single hop.
    KG_MAX_FANOUT: int = 15

    # Model used for triple extraction — text-only (not the VL model).
    # llama3:8b is fast (~3-4s/chunk) and reliable for JSON extraction.
    # Switch to "qwen3:32b" for higher-quality triples at the cost of speed.
    KG_EXTRACTION_MODEL: str = "llama3:8b"

    # KG uses graph traversal — hybrid BM25+dense retrieval is not applicable
    USE_HYBRID_RETRIEVAL: bool = False


@dataclass
class HybridConfig(KGConfig):
    """Configuration for the Hybrid KG + Dense + BM25 RAG Pipeline (System 5)."""

    # Use the already-populated collection built from the same chunks_fixed_size.json
    # the KG graph was built from, so the three signals are over identical chunks.
    VECTOR_DB_COLLECTION: str = "advanced_fixed_size"

    # Reciprocal Rank Fusion constant — higher values flatten rank differences
    HYBRID_RRF_K: int = 60

    # Each signal fetches top_k * multiplier candidates before fusion
    HYBRID_CANDIDATE_MULTIPLIER: int = 10

    # Per-signal RRF weights. KG traversal is empirically noisier than dense/BM25 on
    # this corpus (see KG_ROADMAP.md Epic 4 findings) — equal weighting (1.0) let KG's
    # noise outvote good dense/BM25 candidates and regressed P@1 from 0.59 (Baseline,
    # no KG) to 0.35. Benchmarked 1.0 vs 0.3 vs 0.15 on the full 150-question set;
    # 0.3 recovered most of that regression (P@1 0.49) without a further benchmark
    # sweep to find an exact optimum — set as the default.
    HYBRID_KG_WEIGHT: float = 0.3
    HYBRID_DENSE_WEIGHT: float = 1.0
    HYBRID_BM25_WEIGHT: float = 1.0


@dataclass
class AgenticConfig(AdvancedConfig):
    """Configuration for System 3 Agentic RAG Pipeline."""
    
    # ===== AGENT SETTINGS =====
    AGENT_MAX_RETRIES: int = 1
    """Maximum number of retry iterations for query rewriting (total attempts = retries + 1)"""
    
    RETRY_ON_LOW_CONFIDENCE: bool = True
    """Whether to retry retrieval if grader confidence is below threshold"""
    
    GRADER_CONFIDENCE_THRESHOLD: float = 0.51
    """Minimum confidence threshold (0.0-1.0) for document relevance grading"""
    
    AGENT_DECISION_LOGGING: bool = True
    """Whether to log all agent decisions (query rewriter, grader, generator) for analysis"""
    
    AGENT_LLM_MODEL: str = "qwen3-vl:8b-instruct"  # Lightweight LLM for agent decisions (Query Rewriter, Grader, Generator strategy)

    # ===== KG ROUTING (Epic 5) =====
    ENABLE_KG_ROUTING: bool = False
    """
    Opt-in: adds 'kg_multihop' as a 9th technique the query-rewriter agent can
    select, routing multi-hop/relational questions through HybridKGRetriever
    (KG traversal + dense + BM25) instead of the standard 8 dense-only
    techniques. Requires the pipeline's chunking config to match the chunks
    the KG graph was built from (chunks_fixed_size.json / advanced_fixed_size),
    or the KG's chunk UIDs won't resolve. Off by default — existing Agentic
    (System 3) behavior/benchmarks are unaffected unless explicitly enabled.
    """
    KG_ROUTING_HYBRID_KG_WEIGHT: float = 0.3
    KG_ROUTING_HYBRID_DENSE_WEIGHT: float = 1.0
    KG_ROUTING_HYBRID_BM25_WEIGHT: float = 1.0