# DAT560 Project — Multimodal RAG on MMDocIR

A three-system Retrieval-Augmented Generation pipeline evaluated on the [MMDocIR](https://github.com/MMDocIR/MMDocIR) benchmark (750 queries over 131 documents). Built for DAT 560 at the University of Stavanger.

---

## Table of Contents

- [Systems Overview](#systems-overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Data](#data)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Reproducibility](#reproducibility)

---

## Systems Overview

| System | Type | Description |
|--------|------|-------------|
| **System 1 — Baseline RAG** | Text-only | Jina CLIP v2 embeddings, Qdrant vector store, hybrid BM25 + dense retrieval, Ollama generation |
| **System 2 — Advanced mRAG** | Multimodal | Adds page image, figure, and evidence crop retrieval; 8 query techniques; 5 chunking strategies; 5 prompting strategies |
| **System 3 — Agentic mRAG** | Agent-based | LangGraph StateGraph with query rewriter, grader, and generator agents; adaptive retry on low-confidence retrievals |

### Architecture

```
PDFs + Page Images + Figures
         │
         ▼  (one-time preprocessing)
  pdf_loader → pdf_chunker → embedder → Qdrant
                                          │
         ┌────────────────────────────────┘
         ▼
    User Query
         │
         ▼
  Query Technique  (standard / multi_query / rag_fusion / hyde / step_back / ...)
         │
         ▼
  Retrieved Chunks ± Images
         │
         ▼
  Prompt Strategy  (standard / cot / few_shot / role / ensemble)
         │
         ▼
  LLM via Ollama  (qwen3-vl:8b-instruct, school-provided)
         │
         ▼
     Answer + Metrics
```

System 3 replaces the fixed pipeline with a LangGraph agent loop:

```
Query → [query_rewriter] → [grader] ──(low confidence)──→ retry
                                │
                          (sufficient confidence)
                                ▼
                         [generator] → Answer
```

---

## Project Structure

```
.
├── README.md
├── requirements.txt
├── info/
│   ├── hours.csv
│   ├── mRAG.md               # Task specification
│   └── summary.ipynb
└── src/
    ├── main.py               # Entry point: System 1 & 2
    ├── main_agentic.py       # Entry point: System 3
    ├── .env.example
    ├── docker-compose.yml    # Optional: Qdrant via Docker
    ├── config/
    │   └── config.py         # BaselineConfig, AdvancedConfig, AgenticConfig
    ├── data/
    │   ├── train/
    │   ├── test/
    │   └── preprocessed/     # Cached chunk files (preprocessed-generated files)
    ├── preprocessing/
    │   ├── pdf_loader.py     # Docling-based PDF extraction
    │   ├── pdf_chunker.py    # 5 chunking strategies
    │   ├── image_processor.py
    │   └── extract_figures.py
    ├── indexing/
    │   ├── embedder.py       # Jina CLIP v2 (text + image, 1024D)
    │   ├── vector_database.py
    │   └── hybrid_retriever.py
    ├── query_techniques/     # standard, multi_query, rag_fusion, hyde, step_back, ...
    ├── retrieval_techniques/
    │   └── multimodal.py     # Image-aware retrieval routing
    ├── generation/
    │   ├── generator.py      # BaselineGenerator + VisionGenerator
    │   └── prompts/          # standard, cot, few_shot, role, ensemble
    ├── pipelines/
    │   ├── base_pipeline.py
    │   ├── baseline_pipeline.py
    │   ├── advanced_pipeline.py
    │   ├── agentic_pipeline.py
    │   └── preprocessing_pipeline.py
    ├── agentic/
    │   ├── llm.py
    │   ├── graph/            # LangGraph state, nodes, builder
    │   └── tools/
    ├── evaluation/
    │   ├── retrieval_metrics.py
    │   └── generation_metrics.py
    ├── utils/
    │   └── timer.py
    ├── notebooks/            # Exploration and debugging notebooks
    └── results/              # CSV output from experiments
```

---

## Setup

### Prerequisites

- Python 3.11
- Ollama access provided by the school (no local install required)

### Install

```bash
# Create and activate environment
conda create --name dat560project python=3.11
conda activate dat560project

# Install dependencies
pip install -r requirements.txt

# GPU support (non-Mac only)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Environment variables

Copy the example file and fill in your credentials:

```bash
cp src/.env.example src/.env
```

```env
OLLAMA_API_KEY=your_key_here      # provided by school
```

---

## Data

Download the MMDocIR subset and place it under `src/data/`:

```
src/data/
├── train/
│   ├── pdfs_train/
│   ├── page_images_train/
│   ├── images_train/
│   └── train.jsonl
└── test/
    ├── pdfs_test/
    ├── page_images_test/
    ├── images_test/
    └── test.jsonl
```

---

## Running the Pipeline

### Step 1 — Preprocessing (run once)

Extracts text and images from PDFs, applies all chunking strategies, and builds Qdrant indexes.

```bash
cd src

# Full run: PDF extraction + chunking + build all multimodal indexes
python -m pipelines.preprocessing_pipeline
```

### Running preprocessing on Google Colab

Upload the cloned project to Google Drive, then mount and run:

<details>
<summary>Click to expand Colab setup script</summary>

```python
from google.colab import drive

try:
    drive.flush_and_unmount()
    print("Drive unmounted successfully (or was not mounted).")
except ValueError:
    print("Drive was not mounted or could not be unmounted, proceeding with mount attempt.")

print("Attempting to mount Google Drive...")
drive.mount('/content/drive')

import os
project_path = '/content/drive/My Drive/DAT560project'

if os.path.exists(project_path) and os.path.isdir(project_path):
    print(f"Contents of {project_path}:")
    print(os.listdir(project_path))

    # Uncomment to install requirements:
    # requirements_path = os.path.join(project_path, 'requirements.txt')
    # if os.path.exists(requirements_path):
    #     get_ipython().system(f'pip install -r "{requirements_path}"')

    script_path = os.path.join(project_path, 'src', 'pipelines', 'preprocessing_pipeline.py')

    if os.path.exists(script_path):
        get_ipython().system(f'python "{script_path}"')
    else:
        print(f"The script '{script_path}' does not exist. Please check the path.")
else:
    print(f"The folder '{project_path}' does not exist or is not a directory after mounting.")
    print("Please verify the exact path of your project folder and ensure it's shared correctly.")
```

</details>

---

### Step 2 — Run Systems 1 & 2

```bash
cd src

# Run with default config (enhanced_hierarchical + rag_fusion + cot + multimodal)
python main.py

# Override query technique and prompting strategy
python main.py --technique multi_query --prompting-strategy cot

# Run all ablation experiments
python main.py --run-experiments

# Evaluate on a larger subset
python main.py --eval-subset 50

# Force rebuild the Qdrant index
python main.py --force-rebuild
```

Results are saved to `src/results/pipeline_results.csv`.

---

### Step 3 — Run System 3 (Agentic)

```bash
cd src

# Test with a single query
python main_agentic.py --test-query "What is the revenue growth in Q3?"

# Full evaluation
python main_agentic.py --eval --eval-size 20 --output results_agentic.json

# Rebuild index before running
python main_agentic.py --rebuild-index --eval --eval-size 10
```

The agentic system prints each agent's decision (technique chosen, grader confidence, prompting strategy) per query.

---

## Configuration

All settings live in `src/config/config.py`. Three config classes are available: `BaselineConfig` (System 1), `AdvancedConfig` (System 2), `AgenticConfig` (System 3).

| Parameter | Default | Options |
|-----------|---------|---------|
| `CHUNKING_STRATEGY` | `fixed_size` | `fixed_size`, `sliding_window`, `semantic`, `hierarchical`, `enhanced_hierarchical` |
| `QUERY_TECHNIQUE` | `standard` | `standard`, `multi_query`, `rag_fusion`, `step_back`, `hyde`, `query_rewriting`, `query_expansion`, `query_decomposition` |
| `PROMPTING_STRATEGY` | `standard` | `standard`, `cot`, `few_shot`, `role`, `ensemble` |
| `USE_MULTIMODAL` | `True` | Enable multimodal retrieval (page images, figures, evidence crops) |
| `TOP_K` | `5` | Number of retrieved chunks per query |
| `EVAL_SUBSET_SIZE` | `20` | Number of test questions to evaluate |
| `VECTOR_DB_MODE` | `local` | `local` (disk), `memory` (RAM), `docker` (remote Qdrant) |
| `LLM_MODEL` | `qwen3-vl:8b-instruct` | Any model available via Ollama |
| `LLM_TEMPERATURE` | `0.0` | Generation temperature (reduce hallucination) |

---

## Evaluation Metrics

### Retrieval

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of retrieved docs that are relevant |
| Recall@k | Fraction of relevant docs that are retrieved |
| MAP | Mean Average Precision across all queries |
| MRR | Mean Reciprocal Rank of the first relevant result |
| NDCG@k | Normalized Discounted Cumulative Gain |

### Generation

| Metric | Description |
|--------|-------------|
| Exact Match | Normalized string equality with ground truth |
| Token F1 | SQuAD-style token overlap between prediction and answer |
| Semantic Similarity | Embedding cosine similarity between prediction and answer |

---

## Reproducibility

| Mechanism | Detail |
|-----------|--------|
| Prompt templates | Versioned in `src/generation/prompts/` |
| Preprocessing artifacts | Chunk files cached in `src/data/preprocessed/` and reused |
| Index reuse | Qdrant collections reused unless `--force-rebuild` is passed |
| Deterministic generation | `LLM_TEMPERATURE = 0.0`, `LLM_TOP_P = 0.1` |
| Timing logs | Preprocessing and experiment runtimes saved to `src/results/` |

---

## Remove Environment

```bash
conda remove --name dat560project --all
```
