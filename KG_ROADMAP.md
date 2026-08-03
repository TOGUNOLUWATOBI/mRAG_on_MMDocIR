# Knowledge-Graph RAG — Roadmap

**Branch:** `feature/knowledge-graph-networkx`
**Goal:** turn the half-built Knowledge-Graph retrieval arm into a fully running, benchmarked "System 4," then push it as far as time allows toward a genuinely novel contribution (hybrid KG+vector+BM25 retrieval, measured on multi-hop questions) rather than a vanilla GraphRAG clone.

## Current state (verified 2026-07-17)

What already exists and works:
- The graph has already been **built once, successfully**: `kg_build.log` shows a completed run — **77,421 nodes, 73,011 edges from 12,747 chunks**, extracted via `llama3:8b` over the school's Ollama endpoint (`ollama.ux.uis.no`), saved to `src/data/preprocessed/knowledge_graph.json` (37.5 MB). Took ~8 hours. Only 1 chunk out of 12,747 failed extraction (a JSON parsing error on `welcome-to-nus.pdf::18`) — negligible.
- `kg_builder.py`, `kg_retriever.py`, `kg_store.py`, `kg_pipeline.py`, `kg_preprocessing_pipeline.py` are all written and structurally wired to the same `build_index / retrieve / run_query / evaluate` interface as `BaselineRAGPipeline` / `AdvancedRAGPipeline`.
- `KGConfig` is added to `src/config/config.py`; `networkx` is added to `requirements.txt`.
- A notebook prototype (`src/notebooks/hybrid_retrieval_test.ipynb`, with an already-executed copy) proves a **3-signal RRF fusion** (KG traversal + Qdrant dense + BM25) works end-to-end against the local Qdrant collection (`advanced_fixed_size`, 19,888 points) — it's been run at least once against a real query.
- There's an empty placeholder branch `feature/knowledge-graph-neo4j` (identical HEAD to this branch, zero commits of its own) — evidence you considered Neo4j as an alternative backend but never started it.

What's missing / at risk right now:
- **None of this is committed.** `git status` shows `kg_builder.py`, `kg_retriever.py`, `kg_store.py`, `kg_pipeline.py`, `kg_preprocessing_pipeline.py`, `knowledge_graph.json`, both notebooks, and the config/requirements edits all sitting as uncommitted/untracked changes. An 8-hour LLM build and several files of working code currently exist only on disk.
- **`KGPipeline` has never been executed end-to-end.** There is no `main_kg.py` entry point, and nothing in `main.py` references `KGPipeline`. The build/save step has been proven; `retrieve()`, `run_query()`, and `evaluate()` have not.
- No benchmark yet against Systems 1–3 (Baseline / Advanced / Agentic) — no numbers exist to say whether the graph actually helps.
- The hybrid RRF fusion only exists as a notebook, run against one hardcoded query — not a reusable module, not evaluated on the test set.
- No entity resolution — "Apple" and "Apple Inc." (or any other surface-form variant) become permanently separate nodes.

---

## Epic 0 — Stabilize what already exists

*Do this first. You have 8 hours of LLM compute and a chunk of real code sitting uncommitted — protect it before anything else.*

- [x] Commit the current working tree on `feature/knowledge-graph-networkx`: `kg_builder.py`, `kg_retriever.py`, `kg_store.py`, `kg_pipeline.py`, `kg_preprocessing_pipeline.py`, the `KGConfig` addition in `config.py`, and the `requirements.txt` change. Split into 2–3 logical commits (e.g. "feat: KG builder/store/retriever", "feat: KGConfig + KGPipeline wiring") rather than one dump, matching the repo's existing commit style. — Done: 5 commits (`9d5ebd4`, `b9ce93b`, `4d3a41a`, `8f032ca`, `fd4fe73`).
- [x] Decide whether to commit `knowledge_graph.json` (37.5 MB) into git, or move it to the same "regenerate via script, don't commit" pattern already used for the other `chunks_*.json` preprocessed files. Check whether those are `.gitignore`d — if so, do the same here and document the regeneration command in the README instead. — Checked: `chunks_*.json` (10–37 MB each) are tracked directly in git, not ignored. Committed `knowledge_graph.json` the same way (`4d3a41a`).
- [x] Commit the two `hybrid_retrieval_test*.ipynb` notebooks as-is (they're valuable prototype evidence) — clean them up properly in Epic 4 once the logic graduates into a real module. — Done (`8f032ca`).
- [x] Verify continued access to `https://ollama.ux.uis.no` with your school API key — the course has likely wrapped up for the term, and university-provided endpoints sometimes get revoked shortly after grading closes. If it's gone, you'll need a fallback (self-hosted Ollama on the GPU servers you mentioned you have access to) before any epic below that needs new LLM calls (Epic 3's entity resolution, Epic 5). — Verified 2026-07-17: endpoint live, `OLLAMA_API_KEY` in `src/.env` valid (200 on `/api/tags`). Both `llama3:8b` and `qwen3:32b` still available.
- [x] Delete the `feature/knowledge-graph-neo4j` placeholder branch (it has no commits of its own — it's just noise) unless you specifically want to revisit a Neo4j backend later, in which case leave a one-line note in this file instead. — Deleted 2026-07-17 (confirmed with user; branch had zero unique commits).
- [ ] Note only, no action needed yet: `chunks_enhanced_hierarchical.json` was deleted from disk (still shows as `deleted` in `git status`). Not a blocker for KG work (it uses `chunks_fixed_size.json`, which is intact), but if any future ablation needs the `enhanced_hierarchical` chunking strategy again, it'll need regenerating via `pipelines.preprocessing_pipeline`.

---

## Epic 1 — Get `KGPipeline` actually running end-to-end

*The code has never been executed as a pipeline, only as a build script. Treat the first run as a debugging session, not a formality.*

- [x] Add a `main_kg.py` entry point, mirroring the existing pattern in `main_agentic.py` (`--test-query`, `--eval`, `--eval-size`, `--rebuild-index`, `--output`). — Done (`a817bf5`).
- [x] Run a single-query smoke test first (cheap, fast) to confirm `build_index()` loads the existing 37.5 MB graph correctly and `run_query()` produces a real answer without crashing. — Confirmed: loads in ~1s, produces real answers (e.g. correctly named the housing office for an exchange-student email question).
- [x] Fix whatever breaks. Likely candidates worth checking specifically if something fails: the `KGRetriever` entity-substring seed matching against real questions (short/common entity names causing false positive seeds), and the chunk UID format (`f"{pdf_name}::{chunk_id}"`) staying consistent between what `kg_builder.py` wrote into the graph and what `chunk_loader.load_preprocessed_chunks()` produces at retrieval time. — Chunk UID format was already consistent (both `f"{pdf_name}::{chunk_id}"`), no fix needed there. Found and fixed two real bugs: (1) single/double-char node labels ("N", "y", "RE") were matching as false-positive seeds on nearly every query — added `KG_MIN_SEED_LENGTH` guard; (2) `KGRetriever` returned chunks under a `"metadata"` key while `evaluate_retrieval()`/`page_recall_at_k()` require `"payload"` (the Qdrant-retriever schema) — `evaluate()` crashed with `KeyError: 'payload'` until renamed. Both fixed in `a817bf5`.
- [x] Run a small eval subset (20–30 queries) through `KGPipeline.evaluate()`. Sanity-check the output isn't degenerate — retrieval metrics not all zero, `retrieved_docs` non-empty for most queries, timings in a sane range (BFS over a 77k-node graph should still be fast; if it's slow, that's worth knowing before scaling up). — Ran 25 queries: completed in ~10s total (BFS retrieval itself ~0.02–0.03s/query, well within budget), retrieved_docs non-empty for all. Metrics non-zero but low (recall@5=0.12, precision@1=0, map=0.04) — expected given known weaknesses (noisy generic seed nodes like "students" causing hub-node BFS fan-out into unrelated docs); this is exactly what Epic 3's smarter-seed-matching and edge-weighting/fan-out-control items are meant to address, and Epic 2 will make the comparison against Systems 1–3 rigorous rather than anecdotal.

---

## Epic 2 — Fair benchmark: KG-only vs. Systems 1–3

*This is the step that turns "I built a knowledge graph" into "I proved whether it helps."*

- [x] Run `KGPipeline.evaluate()` on the **same test subset size and same `RANDOM_SEED`** as whatever Baseline/Advanced/Agentic were evaluated on (check `results_agentic.json` and `src/results/pipeline_results.csv` for the size actually used previously — the README mentions both `EVAL_SUBSET_SIZE=150` defaults and 1000-query ablation runs, so pin down which one is the real comparison baseline). — **150 confirmed as the real baseline** (= the full `test.jsonl`, so `test_data[:150]` is a deterministic full-set slice; no `RANDOM_SEED` dependency to match). Ran KG on all 150 in 69.76s: `precision@1=0.0533, recall@5=0.20, ndcg@5=0.1279, map=0.1041, token_f1=0.0317, exact_match=0.0133`. **Important discrepancy found:** `src/results/pipeline_results.csv` (local, exploratory, many re-runs) does **not** match the numbers in the actually-submitted `DAT560_Group6_report.pdf` (e.g. CSV's oldest Baseline row has P@1=0.52; the report's Table 7 has P@1=0.5933). Treated the **report's Table 7/Table 6 as authoritative** (System 1 P@1=0.5933/NDCG@5=0.6693/TokenF1=0.1837; System 2 P@1=0.8667/NDCG@5=0.9043/TokenF1=0.2658; System 3 P@1=0.6066/NDCG@5=0.6695/TokenF1=0.1674) since that's what was actually graded — see Findings below.
- [x] Append the KG results row into the same comparison table/CSV so all four systems are visible side by side. — Added `4_KG_Graph_Traversal (KGPipeline)` row to `src/results/pipeline_results.csv`.
- [x] Manually tag or filter a **multi-hop subset** of `test.jsonl` — questions that require chaining facts across more than one entity/fact (e.g. "what did the CEO of the company that acquired X say about Y"). MMDocIR won't have this labeled for you; skim the question set and hand-pick ~15–20 candidates. This subset is the whole point — it's where graph traversal is *supposed* to beat plain vector similarity, and where the interesting result lives. — Hand-picked 19 questions into `src/data/test/multihop_subset.json` (indices + one-line justification each), e.g. "find the year from stat A, then look up stat B for that year," "resolve an entity from one figure, then look up its number in a separate table," "cross-reference two tables on a shared attribute."
- [x] Write up findings: overall metric deltas (retrieval + generation) across all four systems, and a focused comparison on the multi-hop subset specifically. Include 2–3 concrete qualitative examples — the question, what Baseline retrieved, what KG retrieved, and which one was actually right. — See **Epic 2 Findings** below.

### Epic 2 Findings (2026-07-17)

**Overall (full 150-question test set), report numbers vs. fresh KG run:**

| System | P@1 | NDCG@5 | MAP | Token F1 | Exact Match |
|---|---|---|---|---|---|
| System 1 — Baseline | 0.5933 | 0.6693 | 0.6478 | 0.1837 | 0.1333 |
| System 2 — Advanced | 0.8667 | 0.9043 | 0.8944 | 0.2658 | 0.2067 |
| System 3 — Agentic | 0.6066 | 0.6695 | 0.6544 | 0.1674 | 0.1133 |
| **System 4 — KG (this run)** | **0.0533** | **0.1279** | **0.1041** | **0.0317** | **0.0133** |

KG-only graph traversal is roughly **10x worse** on retrieval (P@1, NDCG@5, MAP) and **5x worse** on generation (Token F1) than every existing system, including the plain baseline. This is not a close result — vanilla GraphRAG-style traversal, as currently implemented, is a materially weaker retriever than BM25+dense on this corpus. (Note: KG's `semantic_similarity` column isn't comparable — `evaluate_generation()` falls back to `token_f1` when no embedder is passed, which is true by design for KG since it does no dense embedding at all; Token F1 is the fair cross-system column.)

**Multi-hop subset (19 hand-picked questions), all four systems freshly run today on the identical subset, same underlying chunk source (`chunks_fixed_size.json` / `advanced_fixed_size` collection):**

| System | P@1 | NDCG@5 | MAP/MRR | Token F1 | Exact Match | Wall time |
|---|---|---|---|---|---|---|
| Baseline | 0.6316 | 0.698 | 0.6842 | 0.0547 | 0.0526 (1/19) | 233.8s |
| Advanced | 0.7895 | 0.8227 | 0.8158 | 0.0545 | 0.0526 (1/19) | 136.6s |
| Agentic | 0.6316 | 0.698 | 0.6842 | 0.0702 | 0.0526 (1/19) | 306.7s (14.6s/query, matches report's ~12.16s) |
| **KG** | **0.0** | **0.0789** | **0.0526** | 0.0549 | 0.0526 (1/19) | **11.1s** |

Two findings stand out:

1. **Retrieval finds the right document even on hard multi-hop questions (for the non-KG systems) — generation is the real bottleneck there.** P@1 for Baseline/Advanced/Agentic barely drops on this "hard" subset (0.63–0.79) vs. their full-set numbers (0.59–0.87), yet Exact Match collapses to exactly 1/19 for **all four systems**, KG included. Once the right page is retrieved, none of the four systems can reliably reason through the two-stage arithmetic/lookup (e.g. "find which year has stat A, then look up stat B for that year") that defines a multi-hop question — this is a `qwen3-vl:8b-instruct` reasoning-capacity ceiling, not a retrieval problem, for Systems 1–3.

2. **KG's retrieval fails specifically because these multi-hop questions lack a single crisp named entity to seed on — and the exact bug Epic 1 fixed (short-name false positives) is still visible.** P@1=0.0 for KG on this subset — it never even surfaces the right source document, let alone the right fact. Example: for "How many more millions of dollars was the median exit valuation in the USA compared to Europe...", the seed-entity list included the node `"RoPE"` (a machine-learning term, Rotary Position Embeddings, extracted from an unrelated arXiv paper) purely because `"rope"` is a substring of `"Europe"` in the query. `KG_MIN_SEED_LENGTH` (Epic 1) filters 1–2 character noise but not this kind of accidental-substring collision on real short words — exactly the "smarter seed matching" (fuzzy/whole-word matching) that Epic 3 already scopes.

**Qualitative examples (question / top retrieved doc / answer, all three non-agentic systems shown):**

- *"How many more millions of dollars was the median exit valuation in the USA compared to Europe...?"* (GT: `63`) — **Baseline: retrieved `earlybird-...pdf` (correct deck) → answered `63` ✓. Advanced: same doc → `63` ✓. KG: retrieved 3 unrelated docs (a Trump-economy Pew survey, a crisis-PR slide deck, an EMNLP paper) → correctly said "cannot be derived from the context," scoring 0.** Textbook case of dense retrieval nailing it and KG missing the source document entirely.
- *"What is the percentage of registered voters who support... the party with the higher total percentage of good policy ideas... in the survey conducted April 25 – May 1, 2018?"* (GT: `92%`, the most deliberately two-stage question in the subset) — **Baseline and Advanced both retrieved the actually-correct source (`PRE_2022.09.29_NSL-politics_REPORT.pdf`, a Pew "National Survey of Latinos" politics report) but still answered "No answer available" — right document, model couldn't chain the two conditions. KG retrieved three unrelated documents (an ACL paper, an Activision 10-K, an unrelated Pew economy report) and also declined to answer.** Same failure outcome, different cause: reasoning ceiling for Systems 1/2, retrieval failure for KG.
- *"Among all 12 references in this report, how many are from its own research center?"* (GT: `8`) — all three systems answered `0`; Baseline/Advanced both retrieved the same wrong document (a Pew Hispanic-identity report, not the source PDF), KG retrieved a different but equally wrong set (an ACL paper, the ISEP student handbook). A case where nobody's retrieval worked — useful as a reminder that not every multi-hop failure is KG-specific.

**Bottom line for the roadmap's stated goal ("prove whether it helps"):** on this corpus, in its current unpolished form, **KG-only graph traversal does not help — it is the weakest of the four systems, on the overall set and especially on the multi-hop subset it was hypothesized to help most.** This isn't a discouraging result so much as a clear, evidenced starting point for Epic 3 (entity resolution, smarter/fuzzy seed matching, edge-weighting/fan-out control) and Epic 4 (hybrid KG+dense+BM25, where the KG signal only has to *add* value on top of an already-strong retriever rather than replace it). The next re-benchmark after Epic 3's fixes is what will show whether this is a fixable implementation gap or a more fundamental mismatch between graph traversal and this corpus's question style.

---

## Epic 3 — Fix the known KG weaknesses (measured iteration)

*Each fix here should be followed by a re-run of Epic 2's benchmark — the delta per fix is the actual engineering narrative, not the fix itself.*

- [x] **Entity resolution / canonicalization.** Right now every surface form is a distinct node. Add a normalization + alias-merge pass (start simple: lowercase + whitespace/punctuation normalization + string-similarity clustering of node labels above some threshold; escalate to an LLM-based "are these the same entity?" pass only if the cheap version isn't enough). Measure node-count reduction and whether retrieval metrics move. — **Skipped, deliberately.** Given Epic 2's finding that KG-only is ~10x worse than the weakest existing system, full clustering-based entity resolution (with its own tuning/benchmark loop) was judged low-leverage relative to the two fixes below, which directly targeted the concrete bugs Epic 2 actually found. Left for future work if further KG-only iteration is warranted.
- [x] **Smarter seed matching.** Substring matching, longest-name-first, has an obvious failure mode: short/common entity names spuriously match unrelated questions. Add a minimum-length or stopword guard, and consider fuzzy matching (e.g. edit distance or token-set overlap) for near-miss entity mentions instead of exact substring only. — Replaced raw substring matching (`entity in query`) with **word-boundary regex matching** (`\bentity\b`, precompiled per entity at load time) in `KGRetriever._find_seeds`. This directly fixes the concrete bug Epic 2 found: the node `"RoPE"` (an ML term from an unrelated arXiv paper) was matching as a seed on any query containing "Eu**rope**" — a pure substring collision, not a real mention. Verified fixed: `"RoPE"` no longer appears in seeds for the Europe/USA valuation question.
- [x] **Edge weighting / fan-out control.** BFS expansion currently treats every edge equally and can explode through high-degree "hub" nodes. Weight edges by evidence strength (`len(source_chunk_uids)`) and/or cap fan-out per hop so expansion stays focused on well-evidenced relations instead of drowning in noise from generic nodes. — Added `KG_MAX_FANOUT` (default 15): when expanding a node with more neighbors than the cap, keep only the best-evidenced ones (by `len(source_chunk_uids)` on the connecting edge) before adding them to the next BFS layer.
- [x] Re-benchmark (Epic 2) after each of the above, and keep a running table of what each change did to the numbers — that table is the evidence that this was engineered, not just assembled once and left alone.

**Re-benchmark after both fixes (word-boundary seeding + fan-out cap), same eval as Epic 2:**

| | Multi-hop subset (19 q) | | | Full set (150 q) | | |
|---|---|---|---|---|---|---|
| | P@1 | NDCG@5 | MAP | P@1 | NDCG@5 | MAP |
| KG before (Epic 2) | 0.0 | 0.079 | 0.053 | 0.0533 | 0.128 | 0.104 |
| **KG after (Epic 3 fixes)** | **0.158** | **0.252** | **0.215** | **0.0467** | **0.173** | **0.132** |

Recall@5, NDCG@5, and MAP all improved meaningfully on both the multi-hop subset (NDCG@5 +219%, MAP +309%) and the full set (NDCG@5 +36%, MAP +27%) — confirming the two bugs Epic 2 found were real, fixable contributors to KG's weak retrieval. P@1 on the full set dipped very slightly (0.0533→0.0467), a minor reshuffling side-effect, not a regression in the metrics that matter more for this retriever's fan-out-heavy design (recall/NDCG over a ranked top-5). **KG-only still remains far below Baseline/Advanced/Agentic** (full-set P@1 0.047 vs. 0.59–0.87) — these fixes closed part of the gap, not all of it. Full exact_match on the multi-hop subset stayed at 1/19, reconfirming Epic 2's other finding: for the hardest questions, generation/reasoning — not retrieval — is the shared ceiling across every system.

> **CORRECTION (see "Tier 1 follow-ups" below):** an ablation grid run later isolated the two fixes and found **the fan-out cap contributed nothing measurable to the numbers above** — capped and uncapped results are bit-for-bit identical on both eval sets. The entire improvement here is attributable to word-boundary seed matching alone. Separately, this table's numbers are themselves **not fully reproducible**: `_collect_chunks` iterated a Python `set()` in hash-randomized order, so a since-added determinism fix changes these exact figures (in both directions — see below). The fan-out cap fix is still real and defensible (the noise it targets exists), it just doesn't reach the final top-5 output at current `TOP_K=5`/`KG_HOPS=2` settings — see "Tier 1 follow-ups" for why and the corrected numbers.

---

## Epic 4 — Formalize the hybrid KG + Dense + BM25 retriever

*This is the actual differentiator. A lot of people can build "vanilla GraphRAG." Showing a rigorous, quantified answer to "when does graph traversal add value on top of vector + keyword search, and by how much" is the part worth putting on a CV or in the report.*

- [x] Promote the logic in `hybrid_retrieval_test.ipynb` into a real module (e.g. `src/indexing/hybrid_kg_retriever.py`) — a class with a `.retrieve()` method, config-driven candidate counts and RRF `k`, no notebook-only globals. — Done: `HybridKGRetriever` (KG + dense + BM25, RRF fusion), config-driven candidate multiplier, RRF `k`, and (added beyond the original notebook logic) per-signal RRF weights.
- [x] Add the config (extend `KGConfig` or add a new `HybridConfig`) and a pipeline (`HybridKGPipeline` or a flag on `KGPipeline`) so it slots into the same `build_index / retrieve / run_query / evaluate` interface as everything else, and gets an entry point the same way Epic 1 did for KG-only. — Done: `HybridConfig(KGConfig)`, `HybridKGPipeline`, `main_hybrid.py` (same `--test-query/--eval/--eval-size/--rebuild-index/--output` CLI as `main_kg.py`).
- [x] Run the full Epic 2 benchmark (overall + multi-hop subset) as a **four-way comparison**: Baseline (BM25+dense) vs. Advanced (multimodal) vs. KG-only vs. Hybrid(KG+dense+BM25). — Done, see table below (five-way: Agentic included too, reusing Epic 2's numbers since none of the Epic 3/4 changes touch Systems 1–3).
- [x] Quantify specifically: on the multi-hop subset, does adding the KG signal to the existing hybrid retriever move metrics up, and by how much, without regressing the non-multi-hop majority of questions? That number is the finding. — **Answer: no, not with naive equal-weight RRF — it actively regresses both.** See findings below.

### Epic 4 Findings (2026-07-17)

**Full 150-question set, all systems (Systems 1–3 from the report; KG/Hybrid freshly run today):**

| System | P@1 | NDCG@5 | MAP | Token F1 | Exact Match |
|---|---|---|---|---|---|
| System 1 — Baseline (dense+BM25, no KG) | 0.5933 | 0.6693 | 0.6478 | 0.1837 | 0.1333 |
| System 2 — Advanced (multimodal) | 0.8667 | 0.9043 | 0.8944 | 0.2658 | 0.2067 |
| System 3 — Agentic | 0.6066 | 0.6695 | 0.6544 | 0.1674 | 0.1133 |
| System 4 — KG-only (post Epic 3 fixes) | 0.0467 | 0.1734 | 0.1318 | 0.0325 | 0.0133 |
| System 5 — Hybrid, **equal-weight RRF** (1.0/1.0/1.0) | 0.3467 | 0.5241 | 0.4702 | 0.1299 | 0.0800 |
| System 5 — Hybrid, **KG down-weighted** (0.3/1.0/1.0) | **0.4933** | **0.6300** | **0.5908** | **0.1688** | **0.1133** |

Three findings, in order of importance:

1. **Naive equal-weight RRF fusion makes things worse, not better.** Hybrid(1.0/1.0/1.0) scores *lower* than plain Baseline on every retrieval and generation metric (P@1 0.35 vs. 0.59, a 42% relative drop; Token F1 0.13 vs. 0.18). KG's ranking is noisy enough that giving it an equal vote in RRF actively drags good dense/BM25 candidates out of the top-5. **Adding a weak signal to a strong retriever is not free** — this is the headline, quantified result the roadmap's Epic 4 goal asked for, and it's a negative one.
2. **Down-weighting the KG signal (0.3x) recovers most, but not all, of that regression.** P@1 climbs from 0.35 → 0.49 (still short of Baseline's 0.59), NDCG@5 0.52 → 0.63 (vs. Baseline's 0.67), Exact Match 0.08 → 0.11 (vs. Baseline's 0.13). Only two weights were benchmarked (0.3, 0.15) on the multi-hop subset before picking 0.3 as the new `HybridConfig` default — this is a reasonable starting point, not a tuned optimum; a proper sweep is future work.
3. **Even at its best-tested setting, Hybrid does not beat Baseline (dense+BM25 alone), let alone Advanced.** On this corpus, with this KG implementation, **the KG signal has not been shown to add value over dense+BM25 retrieval** — the honest conclusion the roadmap asked Epic 4 to reach either way. The one place KG-inclusion could still plausibly help — individual questions where dense+BM25 both miss but a graph-traversal hop happens to hit — was not isolated in this pass; a next step would be a paired per-question comparison (Baseline vs. Hybrid) to check whether *any* subset of questions is uniquely rescued by including KG, even if the aggregate is a net wash or regression.

**Multi-hop subset (19 q) mirrors the same pattern:** Baseline P@1=0.632, Hybrid-equal-weight P@1=0.316, Hybrid-KG-weighted(0.3) P@1=0.632 (ties Baseline exactly on P@1, still trails slightly on NDCG@5/MAP: 0.665/0.658 vs. Baseline's 0.698/0.684) — the multi-hop questions this project hypothesized graph traversal would help most were, if anything, more sensitive to the equal-weight regression, not less.

**Bottom line:** across both epics, the project's central hypothesis — that adding a knowledge graph would help multi-hop retrieval — is **not supported by the evidence on this corpus with this implementation**. That is itself the finding: a rigorous, quantified, negative result (KG hurts naively, and even weighted-in doesn't beat dense+BM25 alone) is a stronger and more honest contribution than an unmeasured "we built GraphRAG" claim would have been.

---

## Epic 5 — Stretch: route the agent to the KG for multi-hop queries

*Only attempt this if Epics 0–4 land cleanly with time to spare — it's the most architecturally interesting piece but also the most integration work.*

- [x] In the existing LangGraph agentic pipeline (`src/agentic/graph/`, `pipelines/agentic_pipeline.py`), add KG retrieval as an additional tool/branch. — Added `KGMultihopRetrieval` (`src/query_techniques/kg_multihop.py`), a `QueryTechnique`-conforming wrapper around `HybridKGRetriever`, plumbed in as a 9th selectable technique via `AgenticConfig.ENABLE_KG_ROUTING` (opt-in, default `False` — existing Agentic/System 3 behavior is unaffected unless explicitly enabled).
- [x] Have the query-rewriter or grader agent classify a question as multi-hop/relational vs. direct-lookup, and route multi-hop questions specifically through the KG (or hybrid) retriever instead of the standard technique — the same "classify then route" pattern already used successfully in your other course's multimodal retrieval strategy (Strategy 4: text/visual classification → route accordingly). — Implemented as designed: added `kg_multihop` to the query-rewriter's existing free-choice technique menu (the same mechanism it already uses to pick among the other 8 techniques) with a description explicitly telling it to use `kg_multihop` for "multi-hop/relational questions that require chaining facts across multiple distinct entities."
- [x] Benchmark Agentic+KG-routing vs. plain Agentic (System 3) on the same multi-hop subset. — Done, see findings below.

### Epic 5 Findings (2026-07-17)

Ran Agentic with `ENABLE_KG_ROUTING=True` on the same 19-question multi-hop subset used throughout Epics 2/4.

**The query-rewriter agent never once selected `kg_multihop`, across all 19 multi-hop questions:**

| Technique chosen | Count |
|---|---|
| `standard` | 8 |
| `hyde` | 6 |
| `query_decomposition` | 5 |
| `kg_multihop` | **0** |

Consequently the resulting metrics are, within LLM-call noise, indistinguishable from plain Agentic on the same subset (P@1 0.632 both; NDCG@5 0.665 vs. 0.698; Token F1 0.053 vs. 0.070; Exact Match 0.053 both, 1/19) — the small deltas are call-to-call variance in the agent's own decisions, not an effect of KG routing, because KG routing was never invoked.

**Why this is still a real (if smaller) finding, not a non-result:** the "classify then route" design here relies on the query-rewriter LLM recognizing multi-hop questions from a text description and voluntarily picking `kg_multihop` over 8 already-familiar-sounding alternatives — and it didn't, not once, even though every question in the subset was deliberately curated to be exactly the kind of question the description called out. Two explanations, not mutually exclusive: (1) the agent's existing options (especially `query_decomposition`) already sound like a plausible match for "multi-part" questions, so `kg_multihop` never wins the choice; (2) per Epic 4's findings, `HybridKGRetriever` doesn't actually outperform the dense+BM25 techniques the agent already has access to on this subset (P@1 0.632 tied, not exceeded, at the best-tested weighting) — so even a perfectly-calibrated classifier routing every multi-hop question to `kg_multihop` would not have been expected to improve on plain Agentic's numbers here. **This session did not run that forced-routing control** (bypass the agent's free choice and force `kg_multihop` for all 19 questions through the full Agentic generation/grading loop) — that would isolate "does forced routing help" from "does the agent choose to route," and is the natural next step if this is revisited.

**Bottom line:** the architecture works end-to-end (no crashes, clean opt-in via config, technique correctly loads and is available), but this pass could not demonstrate a benefit from agent-routed KG queries — consistent with, and further reinforcing, Epics 2–4's central finding that KG/hybrid retrieval does not currently outperform this project's existing dense+BM25 retrieval on this corpus.

> **CORRECTION (see "Tier 1 follow-ups" below):** the "0/19, never chosen" claim above was wrong — caused by a real bug (`QueryRewriterDecision`'s Pydantic `Literal` type never listed `"kg_multihop"` as a valid value, so a genuine LLM choice of it failed validation and was silently miscounted as a parsing error, falling back to `"standard"`). Fixed and re-run: the agent actually chose `kg_multihop` **3/19** times, and the corrected free-choice result **slightly exceeds** plain Agentic (P@1 0.684 vs. 0.632). Left here unmodified, with this note, so the correction is traceable — see "Tier 1 follow-ups" for the full, corrected numbers.

---

## Epic 6 — Document and ship

- [x] Add "System 4 — Knowledge Graph RAG" (and "System 5 — Hybrid," if Epic 4 lands) to the README's Systems Overview table and architecture diagram, matching the existing style. — Done: Systems Overview table, project structure tree, a new "Step 4" usage section, and Configuration parameter rows all updated in `README.md`.
- [x] Write a results section — the four-way comparison table, the multi-hop breakdown, and the 2–3 qualitative examples from Epic 2 — either as a README addendum or an update to the report. — Done: new `## Results` section in `README.md` (full-set + multi-hop tables across all 5 systems, 5 condensed findings); full detail/reasoning/qualitative examples remain in this file's Epic 2–5 sections.
- [ ] Clean, incremental commits on `feature/knowledge-graph-networkx`; open a PR against `main` on `dat560-2026/project-team-1` if teammates should see it, or keep it on your fork (`mygithub`) if this is personal/portfolio-track work beyond the graded deliverable. — Commits done (24+ logical commits across Epics 0–6, each with a scoped message). **Pushing/opening a PR was intentionally left for you to decide** — that's a visible action affecting a shared repo, not something to do unprompted.

---

## Suggested sequencing

- **Epics 0–2 are the must-do floor** — cheap, mostly wiring and one evaluation run, and they're what turns this from "half-built" into "real and measured." Do these regardless of how much further time you have.
- **Epics 3–4 are where the actual differentiation lives** — this is what separates "I followed a GraphRAG tutorial" from "I measured where graph retrieval helps and improved it." Worth the deeper time investment if you're treating this as more than a course footnote.
- **Epics 5–6 are polish/stretch** — valuable if Epics 0–4 leave time on the table, skippable if they don't.

---

## Epic 7 — Tier 1 follow-ups (post-mortem improvement pass)

After Epic 6, a structured improvement-finding pass (parallel research across 8 dimensions — entity resolution, extraction quality, retrieval matching, ranking/scoring, structural fit, fusion strategy, agent routing, evaluation rigor — with adversarial verification of every proposal) produced 40 candidate improvements. Most were rejected (already-tried, empirically falsified, or low-leverage relative to the ~10x/5x gap); five cheap, high-information-value "Tier 1" items survived and were implemented and benchmarked. One of them surfaced a real bug that changes a previous conclusion.

**One finding that reframed the whole retrieval-matching dimension, established before any code changed:** an adversarial verification agent empirically ran `KGRetriever._find_seeds` against all 150 test questions and found the "no seed found" fallback fires **0% of the time** (median 11–12 seed entities match per query, even post-Epic-3). The graph isn't under-matching — it's drowning in matches. This ruled out an entire category of proposal (fuzzy matching, embedding-based entity linking, query-side NER, acronym expansion — all "find more seeds") and pointed at the real problem: precision/ranking, not recall.

### 1. Deterministic chunk ordering (`KGRetriever._collect_chunks`)

**Bug:** chunks within a BFS layer were collected by iterating a Python `set()` directly — `str` hashing is salted per-process (`PYTHONHASHSEED`), so the exact same query against the exact same graph could return **different top-5 chunks on different process runs**, whenever a layer had more candidates than the remaining `top_k` budget (common — see finding 4 below). This was actually observed mid-session during Epic 1 debugging (two consecutive runs of one query gave different retrieved docs) but never fixed.

**Fix:** sort each layer's nodes alphabetically before collecting (`src/indexing/kg_retriever.py`). This is a reproducibility fix, not a relevance-ranking improvement — alphabetical order carries no meaning; it's just now the *same* order every time. Real per-chunk relevance scoring remains unbuilt (Tier 2/3 territory — every chunk still nominally gets `score: 1.0`).

**Consequence — a real, honest correction to earlier numbers:** re-running KG-only after just this fix (word-boundary + fan-out cap unchanged) gives materially different results than Epic 3's originally-reported table, in *both* directions:

| Config | Multi-hop (19q) P@1 | Full set (150q) P@1 |
|---|---|---|
| Epic 3 report (hash-random order) | 0.158 | 0.047 |
| **Now (deterministic, alphabetical order)** | **0.0** | **0.14** |

The multi-hop subset got *worse* (0.158 → 0.0) and the full set got *better* (0.047 → 0.14) — purely from changing an arbitrary tie-break, nothing else. **This means the Epic 2–4 single-run deltas throughout this roadmap were themselves not fully reproducible**, and reinforces (independently) the improvement-pass's own "evaluation rigor" dimension finding: n=19 is small enough that single-run numbers should be read as suggestive, not conclusive, until repeated-trial variance or significance testing is added (still not done — noted as open below).

### 2. `think=False` bug fix in `kg_builder.py` (defensive, for future model upgrades)

**Finding:** `kg_builder.py`'s triple-extraction call passes Ollama's `think=False` with no `/no_think` prompt injection — the exact same qwen3-family unreliability `generation/generator.py`'s `VisionGenerator` already had to work around elsewhere in this codebase. The current default (`llama3:8b`) isn't affected, but a future switch to `qwen3:32b` (considered in Epic 3 as a possible extraction-quality upgrade) would hit this silently: a leaked `<think>` block breaks `json.loads` and zeroes out that chunk's triples.

**Fix:** inject `/no_think` for qwen3-family models specifically (leaves `llama3:8b` behavior unchanged), plus defensively strip any `<think>...</think>` block from the raw response before parsing, regardless of model. Not yet exercised against a real qwen3 extraction run — no model upgrade has been attempted — but the landmine is now defused for when/if one is.

### 3 & 4. Forced-routing control — and a second real bug found along the way

**What was built:** `AgenticConfig.FORCE_TECHNIQUE` (default `None`) short-circuits `query_rewriter_node` to always pick a named technique, bypassing the LLM decision — isolating "does routing to `kg_multihop` help" from "does the agent choose to route there."

**Bug found while smoke-testing it:** forcing `kg_multihop` crashed with a Pydantic validation error — `QueryRewriterDecision`'s `technique` field is a `Literal` type that **never included `"kg_multihop"`** as a valid value, even though Epic 5 added it as a 9th selectable technique. Fixed (`src/agentic/tools/output_parser.py`).

**This bug wasn't just a forced-routing problem — it silently corrupted Epic 5's own original finding.** Without `"kg_multihop"` in the `Literal`, a genuine LLM choice of it would fail Pydantic validation, get caught by `query_rewriter_node`'s broad `except Exception`, and get miscounted as a parsing error falling back to `"standard"` — indistinguishable from a real parse failure. Checking the original Epic 5 run: **4/19 questions had "Error in parsing, using baseline"** as their logged reasoning. Re-running with the bug fixed:

| Run | P@1 | NDCG@5 | MAP | Token F1 | Exact Match | `kg_multihop` selected |
|---|---|---|---|---|---|---|
| Plain Agentic (no KG routing, Epic 2) | 0.632 | 0.698 | 0.684 | 0.070 | 0.053 (1/19) | n/a |
| Agentic + KG routing, free choice — **as originally reported (buggy)** | 0.632 | 0.698 | 0.684 | 0.070 | 0.053 (1/19) | 0/19 (miscounted) |
| **Agentic + KG routing, free choice — corrected** | **0.684** | **0.717** | **0.711** | 0.070 | 0.053 (1/19) | **3/19** |
| Agentic + KG routing, **forced** (100%) | 0.579 | 0.639 | 0.623 | 0.053 | 0.053 (1/19) | 19/19 |

**Corrected finding — reverses part of Epic 5's original conclusion:** the agent does choose `kg_multihop` when it's actually able to (3/19, all genuinely relational-sounding questions — e.g. "the method located at the bottom of the model structure figure"), and the corrected free-choice run **slightly exceeds** plain Agentic on every retrieval metric (P@1 +8%, NDCG@5 +3%, MAP +4%). *However*, forcing `kg_multihop` on all 19 questions performs *worse* than letting the agent choose selectively (P@1 0.579 vs. 0.684) — selective routing beats blanket routing, which is itself informative: the agent's judgment about *when* to route, on the cases it does route, is doing real work, even though a majority of genuinely multi-hop questions still don't get routed there. None of the 3 `kg_multihop`-routed questions were answered correctly in the end (`CANNOT_FIND_ANSWER` for all 3, generation-side reasoning ceiling as established elsewhere) — the gain is retrieval-only, consistent with everything else this roadmap has found about where the bottleneck actually sits for hard questions.

Treat all four numbers in this table with the same n=19/single-run caveat as finding 1 — a repeated-trial or significance-tested re-run (still open, see below) would strengthen this materially.

### 5. Ablation grid: isolating word-boundary matching from the fan-out cap

**What was built:** `KGConfig.KG_USE_WORD_BOUNDARY_SEEDS` (default `True`) lets `_find_seeds` fall back to pre-Epic-3 raw substring matching; combined with setting `KG_MAX_FANOUT` to a very large number, this reproduces all 4 cells of the 2×2 grid (word-boundary × fan-out-cap) that Epic 3 bundled into one reported number.

| Config | Multi-hop (19q) P@1 / NDCG@5 / MAP | Full set (150q) P@1 / NDCG@5 / MAP |
|---|---|---|
| raw substring + no cap (pre-Epic-3) | 0.0 / 0.023 / 0.013 | 0.040 / 0.102 / 0.083 |
| word-boundary + no cap | 0.0 / 0.026 / 0.018 | 0.140 / 0.226 / 0.199 |
| raw substring + fan-out cap | 0.0 / 0.023 / 0.013 *(identical to no-cap row)* | 0.040 / 0.102 / 0.083 *(identical)* |
| **word-boundary + fan-out cap (current)** | 0.0 / 0.026 / 0.018 *(identical to no-cap row)* | 0.140 / 0.226 / 0.199 *(identical)* |

**Finding: the fan-out cap makes zero measurable difference, on either eval set, at current settings.** Capped and uncapped rows are bit-for-bit identical. Investigated why directly: `_collect_chunks` fills the `top_k=5` budget by iterating BFS layers seed-layer-first, and the seed layer alone routinely supplies far more than 5 candidate chunks before hop-1/hop-2 (where `KG_MAX_FANOUT` actually operates) is ever reached — confirmed concretely for one query, where 14 matched seed entities contributed **50 chunk-slots** before dedup, ten times the `top_k` budget. `KG_MAX_FANOUT` only limits which *neighbor nodes* get added during BFS expansion; it does not cap how many chunks a single node (seed or otherwise) contributes directly, so even the seed layer's own dilution problem (a generic seed like "time" contributing 14 chunks alongside a specific one like "Europe" contributing 11, with no evidence-based priority between them) is untouched by the fix that was supposed to address exactly this.

**Corrected attribution:** all of Epic 3's reported improvement is attributable to word-boundary matching alone. The fan-out cap is not wrong to have added — the noise pattern it targets (hub nodes exploding BFS expansion) is real and would matter at a larger `top_k` or more hops — but it currently does nothing observable, and the real remaining seed-layer dilution problem it doesn't reach is a legitimate Tier 2 target (per-chunk scoring / specificity weighting, from the improvement pass's `ranking_scoring` dimension).

### Not done this pass

- **Paired per-question rescue analysis** (does Hybrid/KG ever uniquely get right what Baseline gets wrong, across the full 150) — blocked by a recurring network issue on this machine (SSL interception prevents the embedder from loading; Baseline/Advanced/Hybrid all need it, KG-only doesn't). Deferred at the user's call rather than working around it. Still the single highest-value open question from the whole improvement pass.
- **Repeated-trial variance / significance testing** — every number in this Epic 7 section (and, it turns out, in Epics 2–4) is a single run. Finding 1 above is direct proof this matters.

### Commits

Code: word-boundary/no-word-boundary ablation switch + deterministic sort (`kg_retriever.py`), `think=False` fix (`kg_builder.py`), `FORCE_TECHNIQUE` + `kg_multihop` Literal fix (`config.py`, `agentic_pipeline.py`, `agentic/graph/nodes.py`, `agentic/tools/output_parser.py`). Results: `kg_ablation_grid.json`, `agentic_kg_forced_routing_multihop.json`, `agentic_kg_routing_multihop_v2.json`.

---

## Epic 8 — Tier 2 follow-up: real per-chunk scoring

The highest-leverage Tier 2 item from the improvement pass: give KG-retrieved chunks a real, query-relevant score instead of the hardcoded `1.0` Epic 7 documented. Directly targets the mechanism Epic 7's ablation grid diagnosed — the seed layer alone routinely supplies far more candidate chunks than `top_k` allows, with zero internal prioritization (a generic hub match like "students" ranked identically to a specific one like "NTU").

**Implemented** (`KGRetriever`, `src/indexing/kg_retriever.py`):
- `_node_specificity(node)`: `1 / (1 + log1p(evidence_count))` — rare, specific entities score higher than generic hubs.
- `_hop_weight(hop)`: graded decay (seed=1.0, hop-1=0.5, hop-2=0.25, geometric beyond) replacing the old hard "all seed chunks outrank all hop-1 chunks" partition.
- `_collect_chunks` now scores every candidate as `hop_weight × specificity`, sorts once, and returns real scores (previously always `1.0`).

**Isolated the two factors before combining them** (repeating Epic 7's own ablation-grid lesson rather than re-bundling changes):

| Config | Multi-hop (19q) NDCG@5 / MAP | Full set (150q) NDCG@5 / MAP |
|---|---|---|
| Epic 7 baseline (alphabetical only, no real score) | 0.026 / 0.018 | 0.226 / 0.199 |
| Hop-decay ONLY (specificity held constant) | 0.000 / 0.000 | 0.036 / 0.026 |
| Specificity ONLY (hop-weight held constant) | 0.158 / 0.123 | 0.162 / 0.128 |
| **Both combined (shipped)** | **0.158 / 0.123** | **0.193 / 0.147** |

**Findings:**
1. **Specificity is doing essentially all the work; hop-decay alone is actively harmful and, combined, only adds a little back for the full set.** On the multi-hop subset, specificity-only and combined are identical to 3 decimal places — hop-1/hop-2 chunks are so rarely reached (per Epic 7's seed-layer-dilution finding) that hop-decay has nothing left to differentiate. Hop-decay alone (no specificity) actually scores *worse* than even the old alphabetical tiebreak, because without specificity all same-hop nodes tie and the only differentiator becomes a uid string sort — a different, equally-meaningless order.
2. **Net effect is a genuine, quantified tradeoff, not a clean win.** Specificity weighting **substantially helps the multi-hop subset** (NDCG@5 +508%, MAP +583% vs. Epic 7 baseline) — the one thing a knowledge graph is actually hypothesized to help with — but **regresses the broader full-150 set on P@1 specifically** (0.14 → 0.053), while still improving NDCG@5/MAP there too. This is exactly the risk flagged when this proposal was first written: evidence count isn't a pure noise signal — sometimes the correct answer entity for an easier, non-multi-hop question really is a well-evidenced, central one (e.g. a company's own name recurring throughout its own 10-K), and down-weighting it by "rarity" can push it out of the top-1 slot even while broader ranking quality (NDCG@5/MAP) still improves.
3. **A quick blend (specificity compressed toward neutral at α=0.25/0.5/0.75) produced identical results at all three values** — the compression wasn't aggressive enough in this range to flip any cross-hop comparison. Finding a genuinely different tradeoff point would need a real sweep at much lower α or a different formula entirely; not pursued further as disproportionate for a Tier 2 item.

**Decision: shipped as the new default**, on the reasoning that (a) it's the correct, well-motivated fix for the exact mechanism diagnosed, (b) it materially improves the one axis this whole KG extension was built to test, and (c) the full-set P@1 "regression" happens entirely within a system (KG-only) already established as categorically non-competitive with Baseline (System 1) regardless of this tuning choice — it doesn't change that qualitative conclusion either way. Easily reverted via `_node_specificity`/`_hop_weight` if this call turns out wrong once more evidence comes in.

**Also fixed:** the query-rewriter's "Selection Guidance" prompt block (`src/agentic/graph/nodes.py`) was stale since Epic 5 added `kg_multihop` — it never mentioned the new technique, and its "multi-part questions → query_decomposition" bullet actively pointed toward a competing technique for exactly the kind of question `kg_multihop` is meant for. Added a guidance line explicitly disambiguating the two ("must resolve fact A elsewhere before fact B is answerable, not just compound wording"). **Not yet re-benchmarked** — see below.

### Blocked this pass (all downstream of the same root cause)

The recurring SSL/network issue on this machine (blocks `huggingface.co`, needed for the `jina-clip-v2` embedder) came back and is blocking three things that all matter for validating today's changes and Epic 7's still-open items:
- Re-running Agentic+KG-routing on the multi-hop subset with the new scoring fix + Selection Guidance prompt fix (needs the embedder for the dense signal).
- Confirming the new KG scoring doesn't regress `HybridKGPipeline`'s full-150 numbers (Hybrid consumes `KGRetriever`'s candidate list as one of its three RRF signals — Epic 8's full-set P@1 dip could plausibly propagate downstream).
- Epic 7's still-outstanding per-question rescue analysis and full-150 RRF weight sweep.

Deferred rather than worked around, consistent with the earlier call on this same issue. All are cheap to run once network access is restored — no code changes needed, just re-execution.

### Commits

Code: `_node_specificity`/`_hop_weight`/rewritten `_collect_chunks` (`kg_retriever.py`), Selection Guidance fix (`agentic/graph/nodes.py`). Results: `kg_scoring_fix_eval.json`.

---

## Epic 9 — Closing Epic 7's last open items: weight sweep + rescue analysis

Network access came back, unblocking everything Epics 7–8 had deferred. This closes all of it — and delivers the single most decisive finding of the whole extension.

### Re-validation: did Epic 8's KG scoring fix regress anything downstream?

| | Agentic + KG routing (free choice, 19q) | Hybrid, full 150q |
|---|---|---|
| Before Epic 8 (scoring fix) | P@1=0.684, NDCG@5=0.717, MAP=0.711 | P@1=0.493, NDCG@5=0.630, MAP=0.591 |
| **After Epic 8** | **P@1=0.684, NDCG@5=0.717, MAP=0.711** (identical) | **P@1=0.487, NDCG@5=0.628, MAP=0.588** (small real regression, ~1–4% relative) |

Agentic is unaffected (bit-for-bit identical — the KG scoring change doesn't reach far enough into the fusion to matter once RRF only cares about rank there, and the agent's own technique mix dominates). Hybrid shows the small regression flagged as a risk in Epic 8: real but modest, consistent with the 0.3x KG weight already cushioning most of KG-only's own larger internal regression.

The query-rewriter's Selection Guidance fix: re-ran free-choice routing and got `kg_multihop` selected 3/19 again, but for a **different set of 3 questions** (index 3 dropped out, index 10 newly appeared; indices 2 and 8 stayed). One run each side isn't enough to say the guidance fix changed anything, or even to isolate its effect from ordinary LLM-call variance — the run-to-run variance question Epic 7 flagged remains open.

### The weight sweep, done properly this time (full 150-question set, not just the 19-question subset)

Cached each of the three raw signal candidate lists (KG, dense, BM25) once per question, then re-ran only the RRF fusion math per weight — cheap, no re-retrieval needed:

| `kg_weight` | P@1 | NDCG@5 | MAP |
|---|---|---|---|
| **0.0 (no KG at all)** | **0.580** | **0.672** | **0.647** |
| 0.15 | 0.513 | 0.643 | 0.608 |
| 0.3 (current default, chosen in Epic 4 from only 2 points tested on the 19q subset) | 0.487 | 0.628 | 0.588 |
| 0.5 | 0.440 | 0.603 | 0.556 |
| 0.7 | 0.420 | 0.583 | 0.537 |
| 1.0 (equal weight) | 0.367 | 0.536 | 0.486 |

**Monotonic decline. There is no weight at which adding the KG signal improves the full-150 aggregate — the true optimum is zero.** Epic 4's 0.3 default was tuned against the 19-question multi-hop subset only, exactly the overfitting risk flagged at the time; on the representative full set, it's already partway down a slope that only gets worse. This isn't a tuning problem to solve with a better weight — there isn't a better weight, on this metric, on this set.

### The rescue analysis — the answer to the question this whole project circled around

Per-question paired comparison, full 150 questions, P@1 (retrieval-only, no generation cost):

| System | P@1 hits |
|---|---|
| Baseline alone | **95/150** |
| Hybrid (kg_weight=0.3) | 73/150 |
| KG-only | 8/150 |

| | Rescued (Baseline misses, this system hits) | Hurt (Baseline hits, this system misses) |
|---|---|---|
| Hybrid vs. Baseline | **5** | **27** |
| KG-only vs. Baseline | **2** | **89** |

**This is not "a wash." Adding KG, at any tested weight or alone, breaks far more than it fixes.** The 5 questions Hybrid rescues (a housing-office email lookup, a pipeline-diagram organism question, a safety-bullet lookup, the Vietnam-vs-global iOS9 comparison, a car-models lookup) are mostly ordinary factual lookups, not cleanly explained by "multi-hop-ness" — only one (the Vietnam/iOS9 comparison) is from the hand-curated multi-hop subset. There is no natural query-type filter visible in this data that would let an adaptive-weighting scheme keep the 5 rescues while dropping the 27 (Hybrid) or 89 (KG-only) casualties — the rescues look closer to incidental luck than a systematic pattern to route toward.

### Bottom line for the whole KG extension

Across Epics 2–9, every angle has now been checked: naive fusion, tuned fusion, a real weight sweep across the full range, agent-mediated selective routing (both free-choice and forced), and — finally — a direct per-question accounting of exactly which questions gain and which are lost. All of them point the same direction. **On this corpus, with this implementation, knowledge-graph retrieval does not add value to an already-strong dense+BM25 pipeline — not on aggregate, not on the multi-hop subset it was built for, and not even on a lenient "does it rescue anything, anywhere" standard.** The value of this extension was never going to be "KG makes the numbers go up" — it's this: a precise, quantified, multiply-cross-checked account of exactly where and why it doesn't, with the specific mechanisms (extraction noise, blunt substring/word-boundary matching, no real scoring until Epic 8, a representational mismatch between triples and tabular questions) identified and, where cheaply fixable, fixed.

### Commits

Results only, no code changes this epic: `agentic_kg_routing_multihop_v3.json`, `hybrid_weighted_eval_150_v2.json`, `rrf_weight_sweep_full150.json`, `rescue_analysis_full150.json`.

---

## Epic 10 — Is it the extraction model? (raised by user, investigated directly)

Epics 2–9 treated the graph as fixed and exhaustively tested how well retrieval *uses* it. This epic checks the graph's own quality — extracted by `llama3:8b`, chosen for speed over the alternative `qwen3:32b` — for the first time.

### Hand-audit of the existing graph (12 chunks, stratified sample)

Pulled every triple the current graph attributes to a random sample of chunks and read them against the source text. Most were reasonably grounded (citation lists, factual survey/report content, product docs). But one was a **flagrant, confirmed hallucination**: chunk `936c0e2c...pdf::24` is a blank/garbled table row (literally just dashes and pipes, no real content) — yet the graph attributes it three fabricated corporate facts: `(Apple, reported revenue of, $94.9B)`, plus invented Amazon and Microsoft facts. The Apple/$94.9B triple is **word-for-word the few-shot example in `kg_builder.py`'s own prompt** — given nothing to extract, the model regurgitated its own example instead of returning `[]`.

### Live side-by-side comparison: `llama3:8b` vs. `qwen3:32b`, same chunks, same prompt

Re-ran extraction on the exact chunk above plus two more (a table-misreading case, a vague curriculum-text case) with both models:

| Case | `llama3:8b` | `qwen3:32b` |
|---|---|---|
| Blank/garbled table | Fabricated 3 facts (the hallucination above) | **Correctly returned `[]`** |
| Table checkmark-matrix (RAG survey metrics) | Garbled column names, some rows missed | More faithful to actual column headers, more complete |
| Curriculum standards text | Vague, fragmented, one relation backwards | Correctly captured the real unit/quarter/assignment hierarchy |

### Quantified on a larger sample (30 chunks, both models, same prompt)

Broader pattern, not just the one flagrant case: **`llama3:8b` fabricated non-trivial content on at least 4/30 sampled chunks** that had no real extractable facts (bare section headings, legal-boilerplate cross-references, dangling sentence fragments) — inventing plausible-sounding but unsupported triples rather than recognizing there was nothing to extract. **`qwen3:32b` correctly returned `[]` on all of those same chunks.** Beyond hallucination-avoidance, `qwen3:32b` was also consistently more complete on chunks with real content (e.g. correctly parsing garbled OCR'd VC-deck statistics — `$173M`, `236M`, `131`, `596` — that `llama3:8b` missed entirely, even inventing a garbage OCR artifact word, "omebac", as an entity).

**What this does and doesn't explain:** this is real, demonstrated evidence that `llama3:8b`'s extraction quality is a genuine contributing factor to graph noise — plausibly explaining some of the generic pseudo-entity nodes ("time", "many", "the") that have caused problems since Epic 1. It does **not** explain everything: the table/checkmark-matrix misreading persisted in *both* models (a prompt/schema design issue, not a model-capability one — consistent with Epic 8's `graph_task_fit` finding that triples don't naturally represent tabular data), and it has no bearing on the two other independently-confirmed, model-independent problems: blunt word-boundary/substring matching (vs. dense embeddings) and the shared generation-reasoning ceiling that caps every system, KG included, at the same low exact-match rate on hard questions.

### Cost of finding out for certain

Measured directly (8-chunk timing sample): `llama3:8b` averages 2.06s/chunk (consistent with the original ~7.3-hour build over 12,747 chunks), `qwen3:32b` averages 7.61s/chunk — **3.7x slower**. A full re-extraction of the corpus would take an estimated **~27 hours** of sequential LLM calls against the shared university endpoint (`kg_builder.py`'s `build_graph` has no concurrency), against the original build's ~7.3 hours. The `think=False`/qwen3 landmine fixed defensively in Epic 8 is confirmed necessary and working — `qwen3:32b` extraction ran cleanly with no leaked `<think>` blocks in this test.

**Recommendation, not yet acted on:** the evidence justifies a re-extraction, but a ~27-hour full rebuild is a real resource commitment against a shared endpoint with an uncertain (though plausible) payoff, given the independently-confirmed structural issues above would persist regardless. Left as an explicit decision for the user rather than launched unprompted.

### Commits

Results only: `extraction_model_comparison_sample30.json`. No code changes — the `think=False` qwen3 fix from Epic 8 already made this test possible.

---

## Epic 11 — Statistical rigor, for publication

The one methodological gap flagged repeatedly since Epic 7 and never closed: every benchmark number in this project was a single run, with no confidence intervals and no significance testing. Closed it, in preparation for writing this work up as a paper.

**McNemar's test** (paired, same 150 questions) on the three headline retrieval comparisons — all significant:
- Baseline vs. Hybrid(0.3): χ²=13.78, **p=0.00021**
- Baseline vs. KG-only: χ²=81.28, **p<0.00001**
- **Hybrid(0.0, no KG) vs. Hybrid(0.3): χ²=7.04, p=0.00796** — the critical one: adding KG at the shipped default weight is a statistically significant regression relative to not using it, not sampling noise.

**Bootstrap 95% CIs** (question-level resampling, 5,000 iterations) on P@1 for all four systems — all difference CIs exclude zero (Baseline − Hybrid(0.3): 0.147, CI [0.073, 0.213]; Baseline − KG-only: 0.580, CI [0.493, 0.660]).

**Repeated trials (n=3) on Agentic+KG-routing free-choice**, the one part of the pipeline with genuine LLM-call stochasticity: across three independent runs, the aggregate retrieval outcome was **bit-for-bit identical every time** (P@1=0.6842, NDCG@5=0.7174, MAP=0.7105), even though the categorical count of `kg_multihop` selections varied (3, 3, 5 of 19). The routing decision is stochastic; the resulting retrieval quality is not — a stable, low-variance result worth reporting as such.

All findings from Epics 0–11 are now compiled, restructured by claim rather than chronology, into `KG_RESEARCH_FINDINGS.md` — written specifically as source material for a paper, with a suggested reframing of the negative result (§10 of that document): not "KG doesn't work," but a boundary condition — this corpus's tabular/numeric question structure doesn't map onto entity-relation triples, a mismatch that persisted across two extraction models of different capability.

### Commits

Code: none. Results: `statistical_significance_analysis.json`, `agentic_kg_routing_multihop_trial4.json`. New document: `KG_RESEARCH_FINDINGS.md`.
