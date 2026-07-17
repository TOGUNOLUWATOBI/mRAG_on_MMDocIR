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

- [ ] **Entity resolution / canonicalization.** Right now every surface form is a distinct node. Add a normalization + alias-merge pass (start simple: lowercase + whitespace/punctuation normalization + string-similarity clustering of node labels above some threshold; escalate to an LLM-based "are these the same entity?" pass only if the cheap version isn't enough). Measure node-count reduction and whether retrieval metrics move.
- [ ] **Smarter seed matching.** Substring matching, longest-name-first, has an obvious failure mode: short/common entity names spuriously match unrelated questions. Add a minimum-length or stopword guard, and consider fuzzy matching (e.g. edit distance or token-set overlap) for near-miss entity mentions instead of exact substring only.
- [ ] **Edge weighting / fan-out control.** BFS expansion currently treats every edge equally and can explode through high-degree "hub" nodes. Weight edges by evidence strength (`len(source_chunk_uids)`) and/or cap fan-out per hop so expansion stays focused on well-evidenced relations instead of drowning in noise from generic nodes.
- [ ] Re-benchmark (Epic 2) after each of the above, and keep a running table of what each change did to the numbers — that table is the evidence that this was engineered, not just assembled once and left alone.

---

## Epic 4 — Formalize the hybrid KG + Dense + BM25 retriever

*This is the actual differentiator. A lot of people can build "vanilla GraphRAG." Showing a rigorous, quantified answer to "when does graph traversal add value on top of vector + keyword search, and by how much" is the part worth putting on a CV or in the report.*

- [ ] Promote the logic in `hybrid_retrieval_test.ipynb` into a real module (e.g. `src/indexing/hybrid_kg_retriever.py`) — a class with a `.retrieve()` method, config-driven candidate counts and RRF `k`, no notebook-only globals.
- [ ] Add the config (extend `KGConfig` or add a new `HybridConfig`) and a pipeline (`HybridKGPipeline` or a flag on `KGPipeline`) so it slots into the same `build_index / retrieve / run_query / evaluate` interface as everything else, and gets an entry point the same way Epic 1 did for KG-only.
- [ ] Run the full Epic 2 benchmark (overall + multi-hop subset) as a **four-way comparison**: Baseline (BM25+dense) vs. Advanced (multimodal) vs. KG-only vs. Hybrid(KG+dense+BM25).
- [ ] Quantify specifically: on the multi-hop subset, does adding the KG signal to the existing hybrid retriever move metrics up, and by how much, without regressing the non-multi-hop majority of questions? That number is the finding.

---

## Epic 5 — Stretch: route the agent to the KG for multi-hop queries

*Only attempt this if Epics 0–4 land cleanly with time to spare — it's the most architecturally interesting piece but also the most integration work.*

- [ ] In the existing LangGraph agentic pipeline (`src/agentic/graph/`, `pipelines/agentic_pipeline.py`), add KG retrieval as an additional tool/branch.
- [ ] Have the query-rewriter or grader agent classify a question as multi-hop/relational vs. direct-lookup, and route multi-hop questions specifically through the KG (or hybrid) retriever instead of the standard technique — the same "classify then route" pattern already used successfully in your other course's multimodal retrieval strategy (Strategy 4: text/visual classification → route accordingly).
- [ ] Benchmark Agentic+KG-routing vs. plain Agentic (System 3) on the same multi-hop subset.

---

## Epic 6 — Document and ship

- [ ] Add "System 4 — Knowledge Graph RAG" (and "System 5 — Hybrid," if Epic 4 lands) to the README's Systems Overview table and architecture diagram, matching the existing style.
- [ ] Write a results section — the four-way comparison table, the multi-hop breakdown, and the 2–3 qualitative examples from Epic 2 — either as a README addendum or an update to the report.
- [ ] Clean, incremental commits on `feature/knowledge-graph-networkx`; open a PR against `main` on `dat560-2026/project-team-1` if teammates should see it, or keep it on your fork (`mygithub`) if this is personal/portfolio-track work beyond the graded deliverable.

---

## Suggested sequencing

- **Epics 0–2 are the must-do floor** — cheap, mostly wiring and one evaluation run, and they're what turns this from "half-built" into "real and measured." Do these regardless of how much further time you have.
- **Epics 3–4 are where the actual differentiation lives** — this is what separates "I followed a GraphRAG tutorial" from "I measured where graph retrieval helps and improved it." Worth the deeper time investment if you're treating this as more than a course footnote.
- **Epics 5–6 are polish/stretch** — valuable if Epics 0–4 leave time on the table, skippable if they don't.
