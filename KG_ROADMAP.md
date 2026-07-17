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

- [ ] Commit the current working tree on `feature/knowledge-graph-networkx`: `kg_builder.py`, `kg_retriever.py`, `kg_store.py`, `kg_pipeline.py`, `kg_preprocessing_pipeline.py`, the `KGConfig` addition in `config.py`, and the `requirements.txt` change. Split into 2–3 logical commits (e.g. "feat: KG builder/store/retriever", "feat: KGConfig + KGPipeline wiring") rather than one dump, matching the repo's existing commit style.
- [ ] Decide whether to commit `knowledge_graph.json` (37.5 MB) into git, or move it to the same "regenerate via script, don't commit" pattern already used for the other `chunks_*.json` preprocessed files. Check whether those are `.gitignore`d — if so, do the same here and document the regeneration command in the README instead.
- [ ] Commit the two `hybrid_retrieval_test*.ipynb` notebooks as-is (they're valuable prototype evidence) — clean them up properly in Epic 4 once the logic graduates into a real module.
- [ ] Verify continued access to `https://ollama.ux.uis.no` with your school API key — the course has likely wrapped up for the term, and university-provided endpoints sometimes get revoked shortly after grading closes. If it's gone, you'll need a fallback (self-hosted Ollama on the GPU servers you mentioned you have access to) before any epic below that needs new LLM calls (Epic 3's entity resolution, Epic 5).
- [ ] Delete the `feature/knowledge-graph-neo4j` placeholder branch (it has no commits of its own — it's just noise) unless you specifically want to revisit a Neo4j backend later, in which case leave a one-line note in this file instead.
- [ ] Note only, no action needed yet: `chunks_enhanced_hierarchical.json` was deleted from disk (still shows as `deleted` in `git status`). Not a blocker for KG work (it uses `chunks_fixed_size.json`, which is intact), but if any future ablation needs the `enhanced_hierarchical` chunking strategy again, it'll need regenerating via `pipelines.preprocessing_pipeline`.

---

## Epic 1 — Get `KGPipeline` actually running end-to-end

*The code has never been executed as a pipeline, only as a build script. Treat the first run as a debugging session, not a formality.*

- [ ] Add a `main_kg.py` entry point, mirroring the existing pattern in `main_agentic.py` (`--test-query`, `--eval`, `--eval-size`, `--rebuild-index`, `--output`).
- [ ] Run a single-query smoke test first (cheap, fast) to confirm `build_index()` loads the existing 37.5 MB graph correctly and `run_query()` produces a real answer without crashing.
- [ ] Fix whatever breaks. Likely candidates worth checking specifically if something fails: the `KGRetriever` entity-substring seed matching against real questions (short/common entity names causing false positive seeds), and the chunk UID format (`f"{pdf_name}::{chunk_id}"`) staying consistent between what `kg_builder.py` wrote into the graph and what `chunk_loader.load_preprocessed_chunks()` produces at retrieval time.
- [ ] Run a small eval subset (20–30 queries) through `KGPipeline.evaluate()`. Sanity-check the output isn't degenerate — retrieval metrics not all zero, `retrieved_docs` non-empty for most queries, timings in a sane range (BFS over a 77k-node graph should still be fast; if it's slow, that's worth knowing before scaling up).

---

## Epic 2 — Fair benchmark: KG-only vs. Systems 1–3

*This is the step that turns "I built a knowledge graph" into "I proved whether it helps."*

- [ ] Run `KGPipeline.evaluate()` on the **same test subset size and same `RANDOM_SEED`** as whatever Baseline/Advanced/Agentic were evaluated on (check `results_agentic.json` and `src/results/pipeline_results.csv` for the size actually used previously — the README mentions both `EVAL_SUBSET_SIZE=150` defaults and 1000-query ablation runs, so pin down which one is the real comparison baseline).
- [ ] Append the KG results row into the same comparison table/CSV so all four systems are visible side by side.
- [ ] Manually tag or filter a **multi-hop subset** of `test.jsonl` — questions that require chaining facts across more than one entity/fact (e.g. "what did the CEO of the company that acquired X say about Y"). MMDocIR won't have this labeled for you; skim the question set and hand-pick ~15–20 candidates. This subset is the whole point — it's where graph traversal is *supposed* to beat plain vector similarity, and where the interesting result lives.
- [ ] Write up findings: overall metric deltas (retrieval + generation) across all four systems, and a focused comparison on the multi-hop subset specifically. Include 2–3 concrete qualitative examples — the question, what Baseline retrieved, what KG retrieved, and which one was actually right.

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
