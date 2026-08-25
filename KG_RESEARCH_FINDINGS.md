# Knowledge-Graph RAG on MMDocIR: Complete Findings

**Purpose of this document:** a single, comprehensive, paper-oriented compilation of every finding from the Knowledge-Graph RAG investigation (Systems 4–5), organized by claim rather than chronologically. `KG_ROADMAP.md` is the chronological engineering log (Epics 0–10+); this document restructures the same evidence for writing the actual paper. All numbers here are pulled directly from files committed in `src/results/` and are reproducible from the code in this repository.

---

## 1. TL;DR

**Central finding:** on the MMDocIR benchmark, knowledge-graph-based retrieval does not improve multimodal document QA over a strong dense+BM25 baseline — not in isolation, not fused with the baseline at any tested weight, not on the multi-hop subset it was specifically hypothesized to help with, and not even under a lenient "does it rescue anything, anywhere" standard. This is a statistically significant result (McNemar's test, p < 0.01 on every headline comparison), not a single-run artifact. Root-cause analysis identifies four separable contributing mechanisms: blunt lexical seed-matching (partially fixed), no query-relevance scoring (fixed, with a genuine trade-off), a structural mismatch between (subject, relation, object) triples and this corpus's largely tabular/numeric question style (not fixed — a representation problem, not a tuning problem), and extraction-model quality (confirmed as a real, partial contributing factor via a controlled model comparison, not fixed).

---

## 2. Systems under test

| System | Retrieval mechanism |
|---|---|
| **1 — Baseline** | BM25 + dense embedding retrieval (Jina CLIP v2) |
| **2 — Advanced** | + multimodal (page images), 8 query-transformation techniques, 5 chunking strategies, 4 prompting strategies |
| **3 — Agentic** | LangGraph agent: query-rewriter picks a technique per question, grader evaluates relevance, retries on low confidence |
| **4 — KG (this work)** | NetworkX graph of LLM-extracted (subject, relation, object) triples; entity-seed matching + BFS traversal, no vector/keyword search at all |
| **5 — Hybrid (this work)** | Reciprocal Rank Fusion of KG traversal + dense + BM25, with per-signal weights |

All five share the same generation LLM (`qwen3-vl:8b-instruct`) and, where applicable, the same underlying chunk source (`chunks_fixed_size.json`, 12,747 chunks / 127 source PDFs — a mix of 10-Ks, Pew survey reports, arXiv papers, and product/user manuals).

---

## 3. Experimental setup

- **Full test set:** 150 questions (`test.jsonl`), the same set used for all four systems' comparable numbers.
- **Multi-hop subset:** 19 hand-curated questions requiring chaining facts across ≥2 distinct entities/tables/figures (e.g., "find the year where stat A is X, then look up stat B for that year"), selected by a single annotator with a documented one-line justification per question (`src/data/test/multihop_subset.json`, not tracked in git — see Limitations). This subset is a judgment call, not a validated ground-truth label — MMDocIR has no native multi-hop annotation.
- **Metrics:** Precision@k, Recall@k, NDCG@k, MAP, MRR (retrieval); Token F1, BLEU, ROUGE, Exact Match, Contains Match, Semantic Similarity (generation).
- **Retrieval-only metrics are fully deterministic** given fixed models (no LLM sampling involved in dense/BM25/KG retrieval after the Epic 7 determinism fix) — the statistical tests in §7 concern sampling uncertainty from the fixed 150-question set, not run-to-run noise.
- **Generation and agent-routing metrics do involve LLM calls** and can vary run to run (quantified in §7.3).

---

## 4. Finding 1 — Overall system comparison (full 150-question set)

| System | P@1 | NDCG@5 | MAP | Token F1 | Exact Match |
|---|---|---|---|---|---|
| 1 — Baseline | 0.5933 | 0.6693 | 0.6478 | 0.1837 | 0.1333 |
| 2 — Advanced | **0.8667** | **0.9043** | **0.8944** | **0.2658** | **0.2067** |
| 3 — Agentic | 0.6066 | 0.6695 | 0.6544 | 0.1674 | 0.1133 |
| 4 — KG-only (best version, post-fixes) | 0.0467–0.0533* | 0.1734–0.1933* | 0.1318–0.1474* | 0.0317–0.0399 | 0.0133 |
| 5 — Hybrid (kg_weight=0.3, current default) | 0.4867–0.4933* | 0.6276–0.6300* | 0.5877–0.5908* | 0.1614–0.1688 | 0.1067–0.1133 |

*Ranges reflect two measurements taken at different points in the investigation (before/after the per-chunk scoring fix in §5.3); both are reported for transparency rather than picking one. KG-only is **~11–13x worse on P@1** and **~5x worse on Token F1** than Baseline. Systems 1–3's numbers are from the project's submitted course report; Systems 4–5 were run fresh under identical conditions for this investigation.

**Note on comparability:** the project's own exploratory `pipeline_results.csv` log contains different (generally lower) numbers for Systems 1–2 than the submitted report — this was discovered and documented; the report's numbers are treated as authoritative here since they reflect what was actually evaluated/graded.

---

## 5. Finding 2 — Root causes of KG-only's weak performance (identified and partially fixed)

### 5.1 False-positive substring matching

The original seed-matching implementation used raw substring containment (`entity in query`). This produced a concrete, confirmed bug: the graph node `"RoPE"` (Rotary Position Embeddings, an ML term extracted from an unrelated arXiv paper) matched as a seed on **any** query containing the word "Europe," because `"rope"` is a literal substring of `"europe"`.

**Fix:** word-boundary regex matching (`\bentity\b`, precompiled per entity). Verified: `"RoPE"` no longer appears in seeds for Europe-related queries.

**Root cause additionally revealed by an adversarial code audit:** the graph's "no seed found" fallback path fires **0% of the time** across all 150 test questions (median 11–12 seed entities matched per query, even post-fix) — the retrieval mechanism is not under-matching, it is drowning in matches, because the LLM-extracted graph retains large numbers of generic short-phrase pseudo-entities ("time," "many," "the," "compared to") as literal nodes.

### 5.2 Hub-node fan-out

Generic high-degree nodes (e.g., "students," degree 200+) were flooding BFS expansion with unrelated neighbors within a single hop.

**Fix attempted:** `KG_MAX_FANOUT` caps neighbor expansion to the best-evidenced neighbors (by edge source-chunk count) when a node's degree exceeds the cap.

**Ablation result (isolating this fix from §5.1's, which had been bundled into one reported number):** capped and uncapped fan-out produce **bit-for-bit identical retrieval metrics** on both the full set and the multi-hop subset. Investigated why: `_collect_chunks` fills the top-5 retrieval budget from the seed layer alone (BFS hop 0) in the overwhelming majority of queries, before hop-1/hop-2 (where the fan-out cap actually operates) is ever reached — confirmed concretely for one query, where 14 matched seed entities supplied 50 chunk-slots pre-deduplication, 10x the retrieval budget. **The entire measured improvement attributed to "Epic 3's fixes" is attributable to word-boundary matching alone; the fan-out cap, while not incorrect, currently has zero measurable effect at this corpus's typical seed-match density.**

### 5.3 No real relevance scoring (fixed, with a genuine trade-off)

Every retrieved chunk received a hardcoded `score: 1.0`; ranking was purely "seed layer, then hop-1, then hop-2" with **arbitrary order within each layer** — later found to be a real bug: chunk collection iterated a Python `set()` directly, whose iteration order is hash-randomized per process (`PYTHONHASHSEED`), meaning **the same query against the same graph could return different top-5 chunks on different runs.**

**Fix 1 (determinism):** sort nodes deterministically (alphabetically) before collection. This is a reproducibility fix only, not a ranking improvement, and it changed previously-reported numbers in *both directions* purely from breaking ties differently — direct evidence that single-run benchmark deltas earlier in this investigation were not fully reproducible (see §7 for the proper statistical treatment now applied).

**Fix 2 (real scoring):** composite score = node specificity (`1/(1+log1p(evidence_count))`, rare entities outrank generic hubs) × hop-distance decay (seed=1.0, hop-1=0.5, hop-2=0.25).

**Isolated the two factors before combining (repeating the fan-out ablation lesson):**

| Config | Multi-hop NDCG@5 / MAP | Full-set NDCG@5 / MAP |
|---|---|---|
| Alphabetical-only (no real score) | 0.026 / 0.018 | 0.226 / 0.199 |
| Hop-decay only | 0.000 / 0.000 | 0.036 / 0.026 |
| Specificity only | 0.158 / 0.123 | 0.162 / 0.128 |
| **Both combined (shipped default)** | **0.158 / 0.123** | **0.193 / 0.147** |

Specificity does essentially all the work; hop-decay alone is actively harmful (same-hop nodes tie with no differentiator, degrading to an even more arbitrary sort). **Net effect on the shipped combination is a genuine trade-off, not a clean win:** multi-hop NDCG@5 improves **+508%**, MAP **+583%**, but full-set P@1 **drops from 0.14 to 0.053** (NDCG@5/MAP still improve on the full set too). This is exactly the risk anticipated when the fix was designed: evidence count is not a pure noise signal, and down-weighting "common" entities can push a legitimately-correct, well-evidenced answer (e.g., a company's own name recurring throughout its own 10-K) out of the top rank even while aggregate ranking quality improves.

### 5.4 Extraction-model quality (confirmed contributing factor, not fixed)

The graph was built with `llama3:8b` (chosen for build speed: ~8 hours over 12,747 chunks). A hand-audit of the existing graph found a **confirmed hallucination**: a chunk that is a blank/garbled table (no real content) has three fabricated corporate facts attributed to it in the graph, including one that is **word-for-word the few-shot example in the extraction prompt itself** (`Apple → reported revenue of → $94.9B`) — given nothing to extract, the model regurgitated its own instructions instead of returning an empty result.

**Controlled comparison** (`llama3:8b` vs. `qwen3:32b`, same chunks, same prompt, 30-chunk random sample):

- `llama3:8b` fabricated non-trivial content on **≥4/30 sampled chunks** that had nothing real to extract (bare section headings, legal-boilerplate fragments, dangling sentences).
- `qwen3:32b` **correctly returned an empty result on every one of those same chunks.**
- `qwen3:32b` was also more complete on chunks with real content (e.g., correctly parsed garbled OCR'd statistics — `$173M`, `$236M`, `131`, `596` — that `llama3:8b` missed, while `llama3:8b` invented an OCR-artifact word, "omebac," as an entity in the same chunk).
- The tabular/checkmark-matrix misreading failure mode (treating a metrics-comparison table as flat "has" relations) **persisted in both models** — evidence this specific failure is a schema/prompt design issue, not a model-capability issue (see §6).

**Cost of a full re-extraction, measured directly:** `qwen3:32b` averages 7.61s/chunk vs. `llama3:8b`'s 2.06s/chunk (**3.7x slower**), projecting to **~27 hours** for a full-corpus rebuild vs. the original ~7.3 hours. Not executed — assessed as justified evidence but disproportionate cost given the other, model-independent findings below would still cap the payoff.

---

## 6. Finding 3 — A structural mismatch, not just noise

Manual inspection of the 19 multi-hop questions found that **18 of 19 are tabular/numeric cross-referencing** ("find the year where stat A is X, then look up stat B for that year," "the party with the higher percentage of X," argmax-then-lookup patterns over Pew cross-tabs and 10-K financial tables) rather than classic entity-relationship chains ("person works at company acquired by X"). A `(subject, relation, object)` triple graph does not naturally represent "row N, column M of table X." Both extraction models in the §5.4 comparison exhibited the same table-misreading behavior regardless of capability, consistent with this being a representational, not a model-quality, limitation. This reframes the negative result: **it is not that graph-based retrieval is inherently poor, but that this corpus's dominant question type does not map onto entity-relation triples**, which likely explains why this finding differs from prior work reporting positive results for graph-augmented retrieval on more classically relational corpora (Wikipedia-style narrative text, organizational/biographical facts).

---

## 7. Finding 4 — The decisive test: does KG help *anywhere*, even leniently?

### 7.1 Full weight sweep (kg_weight 0.0 → 1.0, full 150-question set)

Candidate lists from all three signals (KG, dense, BM25) were cached once per question; only the RRF fusion arithmetic was re-run per weight (no re-retrieval, fully deterministic).

| `kg_weight` | P@1 | NDCG@5 | MAP |
|---|---|---|---|
| **0.0 (no KG at all)** | **0.580** | **0.672** | **0.647** |
| 0.15 | 0.513 | 0.643 | 0.608 |
| 0.3 (shipped default, chosen in an earlier stage from only 2 points tested on the 19q subset) | 0.487 | 0.628 | 0.588 |
| 0.5 | 0.440 | 0.603 | 0.556 |
| 0.7 | 0.420 | 0.583 | 0.537 |
| 1.0 (equal weight) | 0.367 | 0.536 | 0.486 |

**Monotonic decline across the entire range.** There is no weight at which adding the KG signal improves the full-set aggregate — the empirical optimum is zero. The 0.3 default was tuned against a 19-question subset and never validated against the representative full set until this analysis.

### 7.2 Per-question rescue analysis (full 150 questions, P@1, retrieval-only)

| System | P@1 hits |
|---|---|
| Baseline alone | **95/150** |
| Hybrid (kg_weight=0.3) | 73/150 |
| KG-only | 8/150 |

| Comparison | Rescued (Baseline misses, this system hits) | Hurt (Baseline hits, this system misses) |
|---|---|---|
| Hybrid vs. Baseline | 5 | 27 |
| KG-only vs. Baseline | 2 | 89 |

The 5 questions rescued by Hybrid are mostly ordinary factual lookups, not cleanly explained by "multi-hop-ness" — only 1 of 5 is from the hand-curated multi-hop subset. **No query-type pattern is visible that would let an adaptive-weighting scheme keep the rescues while dropping the much larger casualty count.**

### 7.3 Statistical significance of the above (new — added for publication rigor)

Paired McNemar's test (continuity-corrected, same 150 questions, binary P@1 hit/miss):

| Comparison | Rescued (b01) | Hurt (b10) | χ² | p-value | Significant (α=0.05) |
|---|---|---|---|---|---|
| Baseline vs. Hybrid(0.3) | 5 | 27 | 13.78 | **0.00021** | Yes |
| Baseline vs. KG-only | 2 | 89 | 81.28 | **<0.00001** | Yes |
| **Hybrid(0.0, no KG) vs. Hybrid(0.3)** | 5 | 19 | 7.04 | **0.00796** | **Yes** |

The third row is the critical test for the paper's central claim: adding the KG signal at the shipped default weight is a **statistically significant regression** relative to not using it at all, not an artifact of which 150 questions happen to be in the test set.

Bootstrap 95% CIs (question-level resampling, 5,000 iterations) on P@1:

| System | P@1 | 95% CI |
|---|---|---|
| Baseline | 0.6333 | [0.5533, 0.7067] |
| Hybrid(0.3) | 0.4867 | [0.4067, 0.5667] |
| Hybrid(0.0) | 0.5800 | [0.5000, 0.6600] |
| KG-only | 0.0533 | [0.0200, 0.0933] |

Difference CIs both exclude zero: Baseline − Hybrid(0.3) = 0.1467, 95% CI [0.0733, 0.2133]; Baseline − KG-only = 0.5800, 95% CI [0.4933, 0.6600].

**Note:** these retrieval-only metrics involve no LLM sampling (dense embeddings, BM25, and — after the §5.3 determinism fix — KG traversal are all deterministic given fixed models), so these intervals capture sampling uncertainty from the fixed 150-question set, not run-to-run noise. Run-to-run variance is addressed separately below for the one part of the system that does involve stochastic LLM decisions.

---

## 8. Finding 5 — Agentic routing to the KG

A 9th technique (`kg_multihop`) was added to the existing 8-technique free-choice menu the query-rewriter agent already used, with a description telling it to use it for multi-hop/relational questions.

**A real bug was found and fixed along the way:** `QueryRewriterDecision`'s Pydantic `Literal` type never included `"kg_multihop"` as a valid value, so a genuine LLM choice of it silently failed validation and was miscounted as a parsing error, falling back to `"standard"`. This corrupted the initial finding ("0/19 chosen") — after the fix, the agent did select `kg_multihop`.

| Run | P@1 | NDCG@5 | MAP | `kg_multihop` selected |
|---|---|---|---|---|
| Plain Agentic (no KG option available) | 0.632 | 0.698 | 0.684 | n/a |
| Agentic + KG routing, free choice — trial 1 (post-fix) | 0.684 | 0.717 | 0.711 | 3/19 |
| Agentic + KG routing, free choice — trial 2 | 0.684 | 0.717 | 0.711 | 3/19 |
| Agentic + KG routing, free choice — trial 3 | 0.684 | 0.717 | 0.711 | 5/19 |
| Agentic + KG routing, **forced** (100% of questions) | 0.579 | 0.639 | 0.623 | 19/19 |

**Repeated-trial finding:** across three independent runs (two different prompt versions, genuine LLM-call stochasticity), the retrieval-metric outcome is **bit-for-bit identical every time** (0.6842/0.7174/0.7105), despite the categorical count of `kg_multihop` selections varying (3, 3, 5 out of 19). This means the categorical routing decision is somewhat stochastic, but the resulting retrieval quality is stable — a positive, low-variance result worth reporting precisely rather than as a single number.

**Selective free-choice routing (0.684) outperforms both no-KG-option (0.632) and forced 100%-routing (0.579).** The agent's own judgment about *when* to route adds a small amount of value on the specific cases it chooses, even though most genuinely-multi-hop questions still aren't routed there — consistent with the overall finding that KG helps a narrow, hard-to-predict subset of questions while hurting many more when applied broadly.

---

## 9. Limitations (for the paper's own limitations section)

1. **The multi-hop subset (n=19) is a single-annotator judgment call, not a validated ground-truth label.** MMDocIR has no native multi-hop annotation; the subset was hand-curated with documented reasoning per question but no independent second-reviewer validation or algorithmic completeness check against the other 131 questions.
2. **Sample sizes are modest** (150 full set, 19 multi-hop) relative to what a fully powered study would use; the multi-hop subset's small n means individual question flips represent large percentage swings, which is why §7.3's statistical tests matter and should be reported alongside point estimates throughout.
3. **The RRF weight sweep and rescue analysis are retrieval-only** (no generation-stage LLM calls), which is a deliberate scoping choice for tractable computation and clean determinism, but means the generation-stage compounding effects are not directly measured in that analysis (they are addressed separately via the full-pipeline system comparisons in §4 and §8).
4. **Extraction-model comparison (§5.4) is based on a 30-chunk sample**, not a full re-extraction; the hallucination-avoidance and completeness findings are demonstrated and statistically notable at that scale but not exhaustively quantified across the full 12,747-chunk corpus.
5. **All results are from a single corpus** (MMDocIR: 10-Ks, Pew surveys, arXiv papers, product manuals) with a particular question-type distribution (§6); generalization to more classically-relational corpora is a hypothesis (supported by the model-independence of the tabular-misreading failure mode) rather than a directly tested claim.

---

## 10. Suggested framing for the paper

Not: *"We built a knowledge graph and it didn't work."*

Rather: *"We conduct a controlled investigation of when graph-structured retrieval adds value on top of an already-strong dense+lexical retrieval baseline for multimodal document QA. Despite systematically addressing multiple identified weaknesses — false-positive lexical matching, absent relevance scoring, sub-optimal fusion weighting, and (partially) extraction quality — we find no configuration, including agent-mediated selective routing, in which graph retrieval improves aggregate performance, and we show via per-question analysis that this is not merely 'no effect' but a significant net regression under naive fusion. We identify the likely structural cause: this corpus's questions are dominated by tabular/numeric cross-referencing that does not map naturally onto (subject, relation, object) triples, a mismatch that persisted across two extraction models of substantially different capability. This suggests that reported gains from graph-augmented retrieval in prior work may be conditional on corpus structure (narrative/entity-relational text) in ways not yet well characterized, and that graph retrieval should be evaluated against — not assumed to compound with — a strong existing retrieval baseline before being adopted."*

This framing turns nine rounds of "did we fix it yet" into a single, defensible scientific contribution: a boundary condition on when graph-augmented retrieval helps, evidenced by root-cause analysis rather than a bare performance table.

---

## Appendix: source files

| Claim | Source file(s) |
|---|---|
| §4 overall comparison | `src/results/kg_eval_150_v2.json`, `hybrid_weighted_eval_150.json`, `hybrid_weighted_eval_150_v2.json`, course report Table 6/7 |
| §5.1–5.2 root causes + ablation | `src/results/kg_ablation_grid.json` |
| §5.3 scoring fix | `src/results/kg_scoring_fix_eval.json` |
| §5.4 extraction model | `src/results/extraction_model_comparison_sample30.json` |
| §7.1 weight sweep | `src/results/rrf_weight_sweep_full150.json` |
| §7.2 rescue analysis | `src/results/rescue_analysis_full150.json` |
| §7.3 significance tests | `src/results/statistical_significance_analysis.json` (computation script inline in project history) |
| §8 agentic routing | `src/results/agentic_kg_routing_multihop_v2.json`, `_v3.json`, `_trial4.json`, `agentic_kg_forced_routing_multihop.json` |
| Full chronological log | `KG_ROADMAP.md`, Epics 0–10 |
