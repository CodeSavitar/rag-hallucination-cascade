# STATUS

Single source of truth for where the project actually is. Updated at the end of every
work session. Read this first when resuming after a gap.

**Last updated:** 2026-10-07
**Current milestone:** M2 — seed set (~120 Q&A) + L1 heuristics
**Total est. effort:** ~55 h across M1–M6

---

## Milestone board

| # | Milestone | Est. | State |
|---|---|---|---|
| M1 | Target RAG + logging schema | ~7 h | **done** (2026-10-06) |
| M2 | Seed set (~120 Q&A) + L1 heuristics | ~10 h | **in progress** |
| M3 | L3 judge + hand-correction + agreement rate | ~12 h | not started |
| M4 | L2 classifier + calibration + baselines | ~10 h | not started |
| M5 | Cascade + threshold sweep + dashboard | ~8 h | not started |
| M6 | Guardrails + deploy + README | ~8 h | not started |

---

## M1 task board

Tasks are sized to finish inside one sitting. Never stop mid-task — stop between them.

| Task | Est. | State |
|---|---|---|
| T1 · Results table (`schema.sql`) | 1 h | **done** — 69 cols, validated |
| T2 · Acquire CUAD corpus | 0.5 h | **done** — `CUAD_v1/`, 510 contracts |
| T3 · Repo scaffold + `.gitignore` + venv + deps | 0.5 h | **done** — Py 3.12 venv, pinned reqs, `c2d6d96` |
| T4 · Select 30 contracts: 16–65 KB, top-30 by clause coverage, all 41 types asserted → `data/selected_contracts.csv` | 1 h | **done** — `src/select_contracts.py`; #30 coverage = 19, clean cut (16 above + 14 at 19) |
| T5 · Ingestion: chunk → embed → FAISS index on disk | 2 h | **done** — `src/build_index.py` (1,744 chunks) + `src/check_retrieval.py` (5/5 rank 1) |
| T6 · Query path: question → retrieve top-k → generate | 2 h | **done** — `src/rag.py` `answer()`; `src/check_rag.py`: plumbing 8/8, 4/5 answerable correct, 3/3 exact refusals |
| T7 · Wire logging — every query writes a complete row | 1 h | **done** — `src/query_logger.py` → `data/query_log.db`; 23/69 columns filled per row (rest = evaluator layers, labels); failures kept with `error` |
| T8 · Smoke test: 10 questions, verify no silent NULLs | 0.5 h | **done** — `src/smoke_test.py`: 10 rows over 5 contracts, 0 errors, 0 silent NULLs; governing law 5/5, assignment 2 clean / 3 flawed |
| T9 · Run the 4 sanity queries at the foot of `schema.sql` | 0.5 h | **done** — `src/sanity_queries.py`: self-test on 10 hand-made rows passes; found and fixed a cost-query bug (cascade showed as dearer than judge-everything) |

## M2 task board

| Task | Est. | State |
|---|---|---|
| T10 · One natural question per clause type (15 types) → `data/question_templates.csv` | 1 h | **done** — approved by Adhith; questions ask for details, not yes/no |
| T11 · Seed set → `data/seed_set.csv`: contract, clause type, question, mode, gold chunk ids, train/test split | 2 h | **done** — `src/build_seed_set.py`; 120 questions, 29 contracts, 15 types, 80 train / 40 test, reproducible |
| T12 · Failure modes: starved_retrieval, distractor, weak_generator | 2 h | **done** — `rag.answer(induction_mode=…)`; tested 1 question per mode; `retrieval_hit` now logged |
| T13 · Run the ~120 into `query_log`; check the share of wrong answers (target 30–40%) | 1 h | **done** — run `run-20261007T150409Z-360c`: 120/120 answered, 0 failed, 21.5 min. Rough likely-wrong share (train): 29/80 = 36%. No rebalance |
| T14 · L1 signals (overlap, similarity, unsupported numbers/dates, hedging, length, refusal) | 2.5 h | todo |
| T15 · L1 verdict rules → l1 columns (thresholds provisional until M3 labels) | 1.5 h | todo |
| T16 · **Push to GitHub**. First decide what stays private. Add CUAD attribution to README | 0.5 h | todo |

**M2 is done when:** every failure mode is represented in `query_log`, and L1 writes
pass / fail / uncertain with every sub-score logged.

**M1 is done when:** one question in → one fully populated `query_log` row out, and the
sanity queries return sensible results.

---

## Decisions locked

| Decision | Choice | When |
|---|---|---|
| Project over Tamil RAG | LLM Eval first; Tamil second | 2026-09-12 |
| Corpus | CUAD (`full_contract_txt/`) | 2026-09-12 |
| Label source | Judge-generated, fully hand-corrected; disagreement rate published | 2026-09-12 |
| Headline metric | Cost-vs-recall curve, quoted at a stated operating point | 2026-09-12 |
| Regression harness | Absorbed into this project as optional diff mode, post-M6 | 2026-09-12 |
| Schema shape | One denormalised `query_log` table, layer columns pre-declared | 2026-09-12 |
| Text structure | CUAD txt paragraph breaks are unreliable (mid-sentence page breaks, 5,000-char blocks) → window-based chunking | 2026-09-12, reason corrected 2026-10-02 |
| Generator model (target pipeline) | Local Mistral 7B via Ollama; judge stays Gemini so they differ | 2026-09-29 |
| Embedding model | `Qwen/Qwen3-Embedding-0.6B` (1024-dim, 32K-token input); queries encoded with `prompt_name="query"`, chunks without | 2026-10-02 |
| Chunking | 1000-character windows, 200 overlap, on raw text (no whitespace cleaning) | 2026-10-02 |
| Similarity | Cosine: normalised vectors in a FAISS `IndexFlatIP` | 2026-10-02 |
| Retrieval scope | Search only the source contract's chunks (`IDSelectorBatch`); searching all 30 contracts found the gold chunk in the top 5 for only 1 of 3 generic questions (0 of 3 with party names), vs 5/5 at rank 1 scoped | 2026-10-06 |
| Generation settings | `k=5` (k=1 reserved for `starved_retrieval`), `num_ctx=8192` set explicitly, temperature 0, prompt `rag-v1` (answer only from context, else say not found) | 2026-10-06 |
| M2 clause types | 12 types present in ≥10 and absent in ≥8 of the 30 (Renewal Term, Audit Rights, Exclusivity, Minimum Commitment, Non-Compete, Post-Termination Services, Uncapped Liability, Change Of Control, Revenue/Profit Sharing, Insurance, Warranty Duration, Termination For Convenience) + Governing Law, Expiration Date, Anti-Assignment as answerable only | 2026-10-07 |
| M2 question wording | One question drafted per type, then edited by Adhith. CUAD's template wording not reused | 2026-10-07 |
| M2 mode mix (120) | natural 40, unanswerable 25, starved_retrieval 20, distractor 20, weak_generator 15; rebalance after T13 if the wrong-answer share is off | 2026-10-07 |
| starved_retrieval | Remove the gold chunk(s), send the next 5. Not k=1: gold ranked #1 in every T5/T6 test, so k=1 would still contain the answer | 2026-10-07 |
| distractor | Same clause type from a different contract, swapped in for the gold chunk | 2026-10-07 |
| weak_generator | `mistral:7b-instruct-v0.3-q2_K`: same model as the generator (v0.3), 2-bit instead of 4-bit | 2026-10-07 |
| Train/test split | 40 of 120 held out now, stratified by mode, stored in `seed_set.csv`; never used to set L1 thresholds or train L2 | 2026-10-07 |

## Decisions still open

| Question | Needed by | Notes |
|---|---|---|
| Judge model + prompt version | M3 | Gemini primary; version the prompt from the first call |
| Is a grounded-but-misleading answer a hallucination? | M3 | T8 example: the clause says "neither party may assign, except on a merger"; the answer says "yes, either party can assign… but only on a merger". Every fact is in the text, but the headline is backwards. The judge rubric and your labels need one rule for this |

## Known risks

- **Gap risk is the main threat, not hours.** Re-orientation cost after a long break is what
  kills variable-schedule projects. Mitigation: this file, plus tasks sized to one sitting.
- M3 hand-correction is the likeliest overrun. It cannot be delegated or skipped.
- CUAD's 14,208 `is_impossible` entries are a large shortcut for the `unanswerable` induction
  mode — but the templated question wording must be rewritten, not reused.

---

## Session log

| Date | Done | Next |
|---|---|---|
| 2026-09-12 | Plan rewritten, project chosen, kickoff brief published, `schema.sql` built and validated, CUAD acquired and analysed | T3 — repo scaffold |
| 2026-09-29 | T3 done: venv, deps (incl. ollama client), README, `.env.example`, package dirs. Generator locked to local Mistral. T4 selection measured and defined | T4 — `src/select_contracts.py` |
| 2026-09-30 | T4 done: 30 contracts selected, 41/41 clause types covered, check asserts and blocks the write on failure | T5 — chunk, embed, FAISS index |
| 2026-10-02 | T5 broken into 8 steps; Step 0 decisions locked (Qwen3-Embedding-0.6B, 1000/200 chunks, cosine) | T5 Step 1 — `src/build_index.py` |
| 2026-10-06 | T5 Steps 1–6 done: 1,744 chunks, 1024-dim float32 vectors, `index/` written and verified on reload. Fixed bfloat16 precision and a torch/faiss import-order segfault. Step 7: `src/check_retrieval.py` passes 5/5 at rank 1 with per-contract search; global search: gold in top 5 for 1/3 generic, 0/3 named | (see next row) |
| 2026-10-06 (cont.) | T6 done: `src/rag.py` (retrieve → rag-v1 prompt → Mistral, warm-ups, `answer()` returns query_log fields) + `src/check_rag.py`. Prompt rules moved after the question (exact refusals 1/3 → 4/5). Found "refuse, then guess" and a false refusal with the right chunk at #1 | (see next row) |
| 2026-10-06 (cont.) | T7–T9 done → **M1 complete**. `query_logger.py`, `smoke_test.py` (10 rows, 0 silent NULLs), `sanity_queries.py` (self-test passes; cost query fixed) | M2 — plan the seed set and L1 |
