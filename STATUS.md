# STATUS

Single source of truth for where the project actually is. Updated at the end of every
work session. Read this first when resuming after a gap.

**Last updated:** 2026-11-01
**Current milestone:** M3 — judge + blind hand-labelling + agreement (T16 push deferred)
**Total est. effort:** ~55 h across M1–M6

---

## Milestone board

| # | Milestone | Est. | State |
|---|---|---|---|
| M1 | Target RAG + logging schema | ~7 h | **done** (2026-10-06) |
| M2 | Seed set (~120 Q&A) + L1 heuristics | ~10 h | **done** (2026-10-08) |
| M3 | L3 judge + hand-correction + agreement rate | ~12 h | **in progress** |
| M4 | L2 classifier + calibration + baselines | ~10 h | **done** 2026-11-17 — L2 does not reliably beat answer length (see T27–T28) |
| M5 | Cascade + threshold sweep + dashboard | ~8 h | **done** 2026-11-17 — test: 35% judged, 92% caught, $2.60 vs $7.08 per 1,000 |
| M6 | Guardrails + deploy + README | ~8 h | **in progress** — T34–T39 below |

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
| T14 · L1 signals (overlap, similarity, unsupported numbers/dates, hedging, length, refusal) | 2.5 h | **done** — `src/l1_signals.py`; 120/120 rows scored, no NULLs, 61 ms median per answer, $0 |
| T15 · L1 verdict rules → l1 columns (thresholds provisional until M3 labels) | 1.5 h | **done** — `l1_verdict()` in `src/l1_signals.py`; train: 10 fail, 13 pass, 57 uncertain (L1 settles 29%; 10 / 10 / 60 = 25% after the T19 refusal fix) |
| T16 · **Push to GitHub** | 0.5 h | **done** 2026-11-17 — public at https://github.com/CodeSavitar/rag-hallucination-cascade. Personal notes and planning docs removed from all history and gitignored; they stay local. README credits CUAD (CC BY 4.0). Commit IDs quoted in older notes predate the history rewrite |

## M3 task board

| Task | Est. | State |
|---|---|---|
| T17 · Judge setup: Anthropic SDK, key in `.env`, model access check, **spending guard** (`src/budget.py`) | 0.5 h | **done** — Sonnet 5.5 accessible (1M context); guard tested; $0 spent |
| T18 · Judge prompt `l3-v1` (verdict, unsupported claim quote, reasoning; structured output) → L3 columns + cost + latency | 1.5 h | **done** — `src/judge.py`; tested on 5 train answers (one per rubric rule): $0.0375, ~3 s each, 4 clearly right + 1 debatable |
| T19 · Run the judge on all 120 (live calls, one at a time) | 0.5 h | **done** — 120/120 judged, 0 failed calls; $0.81 (total spend $0.85); median 2.7 s per answer. Train: 36 fail / 44 pass |
| T20 · Local Streamlit labelling page; judge verdict NEVER shown (blind) | 1.5 h | **done** — `src/label_app.py`; tested on a DB copy: save, resume, back, save-guard, blindness check |
| T21 · **Adhith labels 60: all 40 test + 20 stratified train** (cut from 120 on 2026-11-14 to save his time; question 2 optional; L2 trains on judge labels) | ~1.5 h | **done** 2026-11-14 — 60 of 60 (40 test, 20 train); 1 "not sure", 0 unhelpful |
| T22 · Agreement: raw %, Cohen's κ with bootstrap CI, disagreement breakdown, with and without the "Not sure" rows; headline excludes the 5 calibration rows (see KNOWLEDGE.md) | 1 h | **done** 2026-11-14 — 67% raw, κ 0.25 (0.00–0.49) on 57; `reports/judge_agreement.md` |
| T22b · Settle the 19 disagreements with blind tie-breaker models (ChatGPT, Gemini if needed; `reports/third_judge_blind.md`); report after-review agreement next to the blind figure as an upper bound | ~15 min | **done** 2026-11-16 — ChatGPT sided with the judge 13/19; after review 89% / κ 0.77 (as given), 93% / κ 0.85 (refusal rule applied); judge missed 0 after correction. **M3 done** |

## M6 task board (planned 2026-11-17)

Decided by Adhith 2026-11-17: the live URL lets a visitor **explore the results** (dashboard,
answer browser, try the L0 guardrail): no live model or API calls online. Hosted on **Streamlit
Community Cloud** (free).

| Task | Est. | State |
|---|---|---|
| T34 · L0 guardrails `src/l0_guardrails.py`: rule-based prompt-injection / jailbreak checks on the question; tested on hand-written attacks + benign look-alikes, and on the 320 real questions (false alarms) | 1 h | **done** 2026-11-17 — 6 regex rules; `l0_*` columns written for 320. Attacks the rules were written against: 19 of 19; 5 reworded attacks written afterwards: **0 of 5** (kept in the test file as known misses, so the reported figure is 19 of 24); benign look-alikes 0 of 11 flagged; real questions 0 of 320 flagged; ~0.004 ms each |
| T35 · Export app data to committed CSVs (per-answer table with every layer's output; out-of-fold L2 scores on train) | 1 h | planned |
| T36 · Public app `app/streamlit_app.py` with its own light requirements: results + slider, answer browser, try-L0 box | 2 h | planned |
| T37 · Deploy on Streamlit Community Cloud (Adhith signs in and clicks Deploy); check the live URL | 0.5 h | planned |
| T38 · Full README (problem → results → how → limits → quickstart); remove references to the private plan | 1.5 h | planned |
| T39 · Demo video script (Adhith records) | 0.5 h | planned |

**M6 is done when:** a stranger can open the live URL and explore the results, and the README
lets them understand the project in 90 seconds and run it locally.

## M5 task board (planned 2026-11-17)

**Design decision (2026-11-17):** L2 never blocks an answer on its own. It either clears it
(score below the threshold) or escalates it to the judge. Reason: on test only 46% of L2's flags
were real, so letting L2 block would reject many good answers, and a two-sided band "wins" a
cost sweep by flagging everything for free. One threshold → one clean curve. L1 keeps its rules.

| Task | Est. | State |
|---|---|---|
| T29 · Cascade router `src/cascade.py`: per answer → terminating layer, verdict, cost, latency; mark the 320 answers shadow_mode = 1 (every layer ran) | 1.5 h | **done** 2026-11-17 — `route(rows, score_column, threshold)`; self-test on 5 hand-made answers (layer, verdict, cost, latency, both extremes); 320 rows marked shadow |
| T30 · Threshold sweep → cost-vs-recall curve for L2 and for the length signal, plus the judge-everything and L1-only points. Operating point picked on TRAIN (fewest judge calls with ≥ 90% recall), then read on TEST with intervals | 2 h | **done** 2026-11-17 — `src/sweep.py` → `reports/cascade_curve.csv`, `cascade_curve_{train,test}.svg`, `data/operating_point.json`. L2 threshold 0.293 (train: 46% to judge, 91% recall, out-of-fold). **Test: 35% to the judge (20–50%), recall 92% (71–100%), precision 69%, $2.60 per 1,000 vs $7.08 judging everything.** Length signal at its own train point: 45% to the judge, same recall |
| T31 · Accounting at the operating point: cost per 1,000 answers, latency p50/p95, miss attribution (which layer let each miss through), detection per failure mode | 1.5 h | **done** 2026-11-17 — `src/cascade_report.py` → `reports/cascade_accounting.md`; `terminating_layer` + `cascade_verdict` written for 320. Test: stops L1 30% / L2 35% / judge 35%; evaluator latency median 0.19 s vs 2.65 s judging everything; p95 3.50 s vs 3.24 s (escalated answers pay L1 + L2 + judge). Latency corrected 2026-11-17: the first run left L2's time out; the 1 miss was cleared by L2; 5 false alarms = L1 3 + judge 2. Train: all 10 misses cleared by L2, all 8 false alarms from L1 |
| T32 · Streamlit dashboard: curve with an operating-point slider, confusion matrix, cost, latency, escalation, per-mode | 2 h | **done** 2026-11-17 — `src/dashboard.py` (`.venv/bin/streamlit run src/dashboard.py`, port 8502); checked in the browser: numbers match T30–T31 on test and train, slider updates, no errors (superseded in T36 by `app/streamlit_app.py`; `src/dashboard.py` removed) |
| T33 · Write-up: `reports/cascade_results.md`, headline sentence | 1 h | **done** 2026-11-17 — `reports/cascade_results.md`. **M5 done** |

**M5 is done when:** the cost-vs-recall curve exists and the headline ("at 90% recall, the judge
sees Y% of answers, costing $Z per 1,000") can be read off it, with intervals.

## M4 task board (planned 2026-11-16; all 4 decisions approved by Adhith the same day)

| Task | Est. | State |
|---|---|---|
| T23 · **More training data (decision 1):** ~200 new TRAIN questions on the same contracts, same mode mix, run through RAG + L1 + judge. Judge labels are the L2 training labels. ~$1.50 of API, ~1 h machine time, none of Adhith's. Test set untouched | 1.5 h | **done** 2026-11-17 — `data/extra_train_set.csv` (200 rows, `src/build_extra_train_set.py`, seed 43) → run `run-20261117T035058Z-a4e7` (pinned as `query_logger.EXTRA_RUN_ID`; seed run pinned as `SEED_RUN_ID`): 200/200 answered, 0 failed, 25 min; L1 25 fail / 18 pass / 157 uncertain; judge 70 fail / 130 pass, $1.43 (total $2.28). **L2 training set: 217 L1-uncertain train answers (79 fail / 138 pass)**, up from 60 |
| T24 · Reference labels for the 40 TEST answers (decision 2): human label where human and judge agreed, rule-corrected tie-break where they didn't → `data/reference_labels.csv` | 0.5 h | **done** 2026-11-17 — `src/build_reference_labels.py`: 29 agreed, 10 tie-break, 1 human-only (calibration row `88c27239`, no tie-break vote); 12 of 40 not supported (human 10, judge 15). **Only 6 positives among the 28 L1-uncertain test answers** |
| T25 · L2 features (`src/l2_features.py`): the existing L1 sub-scores + answer↔excerpts embedding similarity, refused-then-guessed flag, sentence count. Banned: induction_mode, retrieval_hit, gold chunks, anything from CUAD's answer | 1.5 h | **done** 2026-11-17 — `src/l2_features.py` → `data/l2_features.csv` (320 rows × 14) and `l2_features_json`; 12 used by the model (2 near-duplicates dropped). New features take ~140 ms per answer. Train check: best single signal is **answer length (AUC 0.71)**; word overlap ~0.52 on L1-uncertain answers; higher similarity goes with MORE hallucination |
| T26 · Baselines: always escalate, L1 only, best single signal (threshold picked on train) | 1 h | **done** 2026-11-17 — `src/baselines.py` → `data/baselines.json`. Systems: judge_everything, l1_only, l1_plus_signal. Signal chosen by rule = `l1_len_ratio` (train AUC 0.713; answer length 0.712 is a tie). **Operating point = 90% recall** (F1 rule rejected: flagged 76%). At 90% recall the signal flags 155 of 217 train answers (71%), precision 0.46 vs a 0.36 base rate: the bar L2 must clear. No test numbers computed yet |
| T27 · L2 = logistic regression (decision 4), trained only on answers L1 leaves uncertain (decision 3); C picked by cross-validation; Platt calibration; model saved to `models/` | 1.5 h | **done** 2026-11-17 — `src/train_l2.py` → `models/l2.joblib` (gitignored), `data/l2_model.json`, `reports/l2_train_cv.md`. C = 0.187, Platt. Out-of-fold on train vs judge: ROC-AUC 0.739 vs signal 0.713, PR-AUC 0.543 vs 0.536, flags 66% vs 71% at 90% recall. **Gap not significant** (bootstrap ROC-AUC +0.026, 95% −0.027 to +0.081; PR-AUC +0.007, −0.063 to +0.077). Calibration rough (Brier 0.201 vs 0.231 for always guessing the base rate) |
| T28 · Evaluate on test (L1-uncertain rows first, all 40 second): precision, recall, PR-AUC with bootstrap CIs vs the 3 baselines; reliability curve; coefficients → `reports/l2_results.md`; write `l2_*` columns to query_log. Added 2026-11-17: also 5-fold cross-validated results on the 217 train answers against judge labels (larger n, measures how well L2 imitates the judge), since the test set has only 6 positives where L2 operates | 2 h | **done** 2026-11-17 — `src/evaluate_l2.py` → `reports/l2_results.md`, `reports/l2_reliability_test.svg`; `l2_*` written for the 245 L1-uncertain answers. Test, all 40, at the train-fixed 90%-recall thresholds: L1+L2 caught 12/12 flagging 65%; L1+signal 12/12 flagging 75%; L1 alone 6/12 flagging 22%; judge 12/12 flagging 38% (partly circular). L2 slice (28, 6 positives): ROC-AUC L2 0.71 vs signal 0.80, PR-AUC 0.34 vs 0.68; gap −0.08 (95% −0.33 to +0.19). **No reliable winner. M4 done** |

**M4 is done when:** L2's precision/recall/PR-AUC on the test set is reported against all three
baselines with intervals, a reliability curve is plotted, and the coefficients are explained. If L2
doesn't beat the best single signal, that is written up as the finding.

**M3 is done when:** 60 human labels exist (all 40 test + 20 train, decided 2026-11-14), and judge-vs-human agreement is reported with a
confidence interval.

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
| **Judge = Claude Sonnet 5.5** (was Gemini) | Adhith's Claude API credit instead of a new Google key. Different family from Mistral (no self-preference). ~$1.70 per full 120-answer run. Live calls, not the batch API, so latency is real | 2026-10-09 |
| **API budget: $18 total, hard cap $15** | Credit can't be topped up. Every paid call goes through `src/budget.py` (worst-case check before, real cost logged after, ledger `data/api_spend.csv`). $3 kept back for estimation error and the M6 demo | 2026-10-09 |
| Blind labelling | Adhith labels without seeing the judge's verdict; agreement computed afterwards. Avoids anchoring, which would inflate agreement | 2026-10-09 |
| Rubric: grounded-but-misleading | **Hallucination** (label 1): the headline claim isn't supported, even if every detail is borrowed | 2026-10-09 |
| Rubric: false refusal | **Not a hallucination** (label 0), noted "unhelpful" in `human_label_notes`, counted separately | 2026-10-09 |
| What the judge sees | Question + the exact excerpts sent + answer. Never CUAD's gold clause (production conditions). The human labeller does see the gold clause | 2026-10-09 |
| Labelling tool | Local Streamlit page, saves straight to `query_log`, batches of 20, progress shown | 2026-10-09 |
| **Label definition = A: supported by the excerpts** | `human_label` 1 = the answer makes a claim the excerpts don't support; judged against the excerpts the generator saw. A second field, "correct for the real contract? yes / no / can't tell" (from CUAD's clause), goes in `human_label_notes` and is never used for training. Reason: an answer faithful to a planted wrong-contract excerpt is a retrieval error that no excerpt-only layer can see; counting it would make recall measure retrieval, not the evaluator. It becomes a README limitation instead | 2026-10-09 |

## Decisions still open

| Question | Needed by | Notes |
|---|---|---|

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
| 2026-10-07 → 10-08 | M2 done: T10 questions, T11 seed set (120, 40 held out), T12 failure modes, T13 full run (36% likely wrong), T14 L1 signals, T15 L1 verdict (settles 29% of train) | T16 — push to GitHub, then M3 |
| 2026-10-09 → 11-01 | T17–T20 done: Claude Sonnet 5.5 judge, $15 spending guard, all 120 judged ($0.85 total), L1 refusal bug fixed, label definition A locked, blind labelling page built. Project paused ~3 weeks (trip, illness). All T16–T20 work still uncommitted | T21 — Adhith labels 120 answers |
