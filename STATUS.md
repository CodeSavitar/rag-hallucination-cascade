# STATUS

Single source of truth for where the project actually is. Updated at the end of every
work session. Read this first when resuming after a gap.

**Last updated:** 2026-10-06
**Current milestone:** M1 — target pipeline + logging
**Total est. effort:** ~55 h across M1–M6

---

## Milestone board

| # | Milestone | Est. | State |
|---|---|---|---|
| M1 | Target RAG + logging schema | ~7 h | **in progress** |
| M2 | Seed set (~120 Q&A) + L1 heuristics | ~10 h | not started |
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
| T6 · Query path: question → retrieve top-k → generate | 2 h | todo |
| T7 · Wire logging — every query writes a complete row | 1 h | todo |
| T8 · Smoke test: 10 questions, verify no silent NULLs | 0.5 h | todo |
| T9 · Run the 4 sanity queries at the foot of `schema.sql` | 0.5 h | todo |

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

## Decisions still open

| Question | Needed by | Notes |
|---|---|---|
| Scope retrieval to the source contract? | T6 | Recommended yes: global search over 30 contracts missed every gold chunk in testing; per-contract search got 5/5 at rank 1. Mirrors production (user asks about one contract). See KNOWLEDGE §7 |
| Retrieval `k` + Ollama `num_ctx` | T6 | `k` × 1000 chars + prompt + answer must fit the context Ollama actually uses |
| Judge model + prompt version | M3 | Gemini primary; version the prompt from the first call |
| Which 41 clause types become natural questions | M2 | Favour contracts with high answerable coverage |

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
| 2026-10-06 | T5 Steps 1–6 done: 1,744 chunks, 1024-dim float32 vectors, `index/` written and verified on reload. Fixed bfloat16 precision and a torch/faiss import-order segfault. Step 7: `src/check_retrieval.py` passes 5/5 at rank 1 with per-contract search; global search fails | T6 — decide retrieval scope, then query path |
