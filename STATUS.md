# STATUS

Single source of truth for where the project actually is. Updated at the end of every
work session. Read this first when resuming after a gap.

**Last updated:** 2026-09-29
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
| T3 · Repo scaffold + `.gitignore` + venv + deps | 0.5 h | todo |
| T4 · Select 20–30 contracts (size 16–65 KB, good clause coverage) | 1 h | todo |
| T5 · Ingestion: chunk → embed → FAISS index on disk | 2 h | todo |
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
| Text structure | CUAD txt has no paragraph breaks → window-based chunking | 2026-09-12 |
| Generator model (target pipeline) | Local Mistral 7B via Ollama; judge stays Gemini so they differ | 2026-09-29 |

## Decisions still open

| Question | Needed by | Notes |
|---|---|---|
| Chunk size / overlap | T5 | Pick one, log it as `pipeline_config_id`; do not compare strategies |
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
