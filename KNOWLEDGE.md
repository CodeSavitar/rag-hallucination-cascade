# KNOWLEDGE

What we have learned while building, as opposed to what we planned. Read this alongside:

- The original project plan (architecture, milestones, rigor bar) is kept private; this file records every decision that changed it
- `STATUS.md`: where we are right now (task board, decisions, session log)

This file holds the **why**, the **data facts**, and the **traps**, so nothing has to be
rediscovered after a gap.

**Last updated:** 2026-11-01 (M1–M2 done, M3 paused at T21)

## Resume here (state on 2026-11-01)

| Item | State |
|---|---|
| Done | M1, M2, and M3 up to T20: RAG test bench, 120-question seed set, L1, judge run, labelling page |
| **Next action** | **M6: L0 guardrails, deploy, full README, demo.** M5 done and T16 done 2026-11-17: repo public at https://github.com/CodeSavitar/rag-hallucination-cascade (remote `origin`). Personal notes and planning docs are kept local (gitignored). Headline: test 35% judged, 92% caught, $2.60 vs $7.08 per 1,000. Spend $2.28 of $15. |
| Start labelling | `.venv/bin/streamlit run src/label_app.py`, then open `http://localhost:8501`. The server does not stay running between sessions; progress is saved per answer |
| Labelling rules | Adhith may use the general wording guide (`src/label_guide.md`) and the "Not sure" tick. Individual answers are not discussed with any AI during T21, so the labels stay independent of the LLM judge. T22 reports agreement with and without the not-sure rows. **Calibration rows:** on 2026-11-14 Adhith's first 5 labels were reviewed outside the blind process (query_ids starting `88c27239`, `e1e43e17`, `42afd15e`, `b510a476`, `08aec7af`). Those 5 are excluded from the headline agreement, which is reported on the other 115, and all 120 are reported as a secondary number. No more per-answer review after this |
| Seed run | `run-20261007T150409Z-360c`: 120 answers, with L1 and judge verdicts stored |
| API spend | $0.85 of the $15 cap ($18 credit, no top-ups). T21 and T22 are free |
| Git | Last commit `51a21cc` (T15). All T16–T20 work is uncommitted, including `src/judge.py`, `src/budget.py`, `src/label_app.py` and `data/api_spend.csv` |
| Kept local | Personal notes and planning docs are gitignored and never committed |

**This file overrides the original plan where they differ.** Main changes: the judge is Claude
Sonnet 5.5 (not Gemini), labels are blind (not judge-first), and a hallucination means "not
supported by the excerpts".

---

## 1. The mental model

**What the product is.** An evaluator that answers one narrow question about a RAG response:
*is this answer actually supported by the documents that were retrieved?* It is hallucination
detection for RAG. It is not a general "judge any LLM output" tool.

**Three tiers, not two:**

| Tier | What | Role |
|---|---|---|
| L1 | Cheap heuristics (e.g. answer/context overlap) | Instant, free, settles the obvious cases |
| L2 | Trained classifier (XGBoost / logistic regression) | Handles the middle |
| L3 | LLM judge (Claude Sonnet 5.5) | Only for what L1 and L2 cannot settle |

**Why we build a RAG system at all (the "test bench").** An evaluator needs labelled examples
of *(question, retrieved context, answer, supported/hallucinated)*. L2 cannot train without
them, and the cascade cannot be measured without them. Nobody provides this dataset, so M1
builds a small **target RAG pipeline** whose only job is to produce answers for the evaluator
to grade. The contracts are the test bench, not the product.

**The dependency chain:**

```
T4  pick 30 contracts           ← the documents                    (DONE)
T5  chunk + embed + index them                                     (DONE)
T6  RAG answers questions       ← produces grounded AND hallucinated answers   (DONE)
T7  log every answer to SQLite (query_log)                          (DONE)
M2  ~120 questions from the contracts + L1 heuristics               (DONE)
M3  LLM judge grades answers, human labels all of them blind        (judge DONE, labels NEXT)
M4  train L2 on the judge's labels (80 train); human labels measure it
M5  chain L1 → L2 → L3, measure cost vs. recall
```

**Why CUAD contracts.** Lawyers annotated where 41 clause types appear in each contract. Each
annotated clause is a question with a known correct answer, so we get ground truth for free.
An **empty** clause (`[]`) is useful too: asking about it should get "not in the document",
and an invented answer is a hallucination to catch.

---

## 2. Decisions and their reasons

(The canonical list lives in `STATUS.md`. This section records the reasoning.)

| Decision | Reason |
|---|---|
| **Generator = local Mistral 7B via Ollama; judge from a different family (Gemini originally, Claude Sonnet 5.5 since 2026-10-09)** | If the same model family writes *and* grades answers, you get **self-preference bias**: judges favour their own family's text. Keeping them different avoids building a documented flaw in. Also: free to re-run 120 questions many times, and `weak_generator` induction becomes trivial (pull a heavier quantisation, e.g. `mistral:7b-instruct-q2_K`). |
| **30 contracts, not 510** | M3 hand-correction is per label. 30 × ~4 questions ≈ 120 is checkable; 510 is not. |
| **Size band 16–65 KB** | Corpus median is 32.4 KB, so the band is centred on it. Smaller contracts are thin; larger ones are mostly boilerplate and slow for a 7B model. Side effect: the richest contracts (coverage up to 33) are excluded, because they are all > 65 KB. |
| **Rank by clause coverage, take the top 30** | More annotated clause types means more checkable questions per contract. Simple ranking *also* covers all 41 types collectively, so no separate "collective coverage" optimisation was needed. |
| **No stratification by contract type** | The top 30 already span ~15 contract types (Distributor most common, ~6). Diverse enough. |
| **Embedding model = `Qwen/Qwen3-Embedding-0.6B`** | ~1.2 GB, 1024-dim, 32K-token input, so no chunk is ever truncated. The 4B version was rejected: ~8 GB next to Mistral on a 16 GB machine, and L1/L2 compute embedding similarities per answer, so a 4B embedder would make the "cheap" layers expensive. |
| **Chunks = 1000 chars, 200 overlap** | 99.1% of the 1,096 annotated clause spans in the 30 contracts are ≤ 1000 chars (median 246), so a chunk can hold almost any clause whole. ~1,750 chunks total. 1000 chars ≈ 250 tokens, so several chunks fit in Mistral's prompt. |
| **Cosine similarity (`IndexFlatIP` on normalised vectors)** | Scores are logged and become L1 features (`l1_max_similarity`). Cosine is bounded (-1 to 1, higher = closer) and comparable across questions; raw L2 distance is neither. |
| **Hand-written logic, guided not generated** | This is a portfolio project. Code Adhith can't defend under "why?" questioning has no interview value. |

---

## 3. CUAD: data facts

Location: `CUAD_v1/` (168 MB, gitignored; the quickstart will need a download step).

| Item | Fact |
|---|---|
| `full_contract_txt/` | 510 `.txt` files, sizes 0.6 KB – 330.3 KB, median 32.4 KB |
| `full_contract_pdf/` | 512 files (not used) |
| `master_clauses.csv` | 510 rows (one per contract), **83 columns** = `Filename` + 41 clause types × 2 |
| Column pairs | `<Clause>` holds the extracted text span(s); `<Clause>-Answer` holds Yes/No or a normalised value |
| Empty clause | The literal **string** `"[]"`; 14,208 such cells. No `NaN`s in clause columns |
| Filled clause | A string that *looks like* a list, e.g. `"['May 22, 2000']"`, not a real Python list |
| `CUADv1.json` | 40 MB SQuAD-format file; source of the `is_impossible` entries (useful for the `unanswerable` induction mode in M2; rewrite the templated question wording, do not reuse it) |

### Traps (each one cost or would have cost real time)

1. **Filename join.** The CSV lists `.pdf`/`.PDF` names while the folder has `.txt` files, so a
   direct join matches **0/510**. Join on the filename **stem**: 500/510 match. The remaining
   10 differ in punctuation/spacing. **Drop them; don't fuzzy-match.**
2. **Header typo.** `'Notice Period To Terminate Renewal- Answer'` has a space before
   `Answer`. `endswith("-Answer")` counts **42** clause types instead of 41. Match with
   `-\s*Answer$`, or check both spellings.
3. **Empty isn't empty.** `"[]"` has length 2 and is truthy, so `len(cell) > 0` or `if cell:`
   marks every clause present and every contract scores 41/41. Compare against `"[]"`.

---

## 4. T4 result: the selected 30

Output: `data/selected_contracts.csv` (columns `Filename` as `.txt` name, `size` in KB,
`clause_count`). Produced by `src/select_contracts.py`.

| Stage | Contracts |
|---|---|
| In CSV | 510 |
| Matched to a `.txt` | 500 |
| And within 16–65 KB | 253 |
| Coverage ≥ 15 / ≥ 18 / ≥ 20 | 107 / 46 / 16 |
| **Selected** | **30** |

- Coverage range of the 30: **19–24**. The 30th contract has coverage **19**.
- **Clean cut:** 16 contracts score above 19 and exactly 14 score 19, so 16 + 14 = 30. No ties
  straddle the cutoff, and the selection is identical on every run regardless of sort stability.
- All **41/41** clause types appear across the 30. The script `assert`s this and refuses to
  write the CSV if it fails. Verified: `head(2)` crashes on `'No-Solicit Of Customers'` with no
  file written.
- Worked example: eDiets Co-Branding Agreement has coverage 21/41.

## 4b. T5 result: the index

Built by `src/build_index.py` (~4 min, re-run any time); checked by `src/check_retrieval.py`.

| Output (`index/`, gitignored) | Contents |
|---|---|
| `chunks.faiss` | 1,744 vectors × 1024 dims, float32, normalised; `IndexFlatIP` (exact cosine) |
| `chunks.csv` | One row per vector, same order: `chunk_id`, `filename`, `chunk_index`, `start_char`, `end_char`, `text` |
| `config.json` | Model, dtype, chunk/overlap, `pipeline_config_id` = `qwen3-embedding-0.6b_c1000_o200`, build time |

- Every row's `text` equals `raw[start_char:end_char]`, verified after reload. No NaNs in the CSV.
- `chunk_id` = `<filename stem>::<3-digit index>`, stable across rebuilds.
- Retrieval check: 5/5 eDiets questions put the gold chunk at rank 1 with per-contract search.

---

## 4c. M2 seed set (T10–T11)

Built by `src/build_seed_set.py` from `data/question_templates.csv` → `data/seed_set.csv`.
Fixed random seed (42): rebuilding gives a byte-identical file.

| Item | Fact |
|---|---|
| Candidate pairs (15 clause types × 30 contracts) | 308 answerable, 142 unanswerable (absent clause, 12 types only) |
| Locating CUAD spans | 522/522 found: 487 exact, 35 by matching the pieces around `<omitted>` with flexible whitespace |
| Selected | 120 = natural 40, unanswerable 25, starved_retrieval 20, distractor 20, weak_generator 15 |
| Split | 80 train / 40 test; a third of each mode held out |
| Spread | 29 of 30 contracts (2–6 questions each), all 15 clause types (3–12 each). No (contract, clause) pair is used twice |
| Gold chunks | Every chunk overlapping the answer (1–9 per question; 2 is most common), **sorted by answer characters held, most first**. First id = the chunk that best contains the clause |
| Distractor | The other contract's best gold chunk only. Governing Law excluded: most contracts are Californian, so a distractor would usually agree with the real clause |

**Question design rule:** questions ask *what the clause says*, never *whether it exists*. For a
missing clause, "Does it have audit rights?" has a correct answer ("no"), so it isn't
unanswerable. "What are the audit terms?" has none.

---

## 5. Environment

| Item | Value |
|---|---|
| Python | **3.12** (Homebrew) in `.venv/`. The system `python3` is Apple's 3.9.6; don't use it |
| Key packages | `sentence-transformers` 6.1.0, `faiss-cpu` 1.15.1, `scikit-learn` 1.9.1, `pandas` 3.0.6, `ollama` 0.6.2 (client), `python-dotenv` 1.2.3, `anthropic` 1.12.1, `streamlit` 1.65.0 |
| Ollama | Installed via Homebrew (0.34.2). `mistral:latest` = Mistral 7B **v0.3**, Q4_K_M, 4.4 GB (generator). `mistral:7b-instruct-v0.3-q2_K`, 2.7 GB, pulled 2026-10-07 (weak_generator mode: same model, 2-bit) |
| Secrets | `.env` (gitignored, never committed) holds `ANTHROPIC_API_KEY`; `.env.example` committed with an empty `ANTHROPIC_API_KEY=` |
| `requirements.txt` | Full `pip freeze`, pinned. **Re-freeze after every new install** |

**Judge SDK (M3):** the judge is Claude, via `anthropic` 1.12.1 (pinned). The unused Google SDK
(`google-generativeai`) was removed. **Load `.env` with an explicit path**
(`load_dotenv(BASE_DIR / '.env')`): bare `load_dotenv()` crashes when code is piped in from a shell.
**Budget:** $18 credit, $15 hard cap, enforced by `src/budget.py`; check before every paid call.

---

## 6. Technical lessons

### Shell
- **`>` truncates the file before the command runs.** `pip freeze > requirements.txt` with
  `pip` missing left an empty file. For anything costly to rebuild, write to a temp file and
  move it into place on success.
- **Venv activation is per terminal tab.** A new tab has no `pip`. Either
  `source .venv/bin/activate`, or call `.venv/bin/pip` / `.venv/bin/python` by path, which
  works regardless of activation.
- When copying a command from chat, copy **only** the command. Stray trailing text made zsh
  fail with `parse error near '\n'`, and a parse error means *nothing* on that line ran.

### pandas 3
- **Relative `Path`s resolve against the current working directory**, not the script's folder.
  Build paths from `BASE_DIR = Path(__file__).resolve().parent.parent`.
- **`for x in df:` iterates over column names, not rows.** Use `df.iterrows()` for rows.
- **`df[col][i]` looks up the row *labelled* `i`, not the `i`-th row.** After dropping rows the
  labels have gaps, and lookups raise `KeyError`. `df.reset_index(drop=True)` renumbers from 0.
- **A column's dtype is fixed by its first values.** `df["size"] = ""` makes a string column,
  and pandas 3 then rejects floats. Initialise numeric columns with `0.0` / `0`.
- **Chained assignment never writes in pandas 3.** `df["col"][i] = v` writes into a temporary
  copy and is lost. Use `df.loc[i, "col"] = v` (it also creates the column if missing).
- **Assigning a scalar to a column fills every row.** `df["col"] = count` inside a loop is a bug.

### Checks
- A check that runs silently may be checking nothing. Prove it can fail: feed it a known-bad
  case (e.g. `head(2)`) and confirm it crashes.
- Use `assert` / `raise` for correctness checks, not `print`. A printed warning scrolls past
  and the bad output still gets written.
- Put the check **before** the write, and the write **outside** the loop.

---

## 7. Findings, decisions and traps from T5 onward

These are lessons from each task, in order, and they apply to every later script.

| Item | Notes |
|---|---|
| **Chunk size + overlap** | Decided 2026-10-02: 1000 / 200 chars, window-based. CUAD text *does* have blank lines (2,985 paragraphs across the 30) but they are unreliable: some fall mid-sentence at page breaks, one contract has 10 blocks of ~5,000 chars, 1,241 paragraphs are < 100 chars. Record the setting as `pipeline_config_id`; do not compare strategies. |
| **Chunk the raw text** | CUAD clause spans are substrings of the raw `.txt`. M2 finds gold chunks by character offset, so stripping or collapsing whitespace before chunking breaks the mapping. |
| **Qwen loads in bfloat16 by default** | Found in T5 Step 5: with `normalize_embeddings=True` the vector lengths still ranged 0.998–1.004 and cosine scores went above 1.0. Fix: `model_kwargs={"dtype": torch.float32}` → lengths exactly 1.0. Cost: ~2.3× slower (full build ~4 min on the M5). **T6 must load the query model the same way**, or query and chunk vectors won't be comparable at full precision. |
| **Import torch before faiss** | Found in T5 Step 6: torch, faiss and sklearn each bundle their own `libomp.dylib`. With `import faiss` before `torch`, loading the model segfaults (exit 139, no Python traceback). Reproduced both orders; torch-first works. Applies to every script that uses both, including T6. |
| **Retrieval must be scoped to one contract** | Found in T5 Step 7. Global search over all 30 fails both ways (tested on 3 questions × 2 phrasings): a generic question ("Which state's law governs the agreement?") mostly returns *other* contracts' clauses, since every contract has one (gold in the top 5 for 1 of 3, at rank 2); naming the parties gets the right contract but not the right clause, because the party names dominate the embedding and match every chunk of that contract (gold in the top 5 for 0 of 3). Searching only the source contract's chunks (`faiss.IDSelectorBatch` on its row positions) put the gold chunk at **rank 1 for 5/5** questions. Schema already has `source_contract_id`. **Decided 2026-10-06: scope to the source contract.** |
| **CUAD spans don't always `find()` in the raw text** | In eDiets, 4 of ~40 annotated spans return -1 on exact search: some contain `<omitted>` where CUAD cut text out, others differ in whitespace. M2's gold-chunk lookup needs a fallback (e.g. match the text before `<omitted>`), not plain `str.find`. |
| **rag-v1 prompt: rules go last** | T6 Step 3. Rules at the top: Mistral used the exact refusal sentence 1/3 times on unanswerable questions, paraphrasing the rest ("The contract does not provide…"). Rules after the question: 4/5. Answerable questions were unaffected. |
| **"Refuse, then guess"** | One answer began with the exact refusal sentence and then continued "However… it can be inferred that…". So refusal detection must check whether the answer *starts with* the phrase AND whether anything follows it. A refusal followed by a guess is a hallucination, not a refusal, and good M2/M3 material. Paraphrased refusals also occur, so L1 needs a small pattern list, not only the exact phrase. |
| **Repeated prompts understate latency** | T6 Step 4. Sending the identical prompt twice: 4,303 ms then 1,497 ms (and 3,811 → 470 ms), because Ollama reuses its cache for a prompt it has just seen. Only first-send latency is honest. Never log latency from a re-run of the same question. |
| **False refusal even with the right chunk** | T6 Step 6. "Is either party barred from challenging the other's trademarks?": the clause was the #1 retrieved chunk, yet Mistral answered with the refusal sentence, then a self-contradicting guess. So a refusal is not proof the answer was missing, and "retrieval hit + refusal" is its own failure type for the evaluator. |
| **Temperature 0 is not fully reproducible across runs** | T6 Step 6. The trademark answer came out with a different last sentence in two separate runs of `check_rag.py` (the other 7 answers were identical, and repeats within one process were identical). Likely small numeric differences on the GPU tipping a near-tie between words. Consequence: **labels attach to the logged `answer_text`, never to a regenerated answer.** Re-running a question gives a new row to label, not the same one. |
| **Failed rows: how they're stored** | T7. schema.sql says keep failed rows, but several columns are NOT NULL. Convention: fill what's known (config, model names, question), use empty values (`''`, `[]`) for what was never produced, and put the message in `error`. **Every analysis query must filter `WHERE error IS NULL`.** |
| **`question_id` = hash of contract + question text** | T7. Stable across runs with no lookup table; any wording change gives a new ID (correctly: it's a new question). M2 may pass its own IDs instead. The unique index on (`run_id`, `question_id`) blocks asking the same question twice in one run, but only at insert time, after the pipeline has already run. |
| **T8: what 10 real answers looked like** | 5 contracts × {governing law, assignment}, checked against CUAD's annotations. Governing law: 5/5 correct. Assignment: 2 clean; 1 "refuse, then guess" (2nd sighting, so it's systematic for this model + prompt, not a one-off); 2 "yes, but only if…" where the contract's rule is "no, except if…": facts grounded, headline backwards. Mistral also often prefixes "The answer is:", so refusal detection must not require the sentence at position 0. |
| **Cost query must stop at the terminating layer** | T9. schema.sql's first cost query summed every layer's cost for the cascade. In shadow mode every layer runs on every row, so on test data it reported the cascade at $0.0041/question vs $0.004 for judging everything: the cascade looked more expensive than the thing it replaces. Fixed: sum only the layers up to `terminating_layer`; judge-everything uses `shadow_mode = 1` rows only. Self-test in `src/sanity_queries.py` guards it. |
| **How the failure modes are wired (T12)** | `rag.answer(..., induction_mode, gold_chunk_ids, distractor_chunk_ids)`. starved_retrieval: ALL gold chunks excluded from the search, next 5 sent. distractor: all gold excluded, the other contract's best gold chunk injected, k stays 5. weak_generator: normal retrieval, `mistral:7b-instruct-v0.3-q2_K` writes (warmed up on first use, outside the timer). Injected chunks get their **real** cosine score and the list is re-sorted, so the logged scores carry no "planted at rank 1" fingerprint for L1/L2 to learn. |
| **`retrieval_hit` = best gold chunk retrieved** | Logged as 1/0 when gold exists, NULL for unanswerable. Uses `gold_chunk_ids[0]` (the chunk holding most of the answer), not "any gold chunk", because slivers that only touch the clause would count as hits. First test: a weak_generator row had a sliver retrieved but `retrieval_hit = 0`, so that row tests retrieval AND the weak model. In T13, report weak_generator results split by `retrieval_hit`. |
| **T13: the full seed-set run** | `run-20261007T150409Z-360c`, 120/120 answered, 21.5 min. Train-only behaviour: 52 answered, 13 clean refusals, 10 paraphrased refusals, 5 refused-then-guessed. `retrieval_hit`: natural 0.89, weak_generator 0.70, starved/distractor 0 (by design). Rough likely-wrong share 29/80 = 36%, so no rebalance. **Summary bug caught:** the first version counted paraphrased refusals as wrong (46%); declining in its own words is correct behaviour. |
| **"Unanswerable" ≠ topic absent** | 10 of 17 train unanswerable questions got real answers. CUAD marks a clause type absent only when no clause fits its narrow legal definition; our plain-English questions are broader, so other clauses can be relevant. Read by hand: some answers are grounded (Allied's license really is exclusive), others are **misreadings**: non-renewal notice given as termination without cause, breach termination as without-cause, joint IP *ownership* as revenue sharing, a survival clause as "warranties last forever". So unanswerable rows can NOT be auto-labelled as hallucinations; every one needs a real label in M3. |
| **Latency numbers from T13 were measured under load** | Median generation 6.0 s, p95 12.1 s, worst 31 s; retrieval ~200 ms (vs ~30 ms alone in T6). Mistral and Qwen share the GPU, and memory was 14.8/16 GB with ~9 GB swapped. Any latency quoted in M5 must say "16 GB laptop, both models loaded". |
| **L1 signals (T14)** | `src/l1_signals.py` writes every `l1_*` column except `l1_verdict` (T15), plus `latency_l1_ms`, `cost_l1_usd = 0`. Loads only the embedding model, never Mistral. Weakest sentence = each answer sentence's best cosine to any retrieved chunk (chunk vectors reused from the index), then the minimum. Token counts use Qwen's tokenizer. `l1_features_json` holds answer behaviour and the hedges / numbers found, so every count can be checked by reading. |
| **Unsupported-number check: two false-alarm sources fixed** | First pass flagged 6 answers; 5 were wrong: 4 **section references** ("Section 7(b)", "Clauses 1 and 10", "(9.9 Heading)") and 1 **number in words** in the contract ("sixty days" vs answer "60 days"). Now section references are stripped from the answer and spelled numbers (one–ninety-nine, incl. "twenty-four") count on both sides. After the fix: 1 flag in 120, and it's real: the weak model cited "the 28th excerpt" when it was given 5. Known gap: "one" in phrases like "one of the parties" counts as the number 1. |
| **First look: L1 separates the modes only weakly** | TRAIN rows that gave an answer: word overlap 0.89 natural vs 0.76–0.80 for every failure mode; weakest-sentence similarity 0.58 vs 0.51–0.55. **Distractor answers keep high overlap (0.79)**, as the plan predicted: the planted clause shares the vocabulary. These are mode averages, not labels; T15 sets verdicts, M3 says what's actually right. |
| **L1 verdict (T15)** | `l1_verdict()`: **fail** = refused then guessed, an unsupported number, or (real answers only) overlap < 0.60 / weakest sentence < 0.30; **pass** = clean refusal only; else **uncertain**. Train after the T19 refusal fix: 10 fail, 10 pass, 60 uncertain (L1 settles 25%). Cut-offs set from train with a stand-in for labels (good = natural + retrieval hit + answered, n=20; bad = answered with no answer in the excerpts, n=24) and are PROVISIONAL until M3. |
| **Why L1 never passes a real answer** | Good and bad overlap heavily: word overlap 0.67–1.00 good vs 0.54–0.97 bad. The strictest pass rule that passed anything (overlap ≥ 0.95 and weakest sentence ≥ 0.45) let through 5/20 good and 1 planted-clause answer (0.97 overlap). Passing one hallucination to save 5 judge calls is the wrong trade. **L1 can reject on evidence, but cannot approve.** |
| **Two rule-order bugs caught by reading the fails** | v1 checked low overlap before refusals, so correct refusals (which naturally share few words with the excerpts) were failed: first the fixed sentence, then 6 refusals in Mistral's own words. Now refusals are decided first, and overlap rules apply only to real answers. |
| **Judge (T18)** | `src/judge.py`: Claude Sonnet 5.5, prompt `l3-v1`, effort `medium`, `max_tokens` 6000, JSON schema via `output_config.format`. Sees question + excerpts actually sent + answer, never the gold clause. **No `fallbacks`** (a refusal could re-route to a pricier model outside the budget guard). Skips rows already judged with the same prompt version, so a re-run never pays twice. Refusal / truncation → `l3_verdict` NULL with the raw response kept. No `temperature` (rejected on this model), so the judge is not guaranteed deterministic. |
| **Judge test: ~$0.0075 and ~3 s per answer** | 5 train answers cost $0.0375 total (~2,800 tokens in, 100–250 out). Full 120 ≈ $0.90. One debatable verdict: a 90-day *non-renewal* notice accepted as "ending the agreement without cause". **Prompt deliberately not tuned to match Claude's opinion** before the human labels exist; that would contaminate the agreement measurement. |
| **T19: judge on all 120** | 120/120, 0 failed calls, $0.81 (project total $0.85), median 2.7 s, max 6.2 s. Train: 36 fail / 44 pass (45% fail; the rough T13 estimate was 36%). |
| **L1 bug found by comparing with the judge** | 3 answers gave a full answer and then appended "The answer is not in the provided contract text if the question…". The sorter called them clean refusals, so L1 passed 2 train answers the judge failed. Fix: the refusal sentence counts only within the first 20 characters. After re-scoring: train L1 fail 10 → judge fail 10/10; L1 pass 10 → judge pass 10/10; 60 uncertain. |
| **Faithful to the wrong excerpt** | The judge passed 6/13 planted-clause answers: they correctly report the planted clause, which comes from a different contract. Grounded in the excerpts, wrong about the contract. **Decided 2026-10-09 (option A): not a hallucination** — labels are judged against the excerpts; "correct for the real contract?" is recorded separately in the notes and reported as a retrieval limitation. No layer that only sees the excerpts can detect it. |
| **Labelling page (T20)** | `.venv/bin/streamlit run src/label_app.py`. Selects only question, contract, excerpts, answer and CUAD's gold clause, so no verdict, mode or split can leak (checked by scanning the rendered page). Fixed shuffled order (seed 7), resumes at the first unlabelled answer, no default selection, Save disabled until both questions are answered. Writes `human_label` (1 = not supported), `human_label_notes` JSON `{correct_for_contract, unhelpful, comment}`, `human_labeled_at`. `LABEL_DB_PATH` env var points it at a test copy. |
| **T22: blind judge-vs-human agreement is low** | `src/agreement.py` → `reports/judge_agreement.md`. Headline 57 answers (calibration rows out): raw agreement **67% (54–79%)**, **Cohen's κ 0.25 (0.00–0.49)**, "fair". Human: 15 not supported; judge failed 22; both on 9. Judge missed 6, judge extra 13. Test 73% vs train 55%. **Most disagreements share causes:** (1) **refusal rules applied differently**: in 5 cases the answer refused then guessed, which the rubric says to judge as a guess; Adhith marked them Supported, while judge and L1 failed them. In 3 cases a plain refusal came where the excerpts had the answer; he marked them Not supported (rubric: Supported + Unhelpful) and ticked Unhelpful 0 times in 60. (2) **Fine misreadings** the judge caught: "either party" for a one-sided audit right, "the parties" for a one-sided payment, a reversed grantor, a notice period taken from a change-of-control clause. (3) **Cut-off excerpts**: the judge is strict when an excerpt ends mid-sentence. These groupings are a first reading of the patterns; each case is adjudicated separately. L1 decided 17 sample answers and matched the human on 8; most mismatches are the same refusal cases. Blind figure stays the honest independent number; post-review figure is reported next to it, labelled as such (the reviewer sees the judge's reasoning, which pulls toward it). |
| **T23: L1's fail rule over-fires on new data** | On the 200 extra train answers, L1 said fail on 25 and the judge passed 8 of them (32%). On the original 80 train answers L1's 10 fails all matched the judge. The L1 thresholds were set on those 80 (T15, marked provisional), so this is the expected small-sample overfit showing up. Don't retune on test; revisit in M5 using train answers only (the 200 extra are fair game). L1 pass stayed clean: 18 of 18 judged pass. Also from T23: the weak Mistral model had been removed from the Mac and was re-downloaded; a stalled `ollama pull` resumes correctly when restarted. |
| **T25: what separates bad answers once L1 has taken the easy ones** | Train only, 217 L1-uncertain answers, judge labels, single-feature AUC: answer length **0.71** (bad answers average 81 tokens vs 52), mean similarity 0.65, average sentence support 0.65, sentence count 0.65; word overlap only **0.52** and best retrieval score 0.53 (L1 already removed the low-overlap ones, so what's left is all high-overlap). **Higher similarity goes with MORE hallucination**: bad answers echo the excerpts' wording while misreading them ("either party" for one party, a clause from the wrong section), so embeddings measure topic, not truth. Paraphrased refusals are less often bad. Near-duplicates dropped from the model: len_ratio (0.99 with length), answer_best_chunk_similarity (0.89 with mean sentence similarity). **Latency:** the new embedding features cost ~140 ms per answer on this Mac (3 encode calls), so the plan's "~5 ms" holds for the classifier only, not feature extraction. Still ~20× faster than the judge and free. |
| **T26: the operating point is 90% recall, not best F1** | Best-F1 threshold on the single signal flagged 76% of train answers, i.e. "send almost everything to the judge", which says nothing about a cascade. Every system is now compared at **90% recall of the judge's fails on train**: "to catch 90%, how few answers must be flagged?" For length ratio that's 155 of 217 (71%), precision 0.46 vs base rate 0.36. PR-AUC stays the threshold-free comparison. The choice was made before any test number existed. |
| **T27: L2 barely beats answer length on train** | Out-of-fold on the 217 train answers vs judge labels: L2 ROC-AUC 0.739, PR-AUC 0.543, flags 66% at 90% recall; length-ratio signal 0.713 / 0.536 / 71% (its threshold set in-sample, which flatters it). Paired bootstrap: ROC-AUC gap +0.026 (95% −0.027 to +0.081), PR-AUC +0.007 (−0.063 to +0.077): **not significant**. Calibration: low scores over-predict (0.18 predicted vs 0.09 actual), the middle under-predicts (0.46 vs 0.63); Brier 0.201 vs 0.231 baseline. Biggest coefficients (per std): retrieval score spread −0.52 (a clear winning chunk → fewer hallucinations), average sentence support +0.44 (the echo effect), answer length +0.44, question relevance +0.35, number count −0.33. The similarity features are correlated, so their individual signs (max −0.22 vs mean +0.28) shouldn't be over-read. Don't tune on test to close the gap; if L2 is to improve, the lever is a genuinely new signal (e.g. an entailment model), checked on train. |
| **T28: on test, L2 and answer length are a draw** | First and only use of the 40 test answers, vs reference labels. Whole systems at the train-fixed 90%-recall thresholds: L1+L2 catches 12 of 12 and flags 65% (precision 46%); L1+length catches 12 of 12 and flags 75% (40%); L1 alone catches 6 of 12 (flags 22%), so L1 alone misses half. Threshold-free on L2's 28-answer slice (6 positives): length is ahead (ROC-AUC 0.80 vs 0.71, PR-AUC 0.68 vs 0.34), but the gap's interval is −0.33 to +0.19. Read: at the operating point L2 saves ~10 points of flags, the same as on train (66% vs 71%); ranking-wise it's no better. Calibration on test is rough (top bin 0.59 predicted vs 0.29 actual, 7 answers per bin). Against Adhith's blind labels every system's recall drops (judge 70%, L1+L2 70%, L1+length 90%) because his labels disagree with the judge's refusal handling. L2 latency: median 140 ms, p95 232 ms. The judge-everything row is partly circular: the reference uses the judge's label wherever he agreed. Do not re-tune on these 40. |
| **T29–T30: the cascade and its curve** | Design: L2 never blocks on its own, it clears or escalates (test precision of L2 flags was 46%; a two-sided band wins a cost sweep by flagging everything free). One threshold, swept over every score value. Train uses L2's OUT-OF-FOLD scores (in-sample ones would flatter it). Operating point picked on train = fewest judge calls with ≥ 90% recall of judge fails: L2 threshold 0.293 → 46% of train answers judged. Read unchanged on test (40, reference labels): **35% judged (95% 20–50%), recall 92% (71–100%, i.e. 11 of 12), precision 69%, $2.60 per 1,000 answers vs $7.08 judging everything (−63%)**. Length signal at its own train-picked point: 45% judged, same recall, $3.27, so L2 saves ~10 points of judge calls again. L1 only: 50% recall, $0. Judge everything: 100% recall (partly circular vs the reference), precision 80%. Test intervals are wide (40 answers, 12 positives). |
| **T31: where the cascade leaks and where it over-flags** | At the L2 threshold 0.293. **Every miss is an L2 clearance**: 1 of 1 on test, 10 of 10 on train (out-of-fold). L1 never passed a hallucination (its pass = clean refusals only). **Every L1-layer false alarm comes from L1's fail rule**: 3 of 5 test false alarms, 8 of 8 on train (the over-firing rule from T23). Latency: median evaluator time 0.07 s on test (65% of answers stop before the judge) vs 2.65 s judging everything; the 95th percentile is the same (~3.3 s), because escalated answers still wait for the judge. Per mode on test: weak_generator 4 of 4 caught with only 1 judge call (L1's refused-then-guessed and number rules do the work); the miss is a natural-mode answer. Levers if this were tuned further (train only): L1's fail rule (false alarms) and L2's clearing threshold (misses). |
| **Qwen query prompt** | At T6, questions must be encoded with `prompt_name="query"`; chunks are encoded with no prompt. Forgetting it costs retrieval quality silently. |
| **`k` and `num_ctx`** | Decided at T6: `k=5`, `num_ctx=8192` set explicitly, temperature 0. `ollama ps` confirmed context 8192. |
| Chunk → contract mapping | Done in T5: `index/chunks.csv`, where row n is vector n, with contract and character offsets. |
| Index location | `index/` and `*.faiss` are gitignored; the index is regenerated from the corpus, never committed. |
| Model files | `*.pkl` / `*.joblib` are gitignored. Before shipping, consider `git add -f` on the **final** classifier so the quickstart runs without retraining. |
| M6 reproducibility | Quickstart must document the CUAD download (the corpus is gitignored). |
