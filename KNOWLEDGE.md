# KNOWLEDGE

What we have learned while building, as opposed to what we planned. Read this alongside:

- The original project plan (architecture, milestones, rigor bar) is kept private; this file records every decision that changed it
- `STATUS.md`: where we are right now (task board, decisions, session log)

This file holds the **why**, the **data facts**, and the **traps**, so nothing has to be
rediscovered after a gap.

**Last updated:** 2026-10-06 (M1 complete)

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
| L3 | LLM judge | Only for what L1 and L2 cannot settle |

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
M2  ~120 questions from the contracts + L1 heuristics
M3  LLM judge labels answers, human hand-corrects all of them
M4  train L2 on those labels
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
| **Generator = local Mistral 7B via Ollama; judge = Gemini** | If the same model family writes *and* grades answers, you get **self-preference bias**: judges favour their own family's text. Keeping them different avoids building a documented flaw in. Also: free to re-run 120 questions many times, and `weak_generator` induction becomes trivial (pull a heavier quantisation, e.g. `mistral:7b-instruct-q2_K`). |
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
| Key packages | `sentence-transformers` 6.1.0, `faiss-cpu` 1.15.1, `scikit-learn` 1.9.1, `pandas` 3.0.6, `ollama` 0.6.2 (client), `python-dotenv` 1.2.3, `google-generativeai` 0.8.6 |
| Ollama | Installed via Homebrew (0.34.2). `mistral:latest` = Mistral 7B **v0.3**, Q4_K_M, 4.4 GB (generator). `mistral:7b-instruct-v0.3-q2_K`, 2.7 GB, pulled 2026-10-07 (weak_generator mode: same model, 2-bit) |
| Secrets | `.env` (gitignored); `.env.example` committed with `GEMINI_API_KEY=` |
| `requirements.txt` | Full `pip freeze`, pinned. **Re-freeze after every new install** |

**Check before M3:** `google-generativeai` is Google's older Gemini SDK; the current one is
`google-genai`. Confirm which one to use before writing judge code, and pin it.

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

## 7. Open items and traps going into T6

(Rows marked *Found in T5* are lessons; they apply to every later script.)

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
| **Qwen query prompt** | At T6, questions must be encoded with `prompt_name="query"`; chunks are encoded with no prompt. Forgetting it costs retrieval quality silently. |
| **`k` and `num_ctx`** | Decide at T6. Check the context length Ollama actually uses for `mistral:latest` (the model supports 32,768, the runtime default is smaller). |
| Chunk → contract mapping | The index needs a side record of which chunk came from which contract and where, or retrieval results can't be traced back. |
| Index location | `index/` and `*.faiss` are gitignored; the index is regenerated from the corpus, never committed. |
| Model files | `*.pkl` / `*.joblib` are gitignored. Before shipping, consider `git add -f` on the **final** classifier so the quickstart runs without retraining. |
| M6 reproducibility | Quickstart must document the CUAD download (the corpus is gitignored). |
