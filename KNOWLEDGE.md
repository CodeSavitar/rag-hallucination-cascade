# KNOWLEDGE

What we have learned while building, as opposed to what we planned. Read this alongside:

- The original project plan (architecture, milestones, rigor bar) is kept private; this file records every decision that changed it
- `STATUS.md`: where we are right now (task board, decisions, session log)

This file holds the **why**, the **data facts**, and the **traps**, so nothing has to be
rediscovered after a gap.

**Last updated:** 2026-10-06 (after T5)

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
T6  RAG answers questions       ← produces grounded AND hallucinated answers
T7  log every answer to SQLite (query_log)
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

## 5. Environment

| Item | Value |
|---|---|
| Python | **3.12** (Homebrew) in `.venv/`. The system `python3` is Apple's 3.9.6; don't use it |
| Key packages | `sentence-transformers` 6.1.0, `faiss-cpu` 1.15.1, `scikit-learn` 1.9.1, `pandas` 3.0.6, `ollama` 0.6.2 (client), `python-dotenv` 1.2.3, `google-generativeai` 0.8.6 |
| Ollama | Installed via Homebrew; `mistral:latest` (4.4 GB) already pulled |
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
| **Retrieval must be scoped to one contract** | Found in T5 Step 7. Global search over all 30 fails both ways: a generic question ("Which state's law governs the agreement?") returns *other* contracts' clauses, since every contract has one; naming the parties gets the right contract but not the right clause, because the party names dominate the embedding and match every chunk of that contract. Searching only the source contract's chunks (`faiss.IDSelectorBatch` on its row positions) put the gold chunk at **rank 1 for 5/5** questions. Schema already has `source_contract_id`. Decide at T6. |
| **CUAD spans don't always `find()` in the raw text** | In eDiets, 4 of ~40 annotated spans return -1 on exact search: some contain `<omitted>` where CUAD cut text out, others differ in whitespace. M2's gold-chunk lookup needs a fallback (e.g. match the text before `<omitted>`), not plain `str.find`. |
| **Qwen query prompt** | At T6, questions must be encoded with `prompt_name="query"`; chunks are encoded with no prompt. Forgetting it costs retrieval quality silently. |
| **`k` and `num_ctx`** | Decide at T6. Check the context length Ollama actually uses for `mistral:latest` (the model supports 32,768, the runtime default is smaller). |
| Chunk → contract mapping | The index needs a side record of which chunk came from which contract and where, or retrieval results can't be traced back. |
| Index location | `index/` and `*.faiss` are gitignored; the index is regenerated from the corpus, never committed. |
| Model files | `*.pkl` / `*.joblib` are gitignored. Before shipping, consider `git add -f` on the **final** classifier so the quickstart runs without retraining. |
| M6 reproducibility | Quickstart must document the CUAD download (the corpus is gitignored). |
