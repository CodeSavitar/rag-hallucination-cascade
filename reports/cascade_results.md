# Hallucination cascade: results (M5)

**Headline (test set, 40 held-out answers):** at the operating point picked on training data, the
cascade sends **35% of answers to the LLM judge** and catches **92% of hallucinations (11 of 12)**,
for **$2.60 per 1,000 answers instead of $7.08** to judge every answer (−63%). Median added time
per answer drops from 2.65 s to 0.19 s.

The test set is small. 95% intervals: share sent to the judge 20–50%, recall 71–100%. Read the
headline as "about a third of answers judged, about nine in ten hallucinations caught", not as
exact percentages.

## What was built

A RAG system answering questions about 30 CUAD contracts (local Mistral 7B, Qwen3 embeddings), and
three checkers in a row on every answer:

| Layer | What it does | Cost, time |
|---|---|---|
| L1 rules | Fails answers with hard evidence (refused then guessed, a number the excerpts never mention, very low overlap); passes clean refusals; the rest is uncertain | $0, ~0.07 s |
| L2 classifier | Logistic regression on 12 cheap signals, trained on 217 judge-labelled answers. Clears answers it is confident about; escalates the rest. **Never blocks an answer on its own** | $0, ~0.14 s |
| L3 judge | Claude Sonnet 5.5 checks every claim against the excerpts | ~$0.007, ~3 s |

Answers had failures induced on purpose (unanswerable questions, the right clause removed, a clause
from another contract planted, a weaker model), so roughly a third are hallucinations.

## The curve

`cascade_curve_test.svg` and `cascade_curve_train.svg` plot hallucinations caught against the
share of answers sent to the judge, for every threshold. Full numbers: `cascade_curve.csv`.

| System (test) | Sent to judge | Caught | Flags that are real | $ per 1,000 |
|---|---|---|---|---|
| Judge everything | 100% | 100%* | 80% | $7.08 |
| **Cascade, L2 clearing** | **35%** | **92%** | **69%** | **$2.60** |
| Cascade, length rule clearing | 45% | 92% | 65% | $3.27 |
| L1 only | 0% | 50% | 67% | $0 |

\* Partly by construction: the test reference uses the judge's label wherever the human labeller agreed with it.

**How the operating point was chosen:** on the 280 training answers, using L2 scores from models
that never saw each answer, the threshold with the fewest judge calls that still caught ≥ 90% of
the judge's fails. It was then applied unchanged to the test set, which was used once.

## Where it goes wrong (`cascade_accounting.md`)

- **Every missed hallucination was cleared by L2** (1 of 1 on test, 10 of 10 on train). L1 never
  passed one.
- **Most false alarms come from L1's fail rule** (3 of 5 on test, 8 of 8 on train). That rule was
  tuned on 80 answers and over-fires on new data.
- The 95th-percentile time is slightly worse than judging everything (3.5 s vs 3.2 s): answers that
  go to the judge pay for L1 and L2 first. The saving is in the typical answer and in cost.

## What each layer is worth

- **L1** is free and catches half the hallucinations on its own, but over-flags.
- **L2** cuts judge calls by about 10 points compared with a plain length rule at the same recall
  (train and test agree). It does not rank answers better than length (test ROC-AUC 0.71 vs 0.80,
  intervals overlapping), so its value is cost, not insight. See `l2_results.md`.
- **The judge** does the work cheap signals can't: answers that reuse the contract's wording while
  misreading it. Hallucinated answers were *more* similar to the excerpts than correct ones.

## Can the judge be trusted? (`judge_agreement.md`)

60 answers labelled blind by a person. Blind agreement with the judge: 67%, Cohen's κ 0.25, mostly
because the labeller applied the refusal rules inconsistently. The 19 disagreements were settled
blind by a model from another family (ChatGPT): it sided with the judge 13 times. After that,
agreement is 93% (κ 0.85) with ChatGPT's two rule slips corrected, and the judge missed no
hallucinations. That figure is an upper bound (the tie-breaker is an LLM too); the blind κ is the
independent number.

## Limits

- 40 test answers, 12 hallucinations: every test number has a wide interval.
- "Hallucination" means a claim the retrieved excerpts don't support. An answer faithful to a wrongly
  retrieved clause counts as supported; catching that is a retrieval problem, outside this evaluator.
- L2 is trained on the judge's labels, so it is a distillation of the judge, checked against people
  on the test set.
- Latency was measured on one MacBook; cost uses Claude Sonnet 5.5 list prices as billed.
- One corpus (CUAD), one generator family (Mistral). The method carries over; the trained weights
  and thresholds would need refitting.

## Reproduce

```
python src/cascade.py         # router self-test; marks shadow-mode rows
python src/sweep.py           # curve + operating point
python src/cascade_report.py  # accounting at the operating point
python src/export_app_data.py  # data for the app
streamlit run app/streamlit_app.py
```
