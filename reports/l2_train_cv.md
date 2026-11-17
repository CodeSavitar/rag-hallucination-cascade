# L2 on train, cross-validated (T27)

Logistic regression, 12 features, C = 0.187, Platt-calibrated. 217 train answers L1 left uncertain; labels = the judge's (79 fail). Every score below is out-of-fold: the answer was scored by a model that never saw it. This measures how well L2 imitates the judge; the test set (T28) measures it against people.

| System | ROC-AUC | PR-AUC | Flagged at 90% recall | Precision | Recall |
|---|---|---|---|---|---|
| L2 | 0.739 | 0.543 | 66% | 0.50 | 0.91 |
| Single signal (l1_len_ratio) | 0.713 | 0.536 | 71% | 0.46 | 0.91 |
| Chance | 0.500 | 0.364 | 90% | 0.36 | 0.90 |

The single signal's threshold was set on these same answers (in-sample), which flatters it slightly.

## Calibration (out-of-fold, 5 equal bins; Brier score 0.201)

| Answers | Predicted chance of hallucination | Actual share (judge) |
|---|---|---|
| 44 | 0.18 | 0.09 |
| 44 | 0.26 | 0.14 |
| 43 | 0.34 | 0.42 |
| 43 | 0.46 | 0.63 |
| 43 | 0.61 | 0.56 |

## Coefficients (per standard deviation; + = more likely a hallucination)

| Feature | Coefficient |
|---|---|
| l1_score_spread | -0.515 |
| mean_sentence_similarity | +0.444 |
| l1_answer_len_tokens | +0.441 |
| question_answer_similarity | +0.353 |
| n_numbers | -0.331 |
| n_sentences | +0.321 |
| l1_mean_similarity | +0.279 |
| l1_max_similarity | -0.217 |
| l1_token_overlap | -0.152 |
| is_paraphrased_refusal | -0.143 |
| l1_hedging_count | +0.053 |
| l1_min_sentence_similarity | +0.022 |
