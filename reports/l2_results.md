# L2 and the baselines on the test set (T28)

40 held-out TEST answers, scored once. Reference labels (T24): 12 of 40 not supported. L2 = `l2-v1`; its threshold (0.269) and the single signal's were fixed on train at 90% recall. Intervals are 95% bootstrap intervals. With 12 positives overall and 6 in the L2 slice, they are wide.

## Whole systems, all 40 answers

| System | Recall | Precision | Flagged | Judge cost per answer |
|---|---|---|---|---|
| judge_everything | 100% (100%–100%) | 80% (57%–100%) | 38% | $0.0071 |
| l1_only | 50% (22%–80%) | 67% (33%–100%) | 22% | $0.0000 |
| l1_plus_signal | 100% (100%–100%) | 40% (23%–57%) | 75% | $0.0000 |
| l1_plus_l2 | 100% (100%–100%) | 46% (27%–67%) | 65% | $0.0000 |

Recall = share of the reference's hallucinations flagged. judge_everything is partly circular: the reference uses the judge's label wherever Adhith agreed with it.

## L2's slice: the 28 answers L1 left uncertain (6 not supported)

| Score | ROC-AUC | PR-AUC | Flagged | Caught |
|---|---|---|---|---|
| L2 | 0.71 (0.53–0.89) | 0.34 (0.15–0.68) | 17 of 28 | 6 of 6 |
| Single signal (l1_len_ratio) | 0.80 (0.52–1.00) | 0.68 (0.23–1.00) | 21 of 28 | 6 of 6 |
| Chance | 0.50 | 0.21 | | |

L2 minus single signal, ROC-AUC: -0.08 (95% -0.33 to +0.19).

## Calibration on the slice (4 equal bins; chart: `l2_reliability_test.svg`)

| Answers | L2 predicted | Actual (reference) |
|---|---|---|
| 7 | 0.16 | 0.00 |
| 7 | 0.24 | 0.14 |
| 7 | 0.36 | 0.43 |
| 7 | 0.59 | 0.29 |

## Same systems against other references

Against Adhith's blind labels:

| System | Recall | Precision | Flagged | Judge cost per answer |
|---|---|---|---|---|
| judge_everything | 70% (40%–100%) | 47% (20%–75%) | 38% | $0.0071 |
| l1_only | 30% (0%–62%) | 33% (0%–67%) | 22% | $0.0000 |
| l1_plus_signal | 90% (67%–100%) | 30% (14%–47%) | 75% | $0.0000 |
| l1_plus_l2 | 70% (38%–100%) | 27% (11%–44%) | 65% | $0.0000 |

Against the judge's labels (what L2 was trained to imitate):

| System | Recall | Precision | Flagged | Judge cost per answer |
|---|---|---|---|---|
| judge_everything | 100% (100%–100%) | 100% (100%–100%) | 38% | $0.0071 |
| l1_only | 40% (14%–67%) | 67% (33%–100%) | 22% | $0.0000 |
| l1_plus_signal | 100% (100%–100%) | 50% (33%–68%) | 75% | $0.0000 |
| l1_plus_l2 | 93% (76%–100%) | 54% (35%–71%) | 65% | $0.0000 |

## Speed and cost

L2 per answer on this Mac (new features + classifier, model already loaded): median 142 ms, 95th percentile 231 ms. Judge: ~3 s and $0.0071 per answer. L1 and L2 cost $0.
