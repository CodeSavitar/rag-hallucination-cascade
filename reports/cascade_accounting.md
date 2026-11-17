# Cascade accounting at the operating point (T31)

L1 → L2 (threshold 0.293, picked on train in T30) → judge. L2 never blocks on its own. Judge cost and latency are the measured values for each answer.

## Test: 40 answers, reference labels

**Where answers stop:** L1 30%, L2 35%, judge 35%.

| | Cascade | Judge everything |
|---|---|---|
| Cost per 1,000 answers | $2.60 | $7.08 |
| Evaluator latency, median | 0.19 s | 2.65 s |
| Evaluator latency, 95th percentile | 3.50 s | 3.24 s |

**Confusion matrix**

| truth | flagged | let through |
| --- | --- | --- |
| fine | 5 | 23 |
| hallucination | 11 | 1 |

**Misses** (1 of 12 hallucinations let through): 1 × L2 cleared it (score below threshold).

**False alarms** (5 fine answers flagged), by layer: L1 3, L3 2.

**By failure mode**

| failure mode | answers | hallucinations | caught | sent to judge |
| --- | --- | --- | --- | --- |
| distractor | 7 | 1 | 1 | 1 |
| natural | 13 | 3 | 2 | 6 |
| starved_retrieval | 7 | 2 | 2 | 1 |
| unanswerable | 8 | 2 | 2 | 5 |
| weak_generator | 5 | 4 | 4 | 1 |

## Train: 280 answers, judge labels, L2 out-of-fold

**Where answers stop:** L1 22%, L2 31%, judge 46%.

| | Cascade | Judge everything |
|---|---|---|
| Cost per 1,000 answers | $3.41 | $7.13 |
| Evaluator latency, median | 0.24 s | 2.77 s |
| Evaluator latency, 95th percentile | 4.75 s | 5.82 s |

**Confusion matrix**

| truth | flagged | let through |
| --- | --- | --- |
| fine | 8 | 166 |
| hallucination | 96 | 10 |

**Misses** (10 of 106 hallucinations let through): 10 × L2 cleared it (score below threshold).

**False alarms** (8 fine answers flagged), by layer: L1 8.

**By failure mode**

| failure mode | answers | hallucinations | caught | sent to judge |
| --- | --- | --- | --- | --- |
| distractor | 46 | 24 | 23 | 26 |
| natural | 94 | 22 | 21 | 50 |
| starved_retrieval | 46 | 18 | 16 | 11 |
| unanswerable | 58 | 22 | 18 | 24 |
| weak_generator | 36 | 20 | 18 | 18 |
