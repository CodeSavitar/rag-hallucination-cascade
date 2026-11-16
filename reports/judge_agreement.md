# Judge vs human agreement (T22)

Judge: Claude Sonnet 5.5, prompt `l3-v1`. Human: Adhith, blind, rubric = supported by the excerpts.
1 = not supported (hallucination). Intervals are 95% bootstrap intervals (2000 resamples).

| Set | n | Raw agreement | Cohen's κ | Judge missed | Judge extra |
|---|---|---|---|---|---|
| Headline (no calibration rows) | 57 | 67% (54%–79%) | 0.25 (-0.00–0.49) | 6 | 13 |
| All labelled | 60 | 67% (53%–78%) | 0.25 (0.00–0.49) | 6 | 14 |
| Headline without "Not sure" | 56 | 68% (55%–80%) | 0.28 (0.01–0.54) | 5 | 13 |

*Judge missed* = human said not supported, judge passed it (the costly error for a detector). *Judge extra* = human said supported, judge failed it.

Headline: human marked 15 of 57 not supported; the judge failed 22; both agreed on 9.

## By train / test split (headline rows)

| split | n | agree | human not supported | judge fail | agree % |
| --- | --- | --- | --- | --- | --- |
| test | 37 | 27 | 10 | 14 | 73 |
| train | 20 | 11 | 5 | 8 | 55 |

## By failure mode (headline rows; groups are small, read as direction only)

| induction_mode | n | agree | human not supported | judge fail | agree % |
| --- | --- | --- | --- | --- | --- |
| distractor | 10 | 6 | 3 | 3 | 60 |
| natural | 18 | 13 | 3 | 6 | 72 |
| starved_retrieval | 9 | 5 | 3 | 3 | 56 |
| unanswerable | 12 | 9 | 3 | 4 | 75 |
| weak_generator | 8 | 5 | 3 | 6 | 62 |

## L1 on the answers it decided

L1 gave pass/fail on 15 of the 57 headline answers and matched the human label on 6.

## After review (T22b): disagreements settled by blind tie-breaker models

Tie-breakers: chatgpt, shown the rubric, question, excerpts and answer only. They sided with the judge on 13 of 19 disagreements. In 2 cases a model called a plain refusal not supported, against the written rule; the second row corrects those.

**This is an upper bound, not an independent result:** the tie-breakers are LLMs and can share the judge's blind spots. The blind headline above stays the independent figure.

| Set | n | Raw agreement | Cohen's κ | Judge missed | Judge extra |
|---|---|---|---|---|---|
| After review: tie-breaker votes as given | 57 | 89% (81%–96%) | 0.77 (0.59–0.93) | 2 | 4 |
| After review: votes corrected to the refusal rule | 57 | 93% (86%–98%) | 0.85 (0.69–0.96) | 0 | 4 |
