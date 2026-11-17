import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from cascade import load_answers, route
from sweep import with_oof_scores

# T31: the accounting at the operating point picked in T30 (L2 threshold from data/operating_point.json).
#
#   python src/cascade_report.py  ->  reports/cascade_accounting.md
#                                     terminating_layer + cascade_verdict in query_log (all 320)
#
# For test (reference labels) and train (judge labels, L2 out-of-fold):
#   where answers stop, cost per 1,000, the evaluator's added latency (p50/p95) next to judging
#   everything, the confusion matrix, which layer let each miss through, and detection per
#   failure mode. Latency here is the EVALUATOR's only; the RAG answer itself (retrieval +
#   generation, ~7 s locally) comes first either way.

BASE_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = BASE_DIR / 'reports' / 'cascade_accounting.md'
MISS_LAYER = {'L1': 'L1 passed it (clean refusal)', 'L2': 'L2 cleared it (score below threshold)',
              'L3': 'the judge passed it'}


def table(frame):
    """A DataFrame as a Markdown table, index included."""
    header = [frame.index.name or '', *map(str, frame.columns)]
    body = [[str(index), *map(str, values)] for index, values in zip(frame.index, frame.values)]
    return ["| " + " | ".join(cells) + " |" for cells in [header, ['---'] * len(header), *body]]


def section(rows, routed, title):
    truth, flagged = rows['truth'], routed['flagged']
    judge_latency = rows['latency_l3_ms']
    layer_share = routed['terminating_layer'].value_counts(normalize=True).reindex(['L1', 'L2', 'L3'], fill_value=0)
    confusion = pd.crosstab(truth.map({1: 'hallucination', 0: 'fine'}).rename('truth'),
                            flagged.map({1: 'flagged', 0: 'let through'}).rename('cascade'))
    misses = routed[(truth == 1) & (flagged == 0)]['terminating_layer'].map(MISS_LAYER).value_counts()
    false_alarms = routed[(truth == 0) & (flagged == 1)]['terminating_layer'].value_counts()
    modes = pd.DataFrame({'answers': rows.groupby('induction_mode').size(),
                          'hallucinations': truth.groupby(rows['induction_mode']).sum(),
                          'caught': (truth & flagged).groupby(rows['induction_mode']).sum(),
                          'sent to judge': (routed['terminating_layer'] == 'L3').groupby(rows['induction_mode']).sum()})
    modes.index.name = 'failure mode'
    return [
        f"## {title}", "",
        f"**Where answers stop:** L1 {layer_share['L1']:.0%}, L2 {layer_share['L2']:.0%}, judge {layer_share['L3']:.0%}.", "",
        "| | Cascade | Judge everything |", "|---|---|---|",
        f"| Cost per 1,000 answers | ${routed['cost_usd'].mean() * 1000:.2f} | ${rows['cost_l3_usd'].mean() * 1000:.2f} |",
        f"| Evaluator latency, median | {routed['latency_ms'].median() / 1000:.2f} s | {judge_latency.median() / 1000:.2f} s |",
        f"| Evaluator latency, 95th percentile | {routed['latency_ms'].quantile(0.95) / 1000:.2f} s | {judge_latency.quantile(0.95) / 1000:.2f} s |",
        "",
        "**Confusion matrix**", "", *table(confusion), "",
        f"**Misses** ({int(((truth == 1) & (flagged == 0)).sum())} of {int(truth.sum())} hallucinations let through): "
        + (", ".join(f"{count} × {reason}" for reason, count in misses.items()) or "none") + ".", "",
        f"**False alarms** ({int(((truth == 0) & (flagged == 1)).sum())} fine answers flagged), by layer: "
        + (", ".join(f"{layer} {count}" for layer, count in false_alarms.items()) or "none") + ".", "",
        "**By failure mode**", "", *table(modes), "",
    ]


if __name__ == '__main__':
    point = json.loads((BASE_DIR / 'data' / 'operating_point.json').read_text())
    threshold = point['L2']['threshold']
    rows = with_oof_scores(load_answers())

    report = ["# Cascade accounting at the operating point (T31)", "",
              f"L1 → L2 (threshold {threshold:.3f}, picked on train in T30) → judge. L2 never blocks on its own. "
              "Judge cost and latency are the measured values for each answer.", ""]
    for split, title in [('test', 'Test: 40 answers, reference labels'),
                         ('train', 'Train: 280 answers, judge labels, L2 out-of-fold')]:
        part = rows[rows['split'] == split].reset_index(drop=True)
        report += section(part, route(part, 'l2_score', threshold), title)
    REPORT_MD.write_text("\n".join(report))
    print("\n".join(report))

    # The deployed model's routing, written for every answer (train rows use the saved model's
    # in-sample probability here; the report above uses out-of-fold scores instead).
    deployed = route(rows, 'l2_probability_calibrated', threshold)
    connection = sqlite3.connect(query_logger.DB_PATH)
    connection.executemany("UPDATE query_log SET terminating_layer = ?, cascade_verdict = ? WHERE query_id = ?",
                           deployed[['terminating_layer', 'cascade_verdict', 'query_id']].values.tolist())
    connection.commit()
    connection.close()
    print(f"terminating_layer and cascade_verdict written for {len(deployed)} answers")
