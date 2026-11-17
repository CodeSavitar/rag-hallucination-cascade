import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger

# T29: the cascade router. For each answer: which layer settles it, its verdict, what it cost and
# how long it took.
#
#   L1 'fail'      -> settled at L1, fail   (hard evidence: refused-then-guessed, unknown number, ...)
#   L1 'pass'      -> settled at L1, pass   (a clean refusal makes no claim)
#   L1 'uncertain' -> L2's score:
#        score <  threshold -> settled at L2, pass  (L2 is confident it's fine)
#        score >= threshold -> escalated to L3      (the judge's verdict stands)
#
# L2 never blocks an answer by itself (design decision 2026-11-17, see STATUS.md M5).
# threshold = 0 sends every uncertain answer to the judge; above every score sends none.
#
# This works OFFLINE: every layer already ran on these answers (shadow mode), so routing is a
# lookup, and a threshold sweep costs nothing. Cost counts only the layers the cascade would
# actually have used (the T9 rule), not the shadow calls.
#
#   python src/cascade.py    marks the 320 seed + extra answers shadow_mode = 1, then self-tests

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / 'data' / 'l2_features.csv'
REFERENCE_CSV = BASE_DIR / 'data' / 'reference_labels.csv'
FREE_SCORES = {'l1_len_ratio'}    # scores L1 already computed: no extra time to use them


def load_answers():
    """All 320 seed + extra answers with every layer's output, cost and latency, and their labels.
    'truth' = the reference label on test (T24), the judge's label on train (what L2 learned)."""
    connection = sqlite3.connect(query_logger.DB_PATH)
    rows = pd.read_sql_query(
        "SELECT query_id, run_id, induction_mode, l1_verdict, l2_probability_calibrated, l3_verdict, "
        "cost_l1_usd, cost_l2_usd, cost_l3_usd, latency_l1_ms, latency_l2_ms, latency_l3_ms FROM query_log "
        "WHERE run_id IN (?, ?) AND error IS NULL", connection,
        params=(query_logger.SEED_RUN_ID, query_logger.EXTRA_RUN_ID))
    connection.close()
    rows = rows.merge(pd.read_csv(FEATURES_CSV)[['query_id', 'split', 'l1_len_ratio']], on='query_id')
    rows = rows.merge(pd.read_csv(REFERENCE_CSV)[['query_id', 'reference_label']], on='query_id', how='left')
    rows['judge_label'] = (rows['l3_verdict'] == 'fail').astype(int)
    rows['truth'] = rows['reference_label'].where(rows['split'] == 'test', rows['judge_label']).astype(int)
    return rows


def route(rows, score_column, threshold):
    """Where the cascade stops for each answer, and its verdict, cost and latency there."""
    l1 = rows['l1_verdict']
    score = rows[score_column]
    escalate = (l1 == 'uncertain') & (score >= threshold)
    layer = np.select([l1 != 'uncertain', escalate], ['L1', 'L3'], default='L2')
    verdict = np.select([l1 == 'fail', l1 == 'pass', escalate],
                        ['fail', 'pass', rows['l3_verdict']], default='pass')
    # The length signal comes free with L1; any other score is L2 proper, which adds its feature +
    # classifier time. (Fixed 2026-11-17: this used to check for one column name only, so the
    # out-of-fold 'l2_score' used by the sweep was timed as free.)
    l2_latency = 0.0 if score_column in FREE_SCORES else rows['latency_l2_ms'].fillna(0)
    reached_l2 = l1 == 'uncertain'
    return pd.DataFrame({
        'query_id': rows['query_id'],
        'terminating_layer': layer,
        'cascade_verdict': verdict,
        'flagged': (verdict == 'fail').astype(int),
        'cost_usd': (rows['cost_l1_usd'].fillna(0) + np.where(reached_l2, rows['cost_l2_usd'].fillna(0), 0)
                     + np.where(escalate, rows['cost_l3_usd'], 0)),
        'latency_ms': rows['latency_l1_ms'] + np.where(reached_l2, l2_latency, 0) + np.where(escalate, rows['latency_l3_ms'], 0),
    }, index=rows.index)


# --- Measuring a routing (used by sweep.py, cascade_report.py and the public app) ---------
def measure(rows, routed):
    truth, flagged = rows['truth'].to_numpy(), routed['flagged'].to_numpy()
    return {
        'judge_share': float((routed['terminating_layer'] == 'L3').mean()),
        'recall': float(flagged[truth == 1].mean()),
        'precision': float(truth[flagged == 1].mean()) if flagged.sum() else np.nan,
        'cost_per_1000': float(routed['cost_usd'].mean() * 1000),
        'latency_p50_ms': float(routed['latency_ms'].median()),
        'latency_p95_ms': float(routed['latency_ms'].quantile(0.95)),
    }


def curve(rows, score_column):
    """Every threshold the score takes on these rows, plus 'send all' (0) and 'send none' (inf)."""
    values = rows.loc[rows['l1_verdict'] == 'uncertain', score_column]
    points = []
    for threshold in [0.0, *sorted(values.unique()), np.inf]:
        points.append({'threshold': threshold, **measure(rows, route(rows, score_column, threshold))})
    return pd.DataFrame(points)


def fixed_points(rows):
    """Judge everything (every verdict is the judge's) and L1 only (uncertain = let through)."""
    judge = pd.DataFrame({'terminating_layer': 'L3', 'flagged': rows['judge_label'],
                          'cost_usd': rows['cost_l3_usd'], 'latency_ms': rows['latency_l3_ms']})
    l1 = route(rows, 'l1_len_ratio', np.inf)
    return {'Judge everything': measure(rows, judge), 'L1 only': measure(rows, l1)}


def self_test():
    """Hand-made answers whose routing is known in advance."""
    made_up = pd.DataFrame({
        'query_id': ['a', 'b', 'c', 'd', 'e'],
        'l1_verdict': ['fail', 'pass', 'uncertain', 'uncertain', 'uncertain'],
        'l2_probability_calibrated': [np.nan, np.nan, 0.10, 0.50, 0.90],
        'l3_verdict': ['pass', 'fail', 'fail', 'pass', 'fail'],    # the judge ran on all (shadow mode)
        'cost_l1_usd': 0.0, 'cost_l2_usd': [np.nan, np.nan, 0.0, 0.0, 0.0], 'cost_l3_usd': 0.007,
        'latency_l1_ms': 50.0, 'latency_l2_ms': [np.nan, np.nan, 100.0, 100.0, 100.0], 'latency_l3_ms': 3000.0,
    })
    routed = route(made_up, 'l2_probability_calibrated', 0.30)
    assert list(routed['terminating_layer']) == ['L1', 'L1', 'L2', 'L3', 'L3']
    assert list(routed['cascade_verdict']) == ['fail', 'pass', 'pass', 'pass', 'fail'], "L1 wins at L1; judge wins at L3"
    assert np.allclose(routed['cost_usd'], [0, 0, 0, 0.007, 0.007]), "the judge is paid only when escalated"
    assert np.allclose(routed['latency_ms'], [50, 50, 150, 3150, 3150])
    renamed = made_up.rename(columns={'l2_probability_calibrated': 'l2_score'})
    assert np.allclose(route(renamed, 'l2_score', 0.30)['latency_ms'], [50, 50, 150, 3150, 3150]), "any L2 score pays L2's time"
    free = made_up.assign(l1_len_ratio=[np.nan, np.nan, 0.01, 0.05, 0.09])
    assert np.allclose(route(free, 'l1_len_ratio', 0.03)['latency_ms'], [50, 50, 50, 3050, 3050]), "the length signal is free"
    everything = route(made_up, 'l2_probability_calibrated', 0.0)
    assert list(everything['terminating_layer']) == ['L1', 'L1', 'L3', 'L3', 'L3'], "threshold 0: every uncertain answer escalates"
    nothing = route(made_up, 'l2_probability_calibrated', 1.01)
    assert (nothing['terminating_layer'] != 'L3').all(), "threshold above every score: no judge calls"
    print("Self-test passed: 5 hand-made answers route, cost and time as expected")


if __name__ == '__main__':
    self_test()
    rows = load_answers()
    assert len(rows) == 320 and rows['l3_verdict'].notna().all(), "Every answer needs a judge verdict"
    assert rows.loc[rows['l1_verdict'] == 'uncertain', 'l2_probability_calibrated'].notna().all()
    # Every layer did run on these answers (L2 wherever it applies: L1 'uncertain'), so they are
    # shadow-mode rows; sanity_queries.py's cost query only counts rows marked this way.
    connection = sqlite3.connect(query_logger.DB_PATH)
    connection.executemany("UPDATE query_log SET shadow_mode = 1 WHERE query_id = ?", [(q,) for q in rows['query_id']])
    connection.commit()
    connection.close()
    print(f"Marked {len(rows)} answers shadow_mode = 1")
    print(rows.groupby('split')['truth'].agg(['count', 'sum']).rename(columns={'sum': 'hallucinations'}).to_string())
