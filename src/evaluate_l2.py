import json
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from baselines import flag
from l1_signals import resources
from l2_features import MODEL_FEATURES, new_features

# T28: the test set, used once. Every system is scored together on the 40 TEST answers against
# the reference labels (T24), so no number below could have shaped how L2 or a baseline was built.
#
#   python src/evaluate_l2.py
#     -> reports/l2_results.md, reports/l2_reliability_test.svg
#     -> l2_* columns in query_log for every answer L1 left uncertain (NULL elsewhere: L2 didn't run)
#
# Two views:
#   SYSTEM (all 40): what each whole system flags. judge_everything / l1_only / l1_plus_signal /
#                    l1_plus_l2. Recall, precision, share flagged, cost per answer.
#   L2 SLICE (the 28 L1 left uncertain): L2's score vs the single signal's, threshold-free
#                    (ROC-AUC, PR-AUC) and at the thresholds fixed on train.
# Plus the same system table against Adhith's blind labels and the judge's, to show how much the
# choice of reference matters.

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / 'data' / 'l2_features.csv'
REFERENCE_CSV = BASE_DIR / 'data' / 'reference_labels.csv'
REPORT_MD = BASE_DIR / 'reports' / 'l2_results.md'
RELIABILITY_SVG = BASE_DIR / 'reports' / 'l2_reliability_test.svg'
MODEL_PATH = BASE_DIR / 'models' / 'l2.joblib'
JUDGE_COST_PER_ANSWER = 0.0071    # measured: $1.43 for 200 answers in T23 (cost_l3_usd averages agree)
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 42


def interval(values, statistic, rng):
    """95% bootstrap interval of statistic(resampled rows)."""
    results = []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, len(values), len(values))
        result = statistic(values[picked])
        if result is not None:
            results.append(result)
    return np.percentile(results, [2.5, 97.5])


def recall(rows):
    positives = rows[rows[:, 0] == 1]
    return None if len(positives) == 0 else positives[:, 1].mean()


def precision(rows):
    flagged = rows[rows[:, 1] == 1]
    return None if len(flagged) == 0 else flagged[:, 0].mean()


def system_table(rows, label_column, settings, rng):
    """One line per system: recall, precision, share flagged (each with a 95% interval), cost."""
    lines = ["| System | Recall | Precision | Flagged | Judge cost per answer |", "|---|---|---|---|---|"]
    for system in ['judge_everything', 'l1_only', 'l1_plus_signal', 'l1_plus_l2']:
        if system == 'l1_plus_l2':    # L1 decides its cases; L2 decides the rest
            flagged = rows['l1_flag'].where(rows['l1_verdict'] != 'uncertain', rows['l2_flag'])
        else:
            flagged = flag(rows, system, settings)
        pairs = np.column_stack([rows[label_column].to_numpy(), flagged.to_numpy()])
        r_low, r_high = interval(pairs, recall, rng)
        p_low, p_high = interval(pairs, precision, rng)
        share = flagged.mean()
        cost = JUDGE_COST_PER_ANSWER if system == 'judge_everything' else 0.0
        lines.append(f"| {system} | {recall(pairs):.0%} ({r_low:.0%}–{r_high:.0%}) | "
                     f"{precision(pairs):.0%} ({p_low:.0%}–{p_high:.0%}) | {share:.0%} | ${cost:.4f} |")
    return lines


def reliability_svg(probabilities, labels, path, bins=4):
    """A small calibration chart: predicted vs actual, one dot per bin, diagonal = perfect."""
    order = np.argsort(probabilities)
    points = [(probabilities[chunk].mean(), labels[chunk].mean(), len(chunk)) for chunk in np.array_split(order, bins)]
    size, pad = 320, 40
    scale = lambda value: pad + value * (size - 2 * pad)
    flip = lambda value: size - scale(value)
    dots = "".join(f'<circle cx="{scale(p):.1f}" cy="{flip(a):.1f}" r="5" fill="#2b6cb0"/>'
                   f'<text x="{scale(p) + 8:.1f}" y="{flip(a) - 6:.1f}" font-size="10" fill="#444">n={n}</text>'
                   for p, a, n in points)
    line = " ".join(f"{scale(p):.1f},{flip(a):.1f}" for p, a, _ in points)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" font-family="sans-serif">'
        f'<rect width="{size}" height="{size}" fill="white"/>'
        f'<line x1="{scale(0)}" y1="{flip(0)}" x2="{scale(1)}" y2="{flip(1)}" stroke="#bbb" stroke-dasharray="4"/>'
        f'<line x1="{scale(0)}" y1="{flip(0)}" x2="{scale(1)}" y2="{flip(0)}" stroke="#333"/>'
        f'<line x1="{scale(0)}" y1="{flip(0)}" x2="{scale(0)}" y2="{flip(1)}" stroke="#333"/>'
        f'<polyline points="{line}" fill="none" stroke="#2b6cb0"/>{dots}'
        f'<text x="{size / 2}" y="{size - 8}" font-size="11" text-anchor="middle">L2 predicted chance of hallucination</text>'
        f'<text x="12" y="{size / 2}" font-size="11" text-anchor="middle" transform="rotate(-90 12 {size / 2})">Actual share (reference)</text>'
        f'<text x="{size / 2}" y="18" font-size="12" text-anchor="middle">L2 calibration, 28 test answers</text></svg>')


if __name__ == '__main__':
    model = joblib.load(MODEL_PATH)
    l2 = json.loads((BASE_DIR / 'data' / 'l2_model.json').read_text())
    settings = json.loads((BASE_DIR / 'data' / 'baselines.json').read_text())

    features = pd.read_csv(FEATURES_CSV)
    connection = sqlite3.connect(query_logger.DB_PATH)
    logged = pd.read_sql_query("SELECT query_id, query_text, answer_text, retrieved_chunk_ids, l1_features_json, l3_verdict "
                               "FROM query_log", connection)
    rows = features.merge(logged, on='query_id')

    # --- L2 on every answer L1 left uncertain, timed like it would run live -------------------
    resources()    # embedding model loaded once, outside the timer
    uncertain = rows['l1_verdict'] == 'uncertain'
    updates = []
    for row in rows[uncertain].itertuples():
        start = time.perf_counter()
        behaviour = json.loads(row.l1_features_json)['answer_behaviour']
        new_features(row.query_text, row.answer_text, json.loads(row.retrieved_chunk_ids), behaviour)    # the work L2 adds
        x = np.array([[getattr(row, name) for name in MODEL_FEATURES]])
        calibrated = model.predict_proba(x)[0, 1]
        latency = (time.perf_counter() - start) * 1000
        raw = np.mean([fold.estimator.predict_proba(x)[0, 1] for fold in model.calibrated_classifiers_])
        updates.append({'query_id': row.query_id, 'l2_probability_raw': float(raw),
                        'l2_probability_calibrated': float(calibrated),
                        'l2_verdict': 'fail' if calibrated >= l2['threshold'] else 'pass',
                        'l2_model_version': l2['version'], 'latency_l2_ms': latency, 'cost_l2_usd': 0.0})
    connection.executemany(
        "UPDATE query_log SET l2_probability_raw = :l2_probability_raw, l2_probability_calibrated = :l2_probability_calibrated, "
        "l2_verdict = :l2_verdict, l2_model_version = :l2_model_version, latency_l2_ms = :latency_l2_ms, "
        "cost_l2_usd = :cost_l2_usd WHERE query_id = :query_id", updates)
    connection.commit()
    connection.close()
    scored = pd.DataFrame(updates)
    rows = rows.merge(scored[['query_id', 'l2_probability_calibrated', 'l2_verdict', 'latency_l2_ms']], on='query_id', how='left')

    # --- The test set ---------------------------------------------------------------------------
    test = rows[rows['split'] == 'test'].merge(pd.read_csv(REFERENCE_CSV)[['query_id', 'reference_label', 'human_label', 'judge_label']], on='query_id')
    assert len(test) == 40
    test['l1_flag'] = (test['l1_verdict'] == 'fail').astype(int)
    test['l2_flag'] = (test['l2_verdict'] == 'fail').astype(int)
    rng = np.random.default_rng(RANDOM_SEED)

    slice_ = test[test['l1_verdict'] == 'uncertain']
    y = slice_['reference_label'].to_numpy()
    l2_score, signal = slice_['l2_probability_calibrated'].to_numpy(), slice_[settings['signal']].to_numpy()
    slice_lines = ["| Score | ROC-AUC | PR-AUC | Flagged | Caught |", "|---|---|---|---|---|"]
    for name, score, flagged in [('L2', l2_score, slice_['l2_flag'].to_numpy()),
                                 (f"Single signal ({settings['signal']})", signal, (signal >= settings['threshold']).astype(int))]:
        both = np.column_stack([y, score])
        roc = interval(both, lambda r: roc_auc_score(r[:, 0], r[:, 1]) if 0 < r[:, 0].sum() < len(r) else None, rng)
        pr = interval(both, lambda r: average_precision_score(r[:, 0], r[:, 1]) if r[:, 0].sum() > 0 else None, rng)
        slice_lines.append(f"| {name} | {roc_auc_score(y, score):.2f} ({roc[0]:.2f}–{roc[1]:.2f}) | "
                           f"{average_precision_score(y, score):.2f} ({pr[0]:.2f}–{pr[1]:.2f}) | "
                           f"{flagged.sum()} of {len(y)} | {int((flagged & y).sum())} of {int(y.sum())} |")
    slice_lines.append(f"| Chance | 0.50 | {y.mean():.2f} | | |")
    paired = np.column_stack([y, l2_score, signal])
    gap = interval(paired, lambda r: roc_auc_score(r[:, 0], r[:, 1]) - roc_auc_score(r[:, 0], r[:, 2])
                   if 0 < r[:, 0].sum() < len(r) else None, rng)

    reliability_svg(l2_score, y, RELIABILITY_SVG)
    order = np.argsort(l2_score)
    calibration_lines = ["| Answers | L2 predicted | Actual (reference) |", "|---|---|---|",
                         *[f"| {len(chunk)} | {l2_score[chunk].mean():.2f} | {y[chunk].mean():.2f} |"
                           for chunk in np.array_split(order, 4)]]

    latencies = rows['latency_l2_ms'].dropna()
    report = [
        "# L2 and the baselines on the test set (T28)",
        "",
        f"40 held-out TEST answers, scored once. Reference labels (T24): {int(test['reference_label'].sum())} of 40 not supported. "
        f"L2 = `{l2['version']}`; its threshold ({l2['threshold']:.3f}) and the single signal's were fixed on train at 90% recall. "
        "Intervals are 95% bootstrap intervals. With 12 positives overall and 6 in the L2 slice, they are wide.",
        "",
        "## Whole systems, all 40 answers",
        "",
        *system_table(test, 'reference_label', settings, rng),
        "",
        "Recall = share of the reference's hallucinations flagged. judge_everything is partly circular: "
        "the reference uses the judge's label wherever Adhith agreed with it.",
        "",
        f"## L2's slice: the {len(slice_)} answers L1 left uncertain ({int(y.sum())} not supported)",
        "",
        *slice_lines,
        "",
        f"L2 minus single signal, ROC-AUC: {roc_auc_score(y, l2_score) - roc_auc_score(y, signal):+.2f} "
        f"(95% {gap[0]:+.2f} to {gap[1]:+.2f}).",
        "",
        "## Calibration on the slice (4 equal bins; chart: `l2_reliability_test.svg`)",
        "",
        *calibration_lines,
        "",
        "## Same systems against other references",
        "",
        "Against Adhith's blind labels:",
        "",
        *system_table(test, 'human_label', settings, rng),
        "",
        "Against the judge's labels (what L2 was trained to imitate):",
        "",
        *system_table(test, 'judge_label', settings, rng),
        "",
        "## Speed and cost",
        "",
        f"L2 per answer on this Mac (new features + classifier, model already loaded): median {latencies.median():.0f} ms, "
        f"95th percentile {latencies.quantile(0.95):.0f} ms. Judge: ~3 s and ${JUDGE_COST_PER_ANSWER} per answer. L1 and L2 cost $0.",
        "",
    ]
    REPORT_MD.write_text("\n".join(report))
    print("\n".join(report))
    print(f"l2_* columns written for {len(updates)} answers (the ones L1 left uncertain)")
