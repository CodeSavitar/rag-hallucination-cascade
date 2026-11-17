import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import TARGET_RECALL, load_train
from cascade import load_answers, route
from l2_features import MODEL_FEATURES
from train_l2 import FOLDS, build_model

# T30: the cost-vs-recall curve (a curve, not a single cherry-picked number).
#
#   python src/sweep.py  ->  reports/cascade_curve.csv, reports/cascade_curve.svg, data/operating_point.json
#
# For each clearing score (L2's probability, and the length signal as the baseline), sweep the
# threshold over every value it takes. At each point: share of ALL answers sent to the judge,
# recall, precision, cost per 1,000 answers. Fixed reference points: judge everything, L1 only.
#
# The operating point is picked on TRAIN: the fewest judge calls that still catch >= 90% of the
# judge's fails. Then it is read, unchanged, on TEST against the reference labels.
# On train, L2's scores are OUT-OF-FOLD (each answer scored by a model that never saw it); its
# in-sample scores would make the cascade look better than it is.

BASE_DIR = Path(__file__).resolve().parent.parent
CURVE_CSV = BASE_DIR / 'reports' / 'cascade_curve.csv'
CURVE_SVG = BASE_DIR / 'reports' / 'cascade_curve.svg'
POINT_JSON = BASE_DIR / 'data' / 'operating_point.json'
SCORES = {'L2': 'l2_score', 'Length signal': 'l1_len_ratio'}
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 42


def with_oof_scores(rows):
    """'l2_score' = out-of-fold probability on train, the saved model's probability on test."""
    train = load_train()
    model = json.loads((BASE_DIR / 'data' / 'l2_model.json').read_text())
    oof = cross_val_predict(build_model(model['C']), train[MODEL_FEATURES].to_numpy(), train['judge_label'].to_numpy(),
                            cv=FOLDS, method='predict_proba')[:, 1]
    rows['l2_score'] = rows['query_id'].map(dict(zip(train['query_id'], oof)))
    is_test = rows['split'] == 'test'
    rows.loc[is_test, 'l2_score'] = rows.loc[is_test, 'l2_probability_calibrated']
    return rows


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


def pick_operating_point(train_curve):
    """Fewest judge calls with recall >= target; ties go to the higher threshold (fewer calls)."""
    good = train_curve[train_curve['recall'] >= TARGET_RECALL]
    return good.sort_values(['judge_share', 'threshold'], ascending=[True, False]).iloc[0]


def intervals(rows, score_column, threshold, rng):
    """95% bootstrap intervals for recall and judge share at a fixed threshold."""
    routed = route(rows, score_column, threshold)
    truth, flagged, judged = rows['truth'].to_numpy(), routed['flagged'].to_numpy(), (routed['terminating_layer'] == 'L3').to_numpy()
    recalls, shares = [], []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, len(rows), len(rows))
        if truth[picked].sum():
            recalls.append(flagged[picked][truth[picked] == 1].mean())
        shares.append(judged[picked].mean())
    return np.percentile(recalls, [2.5, 97.5]), np.percentile(shares, [2.5, 97.5])


def curve_svg(curves, fixed, path, title):
    """Recall (y) against share of answers sent to the judge (x), one line per clearing score."""
    size_x, size_y, pad = 480, 340, 50
    x = lambda value: pad + value * (size_x - 2 * pad)
    y = lambda value: size_y - pad - value * (size_y - 2 * pad)
    colours = {'L2': '#2b6cb0', 'Length signal': '#c05621'}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_x}" height="{size_y}" font-family="sans-serif">',
             f'<rect width="{size_x}" height="{size_y}" fill="white"/>',
             f'<text x="{size_x / 2}" y="20" font-size="13" text-anchor="middle">{title}</text>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(0)}" stroke="#333"/>',
             f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(0)}" y2="{y(1)}" stroke="#333"/>',
             f'<line x1="{x(0)}" y1="{y(TARGET_RECALL)}" x2="{x(1)}" y2="{y(TARGET_RECALL)}" stroke="#bbb" stroke-dasharray="4"/>',
             f'<text x="{x(1)}" y="{y(TARGET_RECALL) - 4}" font-size="10" text-anchor="end" fill="#777">90% recall</text>']
    for tick in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<text x="{x(tick)}" y="{y(0) + 16}" font-size="10" text-anchor="middle">{tick:.0%}</text>')
        parts.append(f'<text x="{x(0) - 6}" y="{y(tick) + 3}" font-size="10" text-anchor="end">{tick:.0%}</text>')
    for row_number, (name, points) in enumerate(curves.items()):
        ordered = points.sort_values('judge_share')
        line = " ".join(f"{x(s):.1f},{y(r):.1f}" for s, r in zip(ordered['judge_share'], ordered['recall']))
        parts.append(f'<polyline points="{line}" fill="none" stroke="{colours[name]}" stroke-width="2"/>')
        parts.append(f'<text x="{x(0.62)}" y="{y(0.32) + 16 * row_number}" font-size="11" fill="{colours[name]}">— {name}</text>')
    for name, point in fixed.items():
        parts.append(f'<circle cx="{x(point["judge_share"]):.1f}" cy="{y(point["recall"]):.1f}" r="5" fill="#2f855a"/>')
        parts.append(f'<text x="{x(point["judge_share"]) - 6:.1f}" y="{y(point["recall"]) + 16:.1f}" font-size="10" text-anchor="end" fill="#2f855a">{name}</text>')
    parts.append(f'<text x="{size_x / 2}" y="{size_y - 10}" font-size="11" text-anchor="middle">Share of answers sent to the judge (cost)</text>')
    parts.append(f'<text x="14" y="{size_y / 2}" font-size="11" text-anchor="middle" transform="rotate(-90 14 {size_y / 2})">Hallucinations caught (recall)</text>')
    path.write_text("".join(parts) + "</svg>")


if __name__ == '__main__':
    rows = with_oof_scores(load_answers())
    rng = np.random.default_rng(RANDOM_SEED)
    tables, chosen = [], {}
    for split in ['train', 'test']:
        part = rows[rows['split'] == split].reset_index(drop=True)
        curves = {}
        for name, column in SCORES.items():
            points = curve(part, column).assign(split=split, score=name)
            curves[name] = points
            tables.append(points)
        fixed = fixed_points(part)
        for name, point in fixed.items():
            tables.append(pd.DataFrame([{**point, 'split': split, 'score': name, 'threshold': np.nan}]))
        if split == 'train':
            chosen = {name: pick_operating_point(points) for name, points in curves.items()}
        curve_svg(curves, fixed, CURVE_SVG.with_name(f"cascade_curve_{split}.svg"),
                  f"Cost vs recall, {split} ({len(part)} answers)")
        if split == 'test':
            test_part, test_fixed = part, fixed

    pd.concat(tables).to_csv(CURVE_CSV, index=False)

    summary = {'target_recall': TARGET_RECALL, 'picked_on': 'train (L2 out-of-fold), judge labels'}
    print(f"Operating points (picked on train: fewest judge calls with recall >= {TARGET_RECALL:.0%})\n")
    print("| Clearing score | Threshold | Train: judge share / recall | Test: judge share | Test: recall | Test: precision | Test: $ per 1,000 |")
    print("|---|---|---|---|---|---|---|")
    for name, point in chosen.items():
        column = SCORES[name]
        test_numbers = measure(test_part, route(test_part, column, point['threshold']))
        (r_low, r_high), (s_low, s_high) = intervals(test_part, column, point['threshold'], rng)
        summary[name] = {'threshold': float(point['threshold']), 'train': {k: point[k] for k in ['judge_share', 'recall', 'precision', 'cost_per_1000']},
                         'test': {**test_numbers, 'recall_95': [float(r_low), float(r_high)], 'judge_share_95': [float(s_low), float(s_high)]}}
        print(f"| {name} | {point['threshold']:.4g} | {point['judge_share']:.0%} / {point['recall']:.0%} | "
              f"{test_numbers['judge_share']:.0%} ({s_low:.0%}–{s_high:.0%}) | {test_numbers['recall']:.0%} ({r_low:.0%}–{r_high:.0%}) | "
              f"{test_numbers['precision']:.0%} | ${test_numbers['cost_per_1000']:.2f} |")
    for name, point in test_fixed.items():
        summary[name] = {'test': point}
        print(f"| {name} | – | – | {point['judge_share']:.0%} | {point['recall']:.0%} | {point['precision']:.0%} | ${point['cost_per_1000']:.2f} |")
    POINT_JSON.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(f"\nWrote {CURVE_CSV.relative_to(BASE_DIR)}, reports/cascade_curve_train.svg, reports/cascade_curve_test.svg, {POINT_JSON.relative_to(BASE_DIR)}")
