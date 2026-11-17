import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from l2_features import FEATURES

# T26: the three baselines L2 has to beat. Every system answers one question per
# answer: flag it as a hallucination (1) or let it through (0).
#
#   judge_everything  every answer goes to the judge          quality ceiling, cost ceiling ($0.007, ~3 s each)
#   l1_only           L1 decides; "uncertain" is let through  the free floor
#   l1_plus_signal    L1 decides its cases; the rest are flagged by ONE signal and a threshold:
#                     "did the ML earn its place?"
#
# The single signal and its threshold are chosen HERE, on TRAIN only, against the judge's labels
# (the same labels L2 trains on). Test results for every system are computed together in T28, so
# no baseline's test score can shape how L2 is built.
#
#   python src/baselines.py    ->  data/baselines.json (the chosen signal, direction, threshold)

BASE_DIR = Path(__file__).resolve().parent.parent
FEATURES_CSV = BASE_DIR / 'data' / 'l2_features.csv'
OUT_JSON = BASE_DIR / 'data' / 'baselines.json'


def load_train():
    """TRAIN answers L1 left uncertain (what L2 and the single signal both have to judge), with judge labels."""
    features = pd.read_csv(FEATURES_CSV)
    connection = sqlite3.connect(query_logger.DB_PATH)
    verdicts = pd.read_sql_query("SELECT query_id, l3_verdict FROM query_log", connection)
    connection.close()
    rows = features.merge(verdicts, on='query_id')
    rows = rows[(rows['split'] == 'train') & (rows['l1_verdict'] == 'uncertain')].copy()
    rows['judge_label'] = (rows['l3_verdict'] == 'fail').astype(int)
    return rows


def choose_signal(rows):
    """The single feature that best separates judge-failed answers (by AUC), from all 14.
    Picked by rule, not by hand: whichever scores highest, even if it isn't the expected one."""
    scores = {}
    for name in FEATURES:
        auc = roc_auc_score(rows['judge_label'], rows[name])
        scores[name] = (max(auc, 1 - auc), 'higher' if auc >= 0.5 else 'lower')
    best = max(scores, key=lambda name: scores[name][0])
    return best, scores[best][1], scores[best][0], scores


TARGET_RECALL = 0.90    # the operating point every system is compared at (L2 too, in T27-T28)


def choose_threshold(values, labels, direction, target_recall=TARGET_RECALL):
    """The strictest cut-off that still catches TARGET_RECALL of the judge's fails on train.

    Why not best F1: on train it flagged 76% of answers, i.e. "send nearly everything to the
    judge", which says nothing about a cascade. The question that matters is "to catch 90% of
    hallucinations, how few answers must be flagged?", so that's the fixed point. L2 uses it too.
    """
    labels = np.asarray(labels)
    candidates = np.unique(values)
    # Strictest first: for "higher = bad", that's the largest threshold.
    for threshold in (candidates[::-1] if direction == 'higher' else candidates):
        flagged = values >= threshold if direction == 'higher' else values <= threshold
        if recall_score(labels, flagged.astype(int)) >= target_recall:
            return float(threshold)
    raise ValueError("No threshold reaches the target recall")


def flag(rows, system, settings):
    """1 = flagged as a hallucination, for each row. `rows` needs l1_verdict, l3_verdict and the features."""
    if system == 'judge_everything':
        return (rows['l3_verdict'] == 'fail').astype(int)
    l1_flag = (rows['l1_verdict'] == 'fail').astype(int)
    if system == 'l1_only':
        return l1_flag
    if system == 'l1_plus_signal':
        values = rows[settings['signal']]
        over = values >= settings['threshold'] if settings['direction'] == 'higher' else values <= settings['threshold']
        return l1_flag.where(rows['l1_verdict'] != 'uncertain', over.astype(int))
    raise ValueError(f"Unknown system: {system}")


if __name__ == '__main__':
    train = load_train()
    signal, direction, auc, all_scores = choose_signal(train)
    threshold = choose_threshold(train[signal].to_numpy(), train['judge_label'], direction)
    settings = {'signal': signal, 'direction': direction, 'threshold': threshold,
                'target_recall': TARGET_RECALL, 'train_auc': round(auc, 3), 'train_rows': len(train), 'chosen_on': 'train, judge labels, L1-uncertain only'}
    OUT_JSON.write_text(json.dumps(settings, indent=2) + "\n")

    flagged = flag(train, 'l1_plus_signal', settings)
    print(f"Chosen on {len(train)} train answers ({train['judge_label'].sum()} judge fails):")
    print(f"  signal: {signal}  ({direction} = more likely a hallucination), AUC {auc:.3f}")
    print(f"  threshold: {threshold:.4g}  -> flags {flagged.sum()} of {len(train)}")
    print(f"  train: precision {precision_score(train['judge_label'], flagged):.2f}, "
          f"recall {recall_score(train['judge_label'], flagged):.2f} (target {TARGET_RECALL:.0%})")
    print("  runner-up signals:", ", ".join(f"{name} {score:.3f}" for name, (score, _) in
                                           sorted(all_scores.items(), key=lambda item: -item[1][0])[1:4]))
    print(f"Wrote {OUT_JSON.relative_to(BASE_DIR)}")
