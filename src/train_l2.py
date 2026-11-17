import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, brier_score_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from baselines import TARGET_RECALL, choose_threshold, load_train
from l2_features import MODEL_FEATURES

# T27: train L2, a logistic regression on the 12 features, on the 217 TRAIN answers L1 left
# uncertain, with the judge's labels (decisions 2026-11-16: LR only, L1-uncertain only).
#
#   python src/train_l2.py    ->  models/l2.joblib (gitignored), data/l2_model.json, reports/l2_train_cv.md
#
# Steps:
#   1. Pick the regularisation strength C by 5-fold cross-validation (log loss).
#   2. Calibrate with Platt scaling (a logistic curve fitted on held-out folds), because the
#      cascade routes on "how sure is L2?", which only works if 0.7 means about 70%.
#   3. Score every train answer with a model that never saw it (out-of-fold), and set the
#      threshold on those scores at the same 90%-recall point as the baselines (T26).
# The test set is not read here.

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'l2.joblib'
MODEL_JSON = BASE_DIR / 'data' / 'l2_model.json'
REPORT_MD = BASE_DIR / 'reports' / 'l2_train_cv.md'
MODEL_VERSION = 'l2-v1'    # bump on any change to features, data or settings
RANDOM_SEED = 42
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)


def pick_c(X, y):
    """Regularisation strength by cross-validation: smaller C = simpler model, less overfitting."""
    search = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=np.logspace(-3, 2, 12), cv=FOLDS, scoring='neg_log_loss', max_iter=5000))
    search.fit(X, y)
    return float(search[-1].C_[0])


def build_model(c):
    """Scale -> logistic regression -> Platt calibration fitted on held-out folds."""
    base = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
    return CalibratedClassifierCV(base, method='sigmoid', cv=FOLDS)


def reliability_table(probabilities, labels, bins=5):
    """Predicted vs actual rate of hallucination, in equal-sized bins of answers."""
    order = np.argsort(probabilities)
    rows = []
    for chunk in np.array_split(order, bins):
        rows.append((len(chunk), probabilities[chunk].mean(), labels[chunk].mean()))
    return rows


if __name__ == '__main__':
    train = load_train()
    X, y = train[MODEL_FEATURES].to_numpy(), train['judge_label'].to_numpy()

    c = pick_c(X, y)
    model = build_model(c)

    # Out-of-fold: each answer scored by a model trained on the other 4/5ths.
    oof = cross_val_predict(build_model(c), X, y, cv=FOLDS, method='predict_proba')[:, 1]
    threshold = choose_threshold(oof, y, 'higher')
    flagged = (oof >= threshold).astype(int)

    # Same numbers for the single-signal baseline, for a side-by-side on train.
    settings = json.loads((BASE_DIR / 'data' / 'baselines.json').read_text())
    signal = train[settings['signal']].to_numpy()
    signal_flagged = (signal >= settings['threshold']).astype(int)

    model.fit(X, y)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    # Coefficients come from the same pipeline fitted on all 217 (before calibration, which only
    # rescales the score). Features are standardised, so sizes are comparable: per 1 std. dev.
    explain = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(X, y)
    coefficients = dict(sorted(zip(MODEL_FEATURES, explain[-1].coef_[0].round(3)), key=lambda item: -abs(item[1])))

    summary = {
        'version': MODEL_VERSION, 'features': MODEL_FEATURES, 'C': c,
        'threshold': float(threshold), 'target_recall': TARGET_RECALL,
        'train_rows': int(len(y)), 'train_positives': int(y.sum()),
        'cv': {'roc_auc': roc_auc_score(y, oof), 'pr_auc': average_precision_score(y, oof),
               'brier': brier_score_loss(y, oof), 'flagged_share': float(flagged.mean()),
               'precision': precision_score(y, flagged), 'recall': recall_score(y, flagged)},
        'coefficients_per_std': {name: float(value) for name, value in coefficients.items()},
    }
    MODEL_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    cv = summary['cv']
    lines = [
        "# L2 on train, cross-validated (T27)",
        "",
        f"Logistic regression, {len(MODEL_FEATURES)} features, C = {c:.3g}, Platt-calibrated. "
        f"{len(y)} train answers L1 left uncertain; labels = the judge's ({int(y.sum())} fail). "
        "Every score below is out-of-fold: the answer was scored by a model that never saw it. "
        "This measures how well L2 imitates the judge; the test set (T28) measures it against people.",
        "",
        "| System | ROC-AUC | PR-AUC | Flagged at 90% recall | Precision | Recall |",
        "|---|---|---|---|---|---|",
        f"| L2 | {cv['roc_auc']:.3f} | {cv['pr_auc']:.3f} | {cv['flagged_share']:.0%} | {cv['precision']:.2f} | {cv['recall']:.2f} |",
        f"| Single signal ({settings['signal']}) | {roc_auc_score(y, signal):.3f} | {average_precision_score(y, signal):.3f} | "
        f"{signal_flagged.mean():.0%} | {precision_score(y, signal_flagged):.2f} | {recall_score(y, signal_flagged):.2f} |",
        f"| Chance | 0.500 | {y.mean():.3f} | 90% | {y.mean():.2f} | 0.90 |",
        "",
        "The single signal's threshold was set on these same answers (in-sample), which flatters it slightly.",
        "",
        f"## Calibration (out-of-fold, 5 equal bins; Brier score {cv['brier']:.3f})",
        "",
        "| Answers | Predicted chance of hallucination | Actual share (judge) |",
        "|---|---|---|",
        *[f"| {n} | {predicted:.2f} | {actual:.2f} |" for n, predicted, actual in reliability_table(oof, y)],
        "",
        "## Coefficients (per standard deviation; + = more likely a hallucination)",
        "",
        "| Feature | Coefficient |",
        "|---|---|",
        *[f"| {name} | {value:+.3f} |" for name, value in coefficients.items()],
        "",
    ]
    REPORT_MD.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved {MODEL_PATH.relative_to(BASE_DIR)}, {MODEL_JSON.relative_to(BASE_DIR)}, {REPORT_MD.relative_to(BASE_DIR)}")
