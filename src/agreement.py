import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# T22: how often does the judge (L3) agree with Adhith's blind labels?
#
#   python src/agreement.py
#
# Reports raw agreement and Cohen's kappa, each with a bootstrap 95% confidence interval, on:
#   - HEADLINE: the 60 labelled answers minus the 3 calibration rows reviewed outside the blind process (57)
#   - all 60, and the headline without the "Not sure" rows, as secondary numbers
# Writes judge_agreed back to query_log, a summary to reports/judge_agreement.md, every
# disagreement in full to reports/judge_disagreements.md for review, and a backup of the human
# labels to data/human_labels.csv (the database itself is gitignored).
#
# Both sides use the same coding: 1 = not supported (hallucination), 0 = supported.

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'data' / 'query_log.db'
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
SAMPLE_CSV = BASE_DIR / 'data' / 'label_sample.csv'
REPORT_MD = BASE_DIR / 'reports' / 'judge_agreement.md'
DISAGREEMENTS_MD = BASE_DIR / 'reports' / 'judge_disagreements.md'
LABELS_CSV = BASE_DIR / 'data' / 'human_labels.csv'
# T22b: blind votes from other-family models on the disagreements (one row per case and model).
TIEBREAK_CSV = BASE_DIR / 'data' / 'tiebreak_votes.csv'
# Same list as in pick_label_sample.py: rows reviewed outside the blind process on 2026-11-14, so not independent.
CALIBRATION_PREFIXES = ('88c27239', 'e1e43e17', '42afd15e', 'b510a476', '08aec7af')
BOOTSTRAP_ROUNDS = 2000
RANDOM_SEED = 42


def load_labelled():
    """Every answer in the labelling sample, with the human label, the judge verdict and context."""
    connection = sqlite3.connect(DB_PATH)
    rows = pd.read_sql_query(
        "SELECT query_id, question_id, query_text, answer_text, retrieved_text, human_label, "
        "human_label_notes, l3_verdict, l3_unsupported_span, l3_rationale, l1_verdict "
        "FROM query_log WHERE human_label IS NOT NULL", connection)
    connection.close()
    rows = rows[rows['query_id'].isin(pd.read_csv(SAMPLE_CSV)['query_id'])]
    rows = rows.merge(pd.read_csv(SEED_CSV, usecols=['question_id', 'split', 'induction_mode']), on='question_id')
    notes = rows['human_label_notes'].map(lambda text: json.loads(text) if text else {})
    rows['not_sure'] = notes.map(lambda n: n.get('not_sure', False))
    rows['unhelpful'] = notes.map(lambda n: n.get('unhelpful', False))
    rows['human_comment'] = notes.map(lambda n: n.get('comment', ''))
    rows['calibration'] = rows['query_id'].str.startswith(CALIBRATION_PREFIXES)
    rows['judge_label'] = rows['l3_verdict'].map({'fail': 1, 'pass': 0})    # 'uncertain' -> NaN
    return rows.reset_index(drop=True)


def cohen_kappa(human, judge):
    """Agreement corrected for chance: 1 = perfect, 0 = no better than guessing from the base rates."""
    observed = np.mean(human == judge)
    expected = np.mean(human) * np.mean(judge) + (1 - np.mean(human)) * (1 - np.mean(judge))
    return np.nan if expected == 1 else (observed - expected) / (1 - expected)


def with_interval(human, judge, statistic):
    """The statistic, plus a 95% bootstrap interval: resample the answers with replacement and
    see how much the number moves. Small samples move a lot, which is the point of reporting it."""
    rng = np.random.default_rng(RANDOM_SEED)
    rounds = []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = rng.integers(0, len(human), len(human))
        rounds.append(statistic(human[picked], judge[picked]))
    low, high = np.nanpercentile(rounds, [2.5, 97.5])
    return statistic(human, judge), low, high


def summarise(rows, name):
    human, judge = rows['human_label'].to_numpy(), rows['judge_label'].to_numpy()
    agreement = with_interval(human, judge, lambda h, j: np.mean(h == j))
    kappa = with_interval(human, judge, cohen_kappa)
    caught = int(((human == 1) & (judge == 1)).sum())
    return {
        'set': name, 'n': len(rows),
        'agreement': agreement, 'kappa': kappa,
        'human_not_supported': int(human.sum()), 'judge_fail': int(judge.sum()),
        'both_fail': caught,
        'judge_missed': int(((human == 1) & (judge == 0)).sum()),         # human: not supported; judge: pass
        'judge_extra': int(((human == 0) & (judge == 1)).sum()),          # human: supported; judge: fail
    }


def by_group(rows, column):
    table = rows.groupby(column).apply(lambda g: pd.Series({
        'n': len(g),
        'agree': int((g['human_label'] == g['judge_label']).sum()),
        'human not supported': int(g['human_label'].sum()),
        'judge fail': int(g['judge_label'].sum()),
    }), include_groups=False)
    table['agree %'] = (100 * table['agree'] / table['n']).round(0).astype(int)
    return table


def markdown_table(table):
    """A DataFrame as a Markdown table (pandas' own to_markdown needs an extra package)."""
    header = [table.index.name or '', *table.columns]
    body = [[str(index), *map(str, values)] for index, values in zip(table.index, table.values)]
    return "\n".join("| " + " | ".join(cells) + " |" for cells in [header, ['---'] * len(header), *body])


def line(result):
    a, k = result['agreement'], result['kappa']
    return (f"| {result['set']} | {result['n']} | {a[0]:.0%} ({a[1]:.0%}–{a[2]:.0%}) | "
            f"{k[0]:.2f} ({k[1]:.2f}–{k[2]:.2f}) | {result['judge_missed']} | {result['judge_extra']} |")


rows = load_labelled()
if rows['judge_label'].isna().any():
    raise SystemExit(f"{rows['judge_label'].isna().sum()} labelled answers have no pass/fail verdict from the judge")
rows['judge_agreed'] = (rows['human_label'] == rows['judge_label']).astype(int)

headline_rows = rows[~rows['calibration']]
results = [summarise(headline_rows, 'Headline (no calibration rows)'),
           summarise(rows, 'All labelled'),
           summarise(headline_rows[~headline_rows['not_sure']], 'Headline without "Not sure"')]
headline = results[0]

# L1 is checked too, but only where it gave a verdict: it says "uncertain" on most answers.
l1_rows = headline_rows[headline_rows['l1_verdict'].isin(['pass', 'fail'])]
l1_agree = int((l1_rows['l1_verdict'].map({'fail': 1, 'pass': 0}) == l1_rows['human_label']).sum())

# --- T22b: settle the disagreements with the tie-breaker votes ----------------------------
# Where human and judge agree, that label stands. Where they disagree, the tie-breaker models'
# majority decides (one model: its vote). Two versions: votes as given, and votes corrected where
# the model broke the written rule "a plain refusal is supported". An UPPER BOUND on the judge's
# agreement: the tie-breakers are LLMs too and can share its blind spots.
after_review = []
if TIEBREAK_CSV.exists():
    votes = pd.read_csv(TIEBREAK_CSV)
    votes['rule_vote'] = votes['vote'].where(votes['breaks_refusal_rule'] == 0, 0)
    majority = votes.groupby('query_id')[['vote', 'rule_vote']].mean().round()    # a tie rounds to 0
    disputed = headline_rows[headline_rows['judge_agreed'] == 0]
    missing = set(disputed['query_id']) - set(majority.index)
    if missing:
        raise SystemExit(f"{len(missing)} disagreements have no tie-breaker vote")
    sided_with_judge = int((majority.loc[disputed['query_id'], 'vote'].to_numpy()
                            == disputed['judge_label'].to_numpy()).sum())
    for column, name in [('vote', 'After review: tie-breaker votes as given'),
                         ('rule_vote', 'After review: votes corrected to the refusal rule')]:
        reviewed = headline_rows.copy()
        settled = reviewed['query_id'].map(majority[column])
        reviewed['human_label'] = settled.fillna(reviewed['human_label']).astype(int)
        after_review.append(summarise(reviewed, name))
    tiebreak_models = ", ".join(sorted(votes['model'].unique()))
    rule_breaks = int(votes['breaks_refusal_rule'].sum())

# --- Write judge_agreed back, and back up the labels ----------------------------------------
connection = sqlite3.connect(DB_PATH)
connection.executemany("UPDATE query_log SET judge_agreed = ? WHERE query_id = ?",
                       rows[['judge_agreed', 'query_id']].astype(object).values.tolist())
connection.commit()
connection.close()
rows[['query_id', 'question_id', 'split', 'human_label', 'human_label_notes']].to_csv(LABELS_CSV, index=False)

# --- Summary report ---------------------------------------------------------------------------
report = [
    "# Judge vs human agreement (T22)",
    "",
    "Judge: Claude Sonnet 5.5, prompt `l3-v1`. Human: Adhith, blind, rubric = supported by the excerpts.",
    "1 = not supported (hallucination). Intervals are 95% bootstrap intervals "
    f"({BOOTSTRAP_ROUNDS} resamples).",
    "",
    "| Set | n | Raw agreement | Cohen's κ | Judge missed | Judge extra |",
    "|---|---|---|---|---|---|",
    *[line(result) for result in results],
    "",
    "*Judge missed* = human said not supported, judge passed it (the costly error for a detector). "
    "*Judge extra* = human said supported, judge failed it.",
    "",
    f"Headline: human marked {headline['human_not_supported']} of {headline['n']} not supported; "
    f"the judge failed {headline['judge_fail']}; both agreed on {headline['both_fail']}.",
    "",
    "## By train / test split (headline rows)",
    "",
    markdown_table(by_group(headline_rows, 'split')),
    "",
    "## By failure mode (headline rows; groups are small, read as direction only)",
    "",
    markdown_table(by_group(headline_rows, 'induction_mode')),
    "",
    "## L1 on the answers it decided",
    "",
    f"L1 gave pass/fail on {len(l1_rows)} of the {len(headline_rows)} headline answers and matched "
    f"the human label on {l1_agree}.",
    "",
]
if after_review:
    report += [
        "## After review (T22b): disagreements settled by blind tie-breaker models",
        "",
        f"Tie-breakers: {tiebreak_models}, shown the rubric, question, excerpts and answer only. "
        f"They sided with the judge on {sided_with_judge} of {len(disputed)} disagreements. "
        f"In {rule_breaks} cases a model called a plain refusal not supported, against the written rule; "
        "the second row corrects those.",
        "",
        "**This is an upper bound, not an independent result:** the tie-breakers are LLMs and can share "
        "the judge's blind spots. The blind headline above stays the independent figure.",
        "",
        "| Set | n | Raw agreement | Cohen's κ | Judge missed | Judge extra |",
        "|---|---|---|---|---|---|",
        *[line(result) for result in after_review],
        "",
    ]
REPORT_MD.write_text("\n".join(report))

# --- Every disagreement in full, for review -------------------------------------------------
cases = ["# Judge vs human disagreements (T22)", "",
         "Each case: what you said, what the judge said and why. Review step: decide who was right.", ""]
LABEL_WORDS = {0: 'Supported', 1: 'Not supported'}
for number, case in enumerate(rows[rows['judge_agreed'] == 0].itertuples(), start=1):
    cases += [
        f"## {number}. `{case.query_id[:8]}` · {case.split} · {case.induction_mode}"
        + (" · calibration row" if case.calibration else ""),
        "",
        f"**Question:** {case.query_text}",
        "",
        f"**Answer:** {case.answer_text}",
        "",
        f"**You:** {LABEL_WORDS[case.human_label]}" + (" (not sure)" if case.not_sure else "")
        + (f". Note: {case.human_comment}" if case.human_comment else ""),
        "",
        f"**Judge:** {LABEL_WORDS[int(case.judge_label)]}. {case.l3_rationale}",
        "",
    ]
    if case.l3_unsupported_span:
        cases += [f"**Claim the judge couldn't support:** {case.l3_unsupported_span}", ""]
    cases += ["<details><summary>Excerpts</summary>", ""]
    for index, excerpt in enumerate(case.retrieved_text.split("\n\n---\n\n"), start=1):
        cases += [f"**Excerpt {index}:** {' '.join(excerpt.split())}", ""]
    cases += ["</details>", ""]
DISAGREEMENTS_MD.write_text("\n".join(cases))

print("\n".join(report[5:10]))
print("\n".join(line(result) for result in after_review))
print(f"\nL1: decided {len(l1_rows)}, matched human on {l1_agree}")
print(f"\nWrote {REPORT_MD.relative_to(BASE_DIR)}, {DISAGREEMENTS_MD.relative_to(BASE_DIR)}, "
      f"{LABELS_CSV.relative_to(BASE_DIR)}; judge_agreed set on {len(rows)} rows")
