import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_logger import DB_PATH, SEED_RUN_ID

# T24: one reference label per TEST answer: what L2, the baselines and the cascade are graded on
# (decided 2026-11-16). 1 = not supported (hallucination), 0 = supported.
#   - Adhith and the judge agreed           -> that label                    (source: agreed)
#   - they disagreed, a tie-break exists    -> the tie-break vote, corrected
#                                              to the written refusal rule   (source: tiebreak)
#   - they disagreed, no tie-break          -> Adhith's label                (source: human_only)
#     (only calibration rows, which were left out of the blind tie-break file)
# Adhith's blind label and the judge's verdict are kept alongside, so every result can also be
# reported against them to show how much this choice matters.
#
#   python src/build_reference_labels.py    ->  data/reference_labels.csv

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
TIEBREAK_CSV = BASE_DIR / 'data' / 'tiebreak_votes.csv'
OUT_CSV = BASE_DIR / 'data' / 'reference_labels.csv'

connection = sqlite3.connect(DB_PATH)
rows = pd.read_sql_query(
    "SELECT query_id, question_id, human_label, l3_verdict FROM query_log WHERE run_id = ? AND error IS NULL",
    connection, params=(SEED_RUN_ID,))
connection.close()
rows = rows.merge(pd.read_csv(SEED_CSV, usecols=['question_id', 'split']), on='question_id')
test = rows[rows['split'] == 'test'].copy()
assert len(test) == 40 and test['human_label'].notna().all(), "Every test answer needs a human label"

test['human_label'] = test['human_label'].astype(int)
test['judge_label'] = test['l3_verdict'].map({'fail': 1, 'pass': 0}).astype(int)

votes = pd.read_csv(TIEBREAK_CSV)
votes['rule_vote'] = votes['vote'].where(votes['breaks_refusal_rule'] == 0, 0)    # a plain refusal is supported
tiebreak = votes.groupby('query_id')['rule_vote'].mean().round()                  # majority; a tie rounds to 0

agreed = test['human_label'] == test['judge_label']
has_tiebreak = test['query_id'].isin(tiebreak.index)
test['reference_label'] = test['human_label']
test.loc[~agreed & has_tiebreak, 'reference_label'] = test['query_id'].map(tiebreak).astype('Int64')
test['source'] = 'agreed'
test.loc[~agreed & has_tiebreak, 'source'] = 'tiebreak'
test.loc[~agreed & ~has_tiebreak, 'source'] = 'human_only'

test = test.sort_values('query_id')
test[['query_id', 'question_id', 'reference_label', 'source', 'human_label', 'judge_label']].to_csv(OUT_CSV, index=False)

print(f"Wrote {len(test)} reference labels to {OUT_CSV.relative_to(BASE_DIR)}")
print(test['source'].value_counts().to_string())
print(f"Not supported: reference {test['reference_label'].sum()}, human {test['human_label'].sum()}, "
      f"judge {test['judge_label'].sum()} (of {len(test)})")
print(f"Reference differs from human on {(test['reference_label'] != test['human_label']).sum()}, "
      f"from judge on {(test['reference_label'] != test['judge_label']).sum()}")
