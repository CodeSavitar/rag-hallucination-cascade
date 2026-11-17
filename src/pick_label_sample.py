import sqlite3
from pathlib import Path

import pandas as pd

# T21 (cut down on 2026-11-14): Adhith labels 60 answers instead of all 120.
#   - all 40 TEST answers: the held-out set the cascade's recall is reported on (M5), so it gets
#     full human ground truth.
#   - 20 TRAIN answers, a stratified random sample: each failure mode keeps its share. Together
#     with the test rows they give 60 labels for judge-vs-human agreement (T22).
# L2 trains on the judge's labels for all 80 train answers (distillation; see KNOWLEDGE.md).
# The 5 calibration rows reviewed outside the blind process are excluded from the agreement figure in T22. The 3 of
# them in the test set stay in the sample, so Adhith relabels them in his own words.
# Writes data/label_sample.csv, which the labelling page reads. Same seed, same 60.

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'data' / 'query_log.db'
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
OUT_CSV = BASE_DIR / 'data' / 'label_sample.csv'
TRAIN_SAMPLE_SIZE = 20
RANDOM_SEED = 42
CALIBRATION_PREFIXES = ('88c27239', 'e1e43e17', '42afd15e', 'b510a476', '08aec7af')

connection = sqlite3.connect(DB_PATH)
run_id = 'run-20261007T150409Z-360c'    # the seed-set run, as pinned in query_logger.SEED_RUN_ID
rows = pd.read_sql_query("SELECT query_id, question_id FROM query_log WHERE run_id = ? AND error IS NULL",
                         connection, params=(run_id,))
connection.close()
rows = rows.merge(pd.read_csv(SEED_CSV, usecols=['question_id', 'split', 'induction_mode']), on='question_id')

test = rows[rows['split'] == 'test']
train = rows[(rows['split'] == 'train') & ~rows['query_id'].str.startswith(CALIBRATION_PREFIXES)]

# Give each failure mode its fair share of the 20, rounding so the shares add up to exactly 20
# (largest remainder: the modes that lost most to rounding down get the leftover places).
groups = train.groupby('induction_mode')
exact = groups.size() * TRAIN_SAMPLE_SIZE / len(train)
shares = exact.astype(int)
leftover = TRAIN_SAMPLE_SIZE - shares.sum()
shares[(exact - shares).sort_values(ascending=False).index[:leftover]] += 1
train_sample = pd.concat(group.sample(n=shares[mode], random_state=RANDOM_SEED) for mode, group in groups)

sample = pd.concat([test, train_sample])
sample[['query_id']].to_csv(OUT_CSV, index=False)
print(f"Wrote {len(sample)} answers to {OUT_CSV.name}")
print(sample.groupby(['induction_mode', 'split']).size().unstack(fill_value=0))
