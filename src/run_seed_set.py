import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from l1_signals import answer_behaviour    # the refusal sorter lives with the other L1 signals

# T13: run all 120 seed questions through the pipeline as ONE run, then summarise.
#   python src/run_seed_set.py                 run everything, then summarise
#   python src/run_seed_set.py --summary RUN   summarise an earlier run only (no model calls)

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'


def run_all():
    seed = pd.read_csv(SEED_CSV)
    connection = query_logger.connect()
    run_id = query_logger.new_run_id()
    print(f"Run {run_id}: {len(seed)} questions", flush=True)

    start = time.time()
    for number, row in enumerate(seed.itertuples(), start=1):
        result = query_logger.run_and_log(
            connection, row.question, row.contract, run_id,
            question_id=row.question_id,             # same ID as in seed_set.csv
            induction_mode=row.induction_mode,
            gold_chunk_ids=json.loads(row.gold_chunk_ids),
            distractor_chunk_ids=json.loads(row.distractor_chunk_ids),
        )
        status = 'ok' if result else 'FAILED'
        print(f"  {number:>3}/{len(seed)}  {row.induction_mode:<17} {status}  ({time.time() - start:.0f} s)", flush=True)

    connection.close()
    return run_id


def summarise(run_id):
    connection = sqlite3.connect(query_logger.DB_PATH)
    logged = pd.read_sql_query("SELECT * FROM query_log WHERE run_id = ?", connection, params=(run_id,))
    connection.close()
    seed = pd.read_csv(SEED_CSV)[['question_id', 'split']]
    df = logged.merge(seed, on='question_id', how='left')

    failed = df['error'].notna().sum()
    df = df[df['error'].isna()].copy()    # failed rows never count in results (T7 convention)
    df['behaviour'] = df['answer_text'].map(answer_behaviour)

    print(f"\nRun {run_id}: {len(df)} answered, {failed} failed")

    # Rebalancing decisions look at TRAIN rows only, so the 40 test answers stay unseen.
    train = df[df['split'] == 'train']
    print("\nTRAIN only: how the answers behaved, per mode")
    print(pd.crosstab(train['induction_mode'], train['behaviour'], margins=True).to_string())

    print("\nTRAIN only: retrieval_hit per mode (1 = the best answer chunk reached the model)")
    print(train.groupby('induction_mode')['retrieval_hit'].agg(['count', 'mean']).round(2).to_string())

    # "Likely wrong", by mode. Where the context can't contain the answer (unanswerable,
    # starved, distractor), giving an answer is very likely a hallucination; declining, in the
    # fixed sentence or in its own words, is the correct behaviour.
    # For natural and weak_generator a refusal-then-guess is the only signal without labels.
    no_answer_in_context = train['induction_mode'].isin(['unanswerable', 'starved_retrieval', 'distractor'])
    likely_wrong = ((no_answer_in_context & (train['behaviour'] == 'answered'))
                    | (train['behaviour'] == 'refused_then_guessed'))
    print(f"\nTRAIN: likely wrong (rough estimate, before reading any answer): "
          f"{likely_wrong.sum()} of {len(train)} = {likely_wrong.mean():.0%}   (target 30-40%)")
    print("Rough in both directions: natural/weak answers that look fine can still be wrong, and")
    print("some 'unanswerable' answers are grounded (CUAD marks a clause absent only when nothing")
    print("fits its narrow legal definition; the topic can still appear). Only M3 labels decide.")


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--summary':
        summarise(sys.argv[2])
    else:
        summarise(run_all())
