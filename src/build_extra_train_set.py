import json
import random
import sys
from pathlib import Path

import pandas as pd

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_seed_set import NO_DISTRACTOR_TYPES, load_candidates
from query_logger import make_question_id

# T23: 200 extra TRAIN questions for L2, labelled by the judge (decided 2026-11-16).
# Same 30 contracts, same 15 question templates and the same failure-mode mix as the seed set,
# but never a (contract, clause type) pair the seed set already uses, so no test question is
# repeated or paraphrased here. Every row is split = 'train'; the 40 test answers stay untouched.
#
#   python src/build_extra_train_set.py    ->  data/extra_train_set.csv

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
OUTPUT_CSV = BASE_DIR / 'data' / 'extra_train_set.csv'

RANDOM_SEED = 43    # not 42: a different draw from the seed set's, still the same on every run
# The seed set's mix (40/25/20/20/15 of 120), scaled to 200.
MODE_COUNTS = {'natural': 67, 'unanswerable': 41, 'starved_retrieval': 33, 'distractor': 33, 'weak_generator': 26}

answerable, unanswerable, chunks = load_candidates()
seed = pd.read_csv(SEED_CSV)
used = set(zip(seed['contract'], seed['clause_type']))
answerable = [pair for pair in answerable if (pair['contract'], pair['clause_type']) not in used]
unanswerable = [pair for pair in unanswerable if (pair['contract'], pair['clause_type']) not in used]
all_answerable = list(answerable)    # distractor chunks may come from any answerable pair, used or not

rng = random.Random(RANDOM_SEED)
rng.shuffle(answerable)
rng.shuffle(unanswerable)


def pick_distractor(pair):
    """Same clause type from a different contract with different wording: its best gold chunk."""
    if pair['clause_type'] in NO_DISTRACTOR_TYPES:
        return None
    options = [other for other in all_answerable
               if other['clause_type'] == pair['clause_type']
               and other['contract'] != pair['contract']
               and other['gold_spans'] != pair['gold_spans']]
    if not options:
        return None
    other = rng.choice(sorted(options, key=lambda row: row['contract']))    # sorted: same pick every run
    return {'distractor_contract': other['contract'], 'distractor_chunk_ids': other['gold_chunk_ids'][:1]}


rows = []


def take(pool, mode, count, extra_check=None):
    """Move `count` pairs from `pool` into the new set as `mode`. Each pair is used once."""
    taken = 0
    for pair in list(pool):
        if taken == count:
            break
        extra = extra_check(pair) if extra_check else {}
        if extra is None:
            continue
        pool.remove(pair)
        rows.append({**pair, 'induction_mode': mode, 'distractor_contract': '', 'distractor_chunk_ids': [], **extra})
        taken += 1
    assert taken == count, f"Only found {taken} of {count} questions for {mode}"


take(unanswerable, 'unanswerable', MODE_COUNTS['unanswerable'])
take(answerable, 'distractor', MODE_COUNTS['distractor'], pick_distractor)
for mode in ['starved_retrieval', 'weak_generator', 'natural']:
    take(answerable, mode, MODE_COUNTS[mode])

extra = pd.DataFrame(rows)
extra['split'] = 'train'
extra['question_id'] = [make_question_id(q, c) for q, c in zip(extra['question'], extra['contract'])]

# --- Checks, all before writing ----------------------------------------------------------
all_chunk_ids = set(chunks['chunk_id'])
assert len(extra) == sum(MODE_COUNTS.values())
assert extra['question_id'].is_unique, "Two rows share a question_id"
assert not set(extra['question_id']) & set(seed['question_id']), "A question repeats the seed set"
assert not set(zip(extra['contract'], extra['clause_type'])) & used, "A (contract, clause type) pair repeats"
for _, row in extra.iterrows():
    if row['induction_mode'] == 'unanswerable':
        assert row['gold_chunk_ids'] == []
    else:
        assert row['gold_chunk_ids'] and set(row['gold_chunk_ids']) <= all_chunk_ids
    if row['induction_mode'] == 'distractor':
        assert row['distractor_contract'] and row['distractor_contract'] != row['contract']

# --- Write, in the same format as seed_set.csv -------------------------------------------
columns = ['question_id', 'split', 'induction_mode', 'contract', 'clause_type', 'question',
           'gold_chunk_ids', 'gold_spans', 'distractor_contract', 'distractor_chunk_ids']
for column in ['gold_chunk_ids', 'gold_spans', 'distractor_chunk_ids']:
    extra[column] = extra[column].apply(json.dumps)
extra = extra[columns].sort_values(['induction_mode', 'contract', 'clause_type']).reset_index(drop=True)
extra.to_csv(OUTPUT_CSV, index=False)

print(f"Wrote {len(extra)} questions to {OUTPUT_CSV.name}")
print(extra['induction_mode'].value_counts().to_string())
print(f"Contracts: {extra['contract'].nunique()}, most questions in one contract: "
      f"{extra['contract'].value_counts().max()}; clause types: {extra['clause_type'].nunique()}")
