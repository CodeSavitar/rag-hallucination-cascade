import ast
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from query_logger import make_question_id    # same question_id formula as the logged rows

BASE_DIR = Path(__file__).resolve().parent.parent
CONTRACTS_FOLDER = BASE_DIR / 'CUAD_v1' / 'full_contract_txt'
MASTER_CSV = BASE_DIR / 'CUAD_v1' / 'master_clauses.csv'
SELECTED_CSV = BASE_DIR / 'data' / 'selected_contracts.csv'
TEMPLATES_CSV = BASE_DIR / 'data' / 'question_templates.csv'
CHUNKS_CSV = BASE_DIR / 'index' / 'chunks.csv'
OUTPUT_CSV = BASE_DIR / 'data' / 'seed_set.csv'

RANDOM_SEED = 42    # fixed, so the seed set is identical on every run

# How many questions per failure mode, and how many of those are held out as test.
# Test share is a third of each mode (40 of 120), so train and test have the same mix.
MODE_COUNTS = {'natural': 40, 'unanswerable': 25, 'starved_retrieval': 20, 'distractor': 20, 'weak_generator': 15}
TEST_COUNTS = {'natural': 13, 'unanswerable': 8, 'starved_retrieval': 7, 'distractor': 7, 'weak_generator': 5}

# Nearly every contract has these, so they're asked only where present (no unanswerable version).
ALWAYS_PRESENT_TYPES = {'Governing Law', 'Expiration Date', 'Anti-Assignment'}
# Most of the 30 contracts are governed by California law, so a distractor would usually say
# the same thing as the real clause and not be wrong at all. Excluded from distractor mode.
NO_DISTRACTOR_TYPES = {'Governing Law'}

# Spread questions out, so no single contract or clause type dominates the results.
MAX_PER_CONTRACT = 6
MAX_PER_CLAUSE_TYPE = 12


def locate_span(text, span):
    """Find where a CUAD clause span sits in the raw text. Returns [(start, end), ...] or None.

    Exact match first. CUAD sometimes cuts text out and writes "<omitted>", and sometimes
    differs in whitespace, so the fallback finds each piece separately, allowing any run of
    whitespace between words.
    """
    start = text.find(span)
    if start != -1:
        return [(start, start + len(span))]

    positions = []
    for piece in span.split('<omitted>'):
        words = piece.split()
        if not words:
            continue
        match = re.search(r'\s+'.join(re.escape(word) for word in words), text)
        if match is None:
            return None
        positions.append((match.start(), match.end()))
    return positions or None


def load_candidates():
    """Every (contract, clause type) pair in the 30 contracts: answerable ones with their gold
    chunks (best chunk first), and unanswerable ones. Also used by build_extra_train_set.py."""
    templates = pd.read_csv(TEMPLATES_CSV)
    selected = set(pd.read_csv(SELECTED_CSV)['Filename'])
    master = pd.read_csv(MASTER_CSV)
    master['contract'] = master['Filename'].map(lambda name: Path(name).stem + '.txt')
    master = master[master['contract'].isin(selected)]
    chunks = pd.read_csv(CHUNKS_CSV)
    texts = {name: (CONTRACTS_FOLDER / name).read_text(encoding='utf-8') for name in selected}

    # --- Every (contract, clause type) pair: answerable with gold chunks, or unanswerable --------
    answerable, unanswerable = [], []

    for _, contract_row in master.iterrows():
        contract = contract_row['contract']
        contract_chunks = chunks[chunks['filename'] == contract]

        for clause_type, question in zip(templates['clause_type'], templates['question']):
            pair = {'contract': contract, 'clause_type': clause_type, 'question': question}
            cell = contract_row[clause_type]

            if cell == '[]':    # CUAD stores "no such clause" as the string "[]"
                if clause_type not in ALWAYS_PRESENT_TYPES:
                    unanswerable.append({**pair, 'gold_chunk_ids': [], 'gold_spans': []})
                continue

            spans = ast.literal_eval(cell)    # the cell is a string that looks like a list
            positions = []
            for span in spans:
                located = locate_span(texts[contract], span)
                assert located is not None, f"Can't locate a span in {contract} [{clause_type}]: {span[:60]!r}"
                positions += located

            # Gold chunks = every chunk that overlaps any part of the answer. Taking all of them
            # (not just one) matters for starved_retrieval: removing only some would leave part
            # of the answer in the context.
            # Sorted by how many answer characters each chunk holds, MOST FIRST: a chunk that only
            # touches the edge of a clause is still listed, but the first id is always the chunk
            # that best contains the answer (used as the distractor, and for retrieval hits).
            def answer_chars_in(chunk):
                return sum(max(0, min(chunk['end_char'], end) - max(chunk['start_char'], start))
                           for start, end in positions)

            overlap = contract_chunks.apply(answer_chars_in, axis=1)                      # answer chars per chunk
            best_first = overlap[overlap > 0].sort_values(ascending=False, kind='stable')   # stable: ties keep document order
            gold_ids = contract_chunks.loc[best_first.index, 'chunk_id'].tolist()
            assert gold_ids, f"No chunk overlaps the answer in {contract} [{clause_type}]"
            answerable.append({**pair, 'gold_chunk_ids': gold_ids, 'gold_spans': spans})

    print(f"Candidates: {len(answerable)} answerable pairs, {len(unanswerable)} unanswerable pairs")
    return answerable, unanswerable, chunks


if __name__ == '__main__':
    answerable, unanswerable, chunks = load_candidates()

    # --- Sample each mode, never reusing a (contract, clause type) pair ------------------------
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(answerable)
    rng.shuffle(unanswerable)

    per_contract, per_clause = {}, {}
    seed_rows = []


    def take(pool, mode, count, extra_check=None):
        """Move `count` pairs from `pool` into the seed set as `mode`, respecting the caps."""
        taken = 0
        for pair in list(pool):
            if taken == count:
                break
            if per_contract.get(pair['contract'], 0) >= MAX_PER_CONTRACT:
                continue
            if per_clause.get(pair['clause_type'], 0) >= MAX_PER_CLAUSE_TYPE:
                continue
            extra = extra_check(pair) if extra_check else {}
            if extra is None:    # the check rejected this pair
                continue
            pool.remove(pair)    # used once only: the same pair never appears in two modes
            per_contract[pair['contract']] = per_contract.get(pair['contract'], 0) + 1
            per_clause[pair['clause_type']] = per_clause.get(pair['clause_type'], 0) + 1
            seed_rows.append({**pair, 'induction_mode': mode, **extra})
            taken += 1
        assert taken == count, f"Only found {taken} of {count} questions for {mode}"


    def pick_distractor(pair):
        """Choose the same clause type from a different contract, with different wording."""
        if pair['clause_type'] in NO_DISTRACTOR_TYPES:
            return None
        options = [other for other in answerable + [row for row in seed_rows if row['gold_chunk_ids']]
                   if other['clause_type'] == pair['clause_type']
                   and other['contract'] != pair['contract']
                   and other['gold_spans'] != pair['gold_spans']]
        if not options:
            return None
        other = rng.choice(sorted(options, key=lambda row: row['contract']))    # sorted: same pick every run
        # Only the other contract's BEST gold chunk: the one that actually contains its clause.
        return {'distractor_contract': other['contract'], 'distractor_chunk_ids': other['gold_chunk_ids'][:1]}


    # Hardest-to-fill modes first, so the caps don't starve them.
    take(unanswerable, 'unanswerable', MODE_COUNTS['unanswerable'])
    take(answerable, 'distractor', MODE_COUNTS['distractor'], pick_distractor)
    for mode in ['starved_retrieval', 'weak_generator', 'natural']:
        take(answerable, mode, MODE_COUNTS[mode])

    # --- Train/test split: a fixed third of each mode is held out ------------------------------
    seed_df = pd.DataFrame(seed_rows)
    seed_df['split'] = 'train'
    for mode, n_test in TEST_COUNTS.items():
        mode_index = seed_df.index[seed_df['induction_mode'] == mode].tolist()
        seed_df.loc[rng.sample(mode_index, n_test), 'split'] = 'test'

    seed_df['question_id'] = [make_question_id(q, c) for q, c in zip(seed_df['question'], seed_df['contract'])]
    for column in ['distractor_contract']:
        seed_df[column] = seed_df[column].fillna('')
    seed_df['distractor_chunk_ids'] = seed_df['distractor_chunk_ids'].apply(lambda ids: ids if isinstance(ids, list) else [])

    # --- Checks, all before writing ----------------------------------------------------------
    all_chunk_ids = set(chunks['chunk_id'])
    assert len(seed_df) == sum(MODE_COUNTS.values()), "Wrong number of questions"
    assert seed_df['question_id'].is_unique, "Two rows share a question_id"
    for mode, count in MODE_COUNTS.items():
        assert (seed_df['induction_mode'] == mode).sum() == count, f"Wrong count for {mode}"
        assert ((seed_df['induction_mode'] == mode) & (seed_df['split'] == 'test')).sum() == TEST_COUNTS[mode]
    for _, row in seed_df.iterrows():
        if row['induction_mode'] == 'unanswerable':
            assert row['gold_chunk_ids'] == [], "An unanswerable question has gold chunks"
        else:
            assert row['gold_chunk_ids'] and set(row['gold_chunk_ids']) <= all_chunk_ids, "Bad gold chunks"
        if row['induction_mode'] == 'distractor':
            assert row['distractor_contract'] and row['distractor_contract'] != row['contract'], "Bad distractor"
            assert set(row['distractor_chunk_ids']) <= all_chunk_ids, "Distractor chunk not in the index"

    # --- Write -------------------------------------------------------------------------------
    columns = ['question_id', 'split', 'induction_mode', 'contract', 'clause_type', 'question',
               'gold_chunk_ids', 'gold_spans', 'distractor_contract', 'distractor_chunk_ids']
    for column in ['gold_chunk_ids', 'gold_spans', 'distractor_chunk_ids']:
        seed_df[column] = seed_df[column].apply(json.dumps)    # lists stored as JSON text, like query_log
    seed_df = seed_df[columns].sort_values(['induction_mode', 'contract', 'clause_type']).reset_index(drop=True)
    seed_df.to_csv(OUTPUT_CSV, index=False)

    print(f"Wrote {len(seed_df)} questions to {OUTPUT_CSV.name}\n")
    print(pd.crosstab(seed_df['induction_mode'], seed_df['split'], margins=True).to_string())
