import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from l1_signals import numbers_in, resources, split_sentences, SECTION_REFERENCE

# T25: the features L2 learns from, for every answer in the seed run and the extra train run.
#
#   python src/l2_features.py    ->  data/l2_features.csv, and l2_features_json in query_log
#
# Two kinds:
#   - L1's signals, reused as they are (already in query_log, no recomputation)
#   - a few new ones that need the embedding model: average sentence support, whole-answer
#     support, answer-to-question relevance; plus counts read from the text
#
# BANNED (never a feature): induction_mode, retrieval_hit, gold chunks or spans, which generator
# model wrote the answer, and any judge or human label. A deployed system knows none of them, and
# the generator model would be a fingerprint of the weak_generator failure mode.

BASE_DIR = Path(__file__).resolve().parent.parent
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
EXTRA_CSV = BASE_DIR / 'data' / 'extra_train_set.csv'
OUT_CSV = BASE_DIR / 'data' / 'l2_features.csv'

L1_FEATURES = ['l1_token_overlap', 'l1_max_similarity', 'l1_mean_similarity', 'l1_score_spread',
               'l1_min_sentence_similarity', 'l1_answer_len_tokens', 'l1_len_ratio', 'l1_hedging_count']
NEW_FEATURES = ['mean_sentence_similarity', 'answer_best_chunk_similarity', 'question_answer_similarity',
                'n_numbers', 'n_sentences', 'is_paraphrased_refusal']
FEATURES = L1_FEATURES + NEW_FEATURES
# What the model in T27 uses. Two near-duplicates are left out (checked on train, 2026-11-17), so
# the coefficients stay readable: l1_len_ratio moves with answer length (correlation 0.99), and
# answer_best_chunk_similarity with mean_sentence_similarity (0.89). Both stay in the CSV.
MODEL_FEATURES = [name for name in FEATURES if name not in ('l1_len_ratio', 'answer_best_chunk_similarity')]


def new_features(question, answer, retrieved_chunk_ids, behaviour):
    """The new signals for one answer. Same embedding model and index vectors as L1."""
    model, index, positions = resources()
    chunk_vectors = np.stack([index.reconstruct(positions[chunk_id]) for chunk_id in retrieved_chunk_ids])
    sentences = split_sentences(answer)

    # Per sentence, its best match among the excerpts; L1 keeps only the lowest, this the average.
    if sentences:
        sentence_vectors = model.encode(sentences, normalize_embeddings=True)
        mean_sentence = float((sentence_vectors @ chunk_vectors.T).max(axis=1).mean())
    else:
        mean_sentence = None
    # The answer as a whole against its closest excerpt.
    answer_vector = model.encode([answer], normalize_embeddings=True)[0]
    best_chunk = float((chunk_vectors @ answer_vector).max())
    # Does the answer address the question? The question gets the "query" prompt, as in retrieval.
    question_vector = model.encode([question], prompt_name="query", normalize_embeddings=True)[0]
    relevance = float(question_vector @ answer_vector)

    return {
        'mean_sentence_similarity': mean_sentence,
        'answer_best_chunk_similarity': best_chunk,
        'question_answer_similarity': relevance,
        'n_numbers': len(numbers_in(SECTION_REFERENCE.sub(' ', answer))),    # claims that can be checked, citations aside
        'n_sentences': len(sentences),
        'is_paraphrased_refusal': int(behaviour == 'paraphrased_refusal'),
    }


def load_rows():
    """Every successful answer of the two runs, with its split and the inputs the features need."""
    connection = sqlite3.connect(query_logger.DB_PATH)
    rows = pd.read_sql_query(
        f"SELECT query_id, question_id, run_id, query_text, answer_text, retrieved_chunk_ids, "
        f"l1_verdict, l1_features_json, {', '.join(L1_FEATURES)} FROM query_log "
        f"WHERE run_id IN (?, ?) AND error IS NULL", connection,
        params=(query_logger.SEED_RUN_ID, query_logger.EXTRA_RUN_ID))
    connection.close()
    splits = pd.concat([pd.read_csv(path, usecols=['question_id', 'split']) for path in (SEED_CSV, EXTRA_CSV)])
    return rows.merge(splits, on='question_id', how='left')


if __name__ == '__main__':
    rows = load_rows()
    assert rows['split'].notna().all(), "Every answer must belong to the seed set or the extra train set"
    resources()    # load the embedding model once, before any timing

    computed, timings = [], []
    for row in rows.itertuples():
        start = time.perf_counter()
        behaviour = json.loads(row.l1_features_json)['answer_behaviour']
        computed.append(new_features(row.query_text, row.answer_text, json.loads(row.retrieved_chunk_ids), behaviour))
        timings.append((time.perf_counter() - start) * 1000)
    features = pd.concat([rows.reset_index(drop=True), pd.DataFrame(computed)], axis=1)

    # Into query_log, so each answer's features sit next to its verdicts...
    connection = sqlite3.connect(query_logger.DB_PATH)
    connection.executemany(
        "UPDATE query_log SET l2_features_json = ? WHERE query_id = ?",
        [(json.dumps({name: (None if pd.isna(value) else float(value)) for name, value in zip(FEATURES, values)}), query_id)
         for query_id, values in zip(features['query_id'], features[FEATURES].to_numpy())])
    connection.commit()
    connection.close()
    # ...and into a CSV that is committed (the database is gitignored), for training in T27.
    features[['query_id', 'run_id', 'split', 'l1_verdict', *FEATURES]].to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(features)} rows x {len(FEATURES)} features to {OUT_CSV.relative_to(BASE_DIR)}")
    print(f"New features: median {np.median(timings):.0f} ms per answer (embedding model already loaded)")
    print("Missing values:", features[FEATURES].isna().sum()[lambda s: s > 0].to_dict() or "none")
