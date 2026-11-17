import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger
from cascade import load_answers
from sweep import with_oof_scores

# T35: everything the public app needs, in committed files, so it never touches the database
# (gitignored) or loads a model.
#
#   python src/export_app_data.py  ->  app/data/answers.csv, app/data/operating_point.json
#
# One row per answer (320): the question and the text the RAG system saw, its answer, and every
# layer's output. L2's score is OUT-OF-FOLD on train, as in the sweep, so the app's curve matches
# the reports.

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DATA = BASE_DIR / 'app' / 'data'

if __name__ == '__main__':
    rows = with_oof_scores(load_answers())
    connection = sqlite3.connect(query_logger.DB_PATH)
    text = pd.read_sql_query(
        "SELECT query_id, source_contract_id, query_text, retrieved_text, answer_text, l0_flagged, "
        "l1_token_overlap, l1_min_sentence_similarity, l1_features_json, l3_rationale, l3_unsupported_span "
        "FROM query_log", connection)
    connection.close()
    rows = rows.merge(text, on='query_id')
    rows['answer_behaviour'] = rows['l1_features_json'].map(lambda value: json.loads(value)['answer_behaviour'])
    labels = pd.read_csv(BASE_DIR / 'data' / 'reference_labels.csv')[['query_id', 'human_label']]
    rows = rows.merge(labels, on='query_id', how='left')

    columns = ['query_id', 'split', 'induction_mode', 'source_contract_id', 'query_text', 'retrieved_text', 'answer_text',
               'l0_flagged', 'l1_verdict', 'answer_behaviour', 'l1_token_overlap', 'l1_min_sentence_similarity', 'l1_len_ratio',
               'l2_score', 'l3_verdict', 'l3_unsupported_span', 'l3_rationale',
               'truth', 'judge_label', 'reference_label', 'human_label',
               'cost_l1_usd', 'cost_l2_usd', 'cost_l3_usd', 'latency_l1_ms', 'latency_l2_ms', 'latency_l3_ms']
    APP_DATA.mkdir(parents=True, exist_ok=True)
    rows[columns].sort_values(['split', 'query_id']).to_csv(APP_DATA / 'answers.csv', index=False)
    (APP_DATA / 'operating_point.json').write_text((BASE_DIR / 'data' / 'operating_point.json').read_text())
    size = (APP_DATA / 'answers.csv').stat().st_size / 1e6
    print(f"Wrote {len(rows)} answers ({size:.1f} MB) and the operating point to {APP_DATA.relative_to(BASE_DIR)}")
