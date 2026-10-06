import hashlib
import json
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Lets this file import rag.py from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / 'schema.sql'
DB_PATH = BASE_DIR / 'data' / 'query_log.db'    # gitignored (*.db): regenerated, never committed

# Columns holding lists or dicts. The schema stores these as JSON text.
JSON_COLUMNS = {'gold_chunk_ids', 'retrieved_chunk_ids', 'retrieved_scores', 'generation_params',
                'l0_rule_hits', 'l1_features_json', 'l2_features_json'}


def connect(db_path=DB_PATH):
    """Open the database, creating the query_log table if it doesn't exist yet."""
    connection = sqlite3.connect(db_path)
    # schema.sql only uses CREATE ... IF NOT EXISTS, so running it every time is safe.
    connection.executescript(SCHEMA_PATH.read_text(encoding='utf-8'))
    return connection


def new_run_id():
    """One ID per batch of questions run together, e.g. 'run-20261006T151203Z-3f9a'."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"run-{timestamp}-{uuid.uuid4().hex[:4]}"    # short random suffix: two runs in the same second stay distinct


def make_question_id(question, contract):
    """Same question about the same contract -> same ID, in every run.

    Built from the text itself (a hash), so no lookup table is needed. M2 can pass its own
    IDs instead. Any wording change gives a new ID, which is correct: it's a new question.
    """
    return "q-" + hashlib.sha1(f"{contract}\n{question}".encode('utf-8')).hexdigest()[:12]


def insert_row(connection, row):
    """Write one dict to query_log. Keys must be column names."""
    values = {column: json.dumps(value) if column in JSON_COLUMNS and value is not None else value
              for column, value in row.items()}
    columns = ', '.join(values)
    placeholders = ', '.join(f':{column}' for column in values)    # named placeholders: values are never pasted into the SQL text
    connection.execute(f"INSERT INTO query_log ({columns}) VALUES ({placeholders})", values)
    connection.commit()    # commit per row, so a crash mid-run loses at most one question


def run_and_log(connection, question, contract, run_id, question_id=None, induction_mode=None,
                gold_chunk_ids=None, k=rag.TOP_K, shadow_mode=0):
    """Ask one question through the RAG pipeline and save the result as one query_log row.

    If the pipeline fails, the row is still written with the error message, as schema.sql
    asks ("keep the row, don't drop it"), and None is returned instead of the row.
    """
    row = {
        'query_id': str(uuid.uuid4()),
        'run_id': run_id,
        'question_id': question_id or make_question_id(question, contract),
        'shadow_mode': shadow_mode,       # 0 until the evaluator layers exist (M2-M5)
        'induction_mode': induction_mode,
        'gold_chunk_ids': gold_chunk_ids,
    }

    try:
        row.update(rag.answer(question, contract, k))
        error = None
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        # Several columns are NOT NULL. Fill them with what is known, and empty values for
        # what was never produced. The non-NULL `error` column marks the row as failed,
        # so every analysis query filters on `error IS NULL`.
        row.update({
            'pipeline_config_id': rag.CONFIG['pipeline_config_id'],
            'created_at': datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            'query_text': question,
            'source_contract_id': contract,
            'embedding_model': rag.CONFIG['embedding_model'],
            'retrieval_k': k,
            'retrieved_chunk_ids': [],
            'retrieved_scores': [],
            'retrieved_text': '',
            'generator_model': rag.GENERATOR_MODEL,
            'answer_text': '',
            'error': error,
        })

    insert_row(connection, row)
    return None if error else row
