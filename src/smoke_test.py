import sqlite3
import sys
from pathlib import Path

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger

# T8 smoke test: 10 questions through the full pipeline into the REAL database, then
# check that no column that should be filled came back NULL. Each run adds a new run_id;
# earlier runs are left alone.

CONTRACTS = [
    'NETGEAR,INC_04_21_2003-EX-10.16-DISTRIBUTOR AGREEMENT.txt',
    'CORIOINC_07_20_2000-EX-10.5-LICENSE AND HOSTING AGREEMENT.txt',
    'AURASYSTEMSINC_06_16_2010-EX-10.25-STRATEGIC ALLIANCE AGREEMENT.txt',
    'EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.txt',
    'LegacyEducationAllianceInc_20200330_10-K_EX-10.18_12090678_EX-10.18_Development Agreement.txt',
]

# Generic questions any contract can be asked. 5 contracts x 2 questions = 10 rows.
QUESTIONS = [
    "Which state's law governs the agreement?",
    "Can either party assign the agreement without the other party's consent?",
]

# Every column the RAG side is responsible for. NULL in any of these is a silent bug.
# (Evaluator columns, labels and gold_chunk_ids are NULL on purpose until M2-M5.)
EXPECTED_FILLED = [
    'query_id', 'run_id', 'question_id', 'pipeline_config_id', 'created_at', 'shadow_mode',
    'query_text', 'source_contract_id', 'induction_mode',
    'embedding_model', 'retrieval_k', 'retrieved_chunk_ids', 'retrieved_scores', 'retrieved_text',
    'generator_model', 'generation_params', 'prompt_version', 'answer_text',
    'prompt_tokens', 'completion_tokens',
    'latency_retrieval_ms', 'latency_generation_ms', 'cost_embedding_usd', 'cost_generation_usd',
]

connection = query_logger.connect()
run_id = query_logger.new_run_id()
print(f"Run {run_id}\n")

for contract in CONTRACTS:
    for question in QUESTIONS:
        # induction_mode='natural': the unmodified pipeline, the honest baseline
        row = query_logger.run_and_log(connection, question, contract, run_id, induction_mode='natural')
        status = row['answer_text'][:90] if row else "FAILED (see the error column)"
        print(f"{contract[:28]:<28} | {question[:30]:<30} | {status}")

# Read back from the database, not from memory: this checks what was actually stored.
rows = connection.execute("SELECT * FROM query_log WHERE run_id = ?", (run_id,)).fetchall()
column_names = [description[0] for description in connection.execute("SELECT * FROM query_log LIMIT 0").description]
connection.close()

problems = []
for values in rows:
    row = dict(zip(column_names, values))
    if row['error'] is not None:
        problems.append(f"{row['question_id']}: failed: {row['error']}")
    for column in EXPECTED_FILLED:
        if row[column] in (None, '', '[]'):
            problems.append(f"{row['question_id']}: {column} is empty")

print(f"\nRows written: {len(rows)} | problems: {len(problems)}")
for problem in problems:
    print("  " + problem)

assert len(rows) == len(CONTRACTS) * len(QUESTIONS), "Some questions were not written to the database"
assert not problems, "Smoke test failed"
print("Smoke test passed: every row complete, no silent NULLs.")
