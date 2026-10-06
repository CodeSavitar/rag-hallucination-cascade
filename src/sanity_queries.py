import sqlite3
import sys
from pathlib import Path

# Lets this script import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger

# T9: the four sanity queries from the foot of schema.sql, proving the table can answer the
# questions it was designed for. Every query filters `error IS NULL` (failed rows, see T7).
#
# Two passes:
#   1. Self-test on 10 made-up rows whose correct answers were worked out by hand.
#      This proves the SQL is right before there is any real evaluator data.
#   2. The real database. Evaluator columns are still empty, so most results are empty;
#      what matters there is that the queries run.

QUERIES = {
    # Share of questions where the cascade stopped at each layer. The headline number.
    'escalation_rate': """
        SELECT terminating_layer, COUNT(*) * 1.0 / SUM(COUNT(*)) OVER () AS share
        FROM query_log WHERE run_id = :run_id AND error IS NULL
        GROUP BY terminating_layer ORDER BY terminating_layer""",

    # Average evaluation cost per question: cascade vs judging everything.
    # FIXED from schema.sql's first version, which added up ALL layers' costs for the
    # cascade. In shadow mode every layer runs on every row, so that counted judge calls
    # the cascade would never have made, and the cascade came out MORE expensive than
    # judging everything. A cascade only pays for the layers up to where it stopped.
    'cost_per_query': """
        SELECT AVG(COALESCE(cost_embedding_usd, 0) +
                   CASE terminating_layer
                       WHEN 'L1' THEN COALESCE(cost_l1_usd, 0)
                       WHEN 'L2' THEN COALESCE(cost_l1_usd, 0) + COALESCE(cost_l2_usd, 0)
                       WHEN 'L3' THEN COALESCE(cost_l1_usd, 0) + COALESCE(cost_l2_usd, 0) + COALESCE(cost_l3_usd, 0)
                   END) AS cascade_cost,
               AVG(COALESCE(cost_embedding_usd, 0) + COALESCE(cost_l3_usd, 0)) AS judge_everything_cost
        FROM query_log
        WHERE run_id = :run_id AND error IS NULL
          AND shadow_mode = 1    -- judge-everything needs the judge to have run on EVERY row""",

    # Of the hallucinations the cascade let through, which layer wrongly cleared them.
    'miss_attribution': """
        SELECT terminating_layer, COUNT(*) AS misses
        FROM query_log
        WHERE run_id = :run_id AND error IS NULL AND human_label = 1 AND cascade_verdict = 'pass'
        GROUP BY terminating_layer ORDER BY terminating_layer""",

    # Share of hallucinations caught, per way of causing them. Where the honest finding lives.
    'detection_by_mode': """
        SELECT induction_mode,
               SUM(CASE WHEN cascade_verdict = 'fail' THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS caught
        FROM query_log
        WHERE run_id = :run_id AND error IS NULL AND human_label = 1
        GROUP BY induction_mode ORDER BY induction_mode""",
}


def run_queries(connection, run_id):
    """Run all four queries for one run. Returns {name: list of result rows}."""
    return {name: connection.execute(sql, {'run_id': run_id}).fetchall() for name, sql in QUERIES.items()}


def self_test():
    """Run the queries on 10 hand-made rows and compare with answers worked out by hand."""
    connection = query_logger.connect(':memory:')    # a throwaway database in memory

    # (terminating layer, human_label, cascade_verdict, induction_mode) for each of 10 rows
    rows = [
        ('L1', 1, 'pass', 'distractor'),      # a hallucination L1 wrongly passed -> a miss at L1
        ('L1', 0, 'pass', 'natural'),
        ('L1', 0, 'pass', 'natural'),
        ('L1', 0, 'pass', 'natural'),
        ('L1', 0, 'pass', 'natural'),
        ('L2', 1, 'fail', 'distractor'),      # caught at L2
        ('L2', 0, 'pass', 'natural'),
        ('L2', 0, 'pass', 'natural'),
        ('L3', 1, 'fail', 'unanswerable'),    # caught at L3
        ('L3', 1, 'pass', 'unanswerable'),    # a hallucination even the judge passed -> a miss at L3
    ]
    for i, (layer, label, verdict, mode) in enumerate(rows):
        query_logger.insert_row(connection, {
            'query_id': f'test-{i}', 'run_id': 'test-run', 'question_id': f'q{i}',
            'pipeline_config_id': 'test', 'created_at': '2026-01-01T00:00:00Z', 'shadow_mode': 1,
            'query_text': 'q', 'embedding_model': 'm', 'retrieval_k': 5,
            'retrieved_chunk_ids': [], 'retrieved_scores': [], 'retrieved_text': '',
            'generator_model': 'g', 'answer_text': 'a',
            'terminating_layer': layer, 'human_label': label, 'cascade_verdict': verdict,
            'induction_mode': mode,
            # Shadow mode: every layer ran, so every row carries every layer's cost.
            'cost_embedding_usd': 0.0, 'cost_l1_usd': 0.0, 'cost_l2_usd': 0.0001, 'cost_l3_usd': 0.004,
        })

    results = run_queries(connection, 'test-run')
    connection.close()

    assert results['escalation_rate'] == [('L1', 0.5), ('L2', 0.3), ('L3', 0.2)], results['escalation_rate']

    # By hand: L1 rows cost 0; L2 rows 0.0001; L3 rows 0.0041.
    # (5 x 0 + 3 x 0.0001 + 2 x 0.0041) / 10 = 0.00085. Judge-everything: 0.004 on every row.
    cascade_cost, judge_cost = results['cost_per_query'][0]
    assert abs(cascade_cost - 0.00085) < 1e-12 and abs(judge_cost - 0.004) < 1e-12, results['cost_per_query']

    assert results['miss_attribution'] == [('L1', 1), ('L3', 1)], results['miss_attribution']
    assert results['detection_by_mode'] == [('distractor', 0.5), ('unanswerable', 0.5)], results['detection_by_mode']
    print("Self-test passed: all 4 queries match the hand-worked answers.")


if __name__ == '__main__':
    self_test()

    connection = query_logger.connect()
    run_id = connection.execute("SELECT run_id FROM query_log ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    rows = connection.execute("SELECT COUNT(*) FROM query_log WHERE run_id = ?", (run_id,)).fetchone()[0]
    print(f"\nReal database, latest run {run_id} ({rows} rows):")
    for name, result in run_queries(connection, run_id).items():
        print(f"  {name:<18} {result}")
    connection.close()
