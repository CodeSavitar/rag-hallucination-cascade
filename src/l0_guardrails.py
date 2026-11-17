import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger

# T34: L0, input guardrails. Runs on the user's QUESTION, before retrieval, and flags text that
# tries to take over the model instead of asking about the contract.
#
# Deliberately rule-based and small (plan scope: L0 is a checkbox, not the project). Rules catch
# known phrasings only; a reworded attack gets through. That limit is stated in the README.
#
#   python src/l0_guardrails.py
#     -> tests the rules on data/l0_test_prompts.csv (hand-written attacks and benign look-alikes)
#     -> runs them on the 320 real questions (every hit there is a false alarm) and writes l0_* columns

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_CSV = BASE_DIR / 'data' / 'l0_test_prompts.csv'

RULES = {
    # "Ignore the previous instructions", "disregard all prior rules", "forget your instructions"
    'override_instructions': re.compile(
        r"\b(ignore|disregard|forget|override|bypass)\b\W+(?:\w+\W+){0,3}?"
        r"(instructions?|rules|prompts?|guidelines|directions|constraints)\b", re.IGNORECASE),
    # Asking the model to reveal or repeat its hidden setup
    'reveal_prompt': re.compile(
        r"\b(reveal|show|print|repeat|output|display|tell me)\b\W+(?:\w+\W+){0,3}?"
        r"(system\s+prompt|hidden\s+prompt|initial\s+prompt|your\s+(?:instructions|prompt|rules))\b", re.IGNORECASE),
    # Re-assigning the model's role: "you are now", "pretend you are", "act as an unrestricted"
    'role_change': re.compile(
        r"\byou\s+are\s+now\b|\bpretend\s+(?:to\s+be|you\s+are)\b|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|unfiltered|different|new)\b"
        r"|\bfrom\s+now\s+on\s+you\b", re.IGNORECASE),
    # Known jailbreak names and modes
    'jailbreak_terms': re.compile(r"\b(jailbreak|DAN|developer\s+mode|god\s+mode|do\s+anything\s+now)\b"),
    # Fake chat-format markers that try to start a new system or assistant turn
    'fake_turn_markers': re.compile(
        r"</?\s*(system|assistant)\s*>|\[/?INST\]|<\|im_start\|>|^\s*#{2,}\s*(system|instruction)", re.IGNORECASE | re.MULTILINE),
    # A long base64-looking run: hidden instructions smuggled in an encoding
    'encoded_payload': re.compile(r"[A-Za-z0-9+/]{60,}={0,2}"),
}


def check(question):
    """{'flagged': 0/1, 'rule_hits': [names of rules that fired], 'latency_ms': ...} for one question."""
    start = time.perf_counter()
    hits = [name for name, pattern in RULES.items() if pattern.search(question)]
    return {'flagged': int(bool(hits)), 'rule_hits': hits, 'latency_ms': (time.perf_counter() - start) * 1000}


if __name__ == '__main__':
    # 1. Hand-written test prompts: attacks should be flagged, look-alikes should not.
    tests = pd.read_csv(TEST_CSV)
    tests['flagged'] = [check(text)['flagged'] for text in tests['text']]
    tests['hits'] = [", ".join(check(text)['rule_hits']) for text in tests['text']]
    attacks, benign = tests[tests['is_attack'] == 1], tests[tests['is_attack'] == 0]
    print(f"Hand-written attacks flagged: {attacks['flagged'].sum()} of {len(attacks)}")
    print(f"Benign look-alikes flagged:   {benign['flagged'].sum()} of {len(benign)}")
    for row in tests[tests['flagged'] != tests['is_attack']].itertuples():
        print(f"  WRONG ({'missed attack' if row.is_attack else 'false alarm'}): {row.text[:90]}  [{row.hits}]")

    # 2. The 320 real questions: all are genuine, so every flag is a false alarm.
    connection = sqlite3.connect(query_logger.DB_PATH)
    questions = connection.execute(
        "SELECT query_id, query_text FROM query_log WHERE run_id IN (?, ?) AND error IS NULL",
        (query_logger.SEED_RUN_ID, query_logger.EXTRA_RUN_ID)).fetchall()
    results = [(query_id, check(text)) for query_id, text in questions]
    connection.executemany(
        "UPDATE query_log SET l0_flagged = ?, l0_rule_hits = ?, latency_l0_ms = ? WHERE query_id = ?",
        [(r['flagged'], json.dumps(r['rule_hits']), r['latency_ms'], query_id) for query_id, r in results])
    connection.commit()
    connection.close()
    false_alarms = sum(r['flagged'] for _, r in results)
    latencies = sorted(r['latency_ms'] for _, r in results)
    print(f"Real questions flagged (false alarms): {false_alarms} of {len(results)}; "
          f"median {latencies[len(latencies) // 2]:.3f} ms per question")
