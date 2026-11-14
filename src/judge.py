import json
import sqlite3
import sys
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import budget
import query_logger

# L3: the LLM judge (T18). Reads a logged answer, asks Claude whether the answer is supported
# by the excerpts the RAG system actually used, and writes the l3_* columns.
#
#   python src/judge.py --sample N    judge N train answers (testing the prompt; costs cents)
#   python src/judge.py --all         judge every answer in the latest seed run (T19)
#
# Every call goes through budget.py: refused if its worst case would cross the $15 cap.

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / '.env')    # explicit path: bare load_dotenv() can fail (see KNOWLEDGE.md)

JUDGE_MODEL = 'claude-sonnet-5-5'    # decided 2026-10-09; a different family from the Mistral generator
PROMPT_VERSION = 'l3-v1'             # bump on ANY change to the prompt or schema below
EFFORT = 'medium'                    # how hard the model thinks; thinking tokens are billed as output
MAX_TOKENS = 6000                    # thinking + the JSON verdict; also the budget guard's worst case
# No `fallbacks` parameter on purpose: a refused request would be re-run on another, possibly
# pricier model, outside the budget guard. A refusal is logged as "judge didn't decide" instead.

SYSTEM_PROMPT = """You are a strict grader for a contract question-answering system. You check one
thing: is the system's answer supported by the contract excerpts it was given?

Grade ONLY against the excerpts. Do not use outside knowledge or what contracts usually say.

Verdict "fail" if the answer contains any claim the excerpts do not support:
- a fact, number, date, party, condition, or yes/no conclusion that is not in the excerpts, or
  that contradicts them;
- a conclusion that misstates what the excerpts say, even when its details are copied from them.
  Example: the excerpts say "neither party may assign without consent, except on a merger", and
  the answer says "yes, either party can assign without consent, but only on a merger". The
  headline claim is unsupported, so this is "fail".
- If the answer says the information is not available and then guesses anyway, grade the guess.

Verdict "pass" if every claim in the answer is supported by the excerpts. In particular:
- An answer that only says the excerpts do not contain the answer is "pass", even if the excerpts
  did contain it. That is unhelpful, not unsupported. Set is_refusal to true.
- Ignore section or excerpt numbers the answer uses as citations; grade the substance.

Fields:
- verdict: "pass" or "fail"
- groundedness: 0.0 to 1.0, the share of the answer's claims that the excerpts support
- relevance: 0.0 to 1.0, how directly the answer addresses the question
- unsupported_span: copy the exact words of the main unsupported claim from the answer; "" if none
- is_refusal: true if the answer declines to answer (with or without a guess afterwards)
- rationale: at most three sentences explaining the verdict"""

OUTPUT_SCHEMA = {
    'type': 'object',
    'properties': {
        'verdict': {'type': 'string', 'enum': ['pass', 'fail']},
        'groundedness': {'type': 'number'},
        'relevance': {'type': 'number'},
        'unsupported_span': {'type': 'string'},
        'is_refusal': {'type': 'boolean'},
        'rationale': {'type': 'string'},
    },
    'required': ['verdict', 'groundedness', 'relevance', 'unsupported_span', 'is_refusal', 'rationale'],
    'additionalProperties': False,
}

CLIENT = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY from the environment (.env above)


def build_user_message(question, context, answer):
    """The judge sees exactly what the generator saw, plus the answer. Never the gold clause."""
    return (f"<excerpts>\n{context}\n</excerpts>\n\n"
            f"<question>\n{question}\n</question>\n\n"
            f"<answer>\n{answer}\n</answer>")


def judge(question, context, answer):
    """Grade one answer. Returns a dict keyed by query_log column names.

    On a refusal, truncation or unreadable output, l3_verdict stays None (the judge didn't
    decide) and the raw response is kept, so nothing is lost and nothing is guessed.
    """
    user_message = build_user_message(question, context, answer)
    # Budget check uses a deliberate over-estimate of the prompt: ~3 characters per token
    # (English is closer to 4), so the guard always errs on the side of stopping.
    estimated_input_tokens = (len(SYSTEM_PROMPT) + len(user_message)) // 3
    budget.check_budget(JUDGE_MODEL, estimated_input_tokens, MAX_TOKENS)

    start = time.perf_counter()
    response = CLIENT.messages.create(
        model=JUDGE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_message}],
        output_config={'effort': EFFORT,
                       'format': {'type': 'json_schema', 'schema': OUTPUT_SCHEMA}},
    )
    latency_ms = (time.perf_counter() - start) * 1000

    # Charged whatever the outcome, so recorded before anything can go wrong below.
    cost = budget.record_spend(JUDGE_MODEL, response.usage.input_tokens, response.usage.output_tokens,
                               purpose=f'judge {PROMPT_VERSION}')

    row = {
        'l3_judge_model': JUDGE_MODEL,
        'l3_prompt_version': PROMPT_VERSION,
        'latency_l3_ms': latency_ms,
        'cost_l3_usd': cost,
        'l3_verdict': None, 'l3_groundedness': None, 'l3_relevance': None,
        'l3_unsupported_span': None, 'l3_rationale': None,
    }
    text = next((block.text for block in response.content if block.type == 'text'), '')
    row['l3_raw_response'] = json.dumps({'stop_reason': response.stop_reason, 'text': text,
                                         'input_tokens': response.usage.input_tokens,
                                         'output_tokens': response.usage.output_tokens})

    if response.stop_reason != 'end_turn':    # refusal, or ran out of max_tokens mid-answer
        return row
    verdict = json.loads(text)                # output_config.format guarantees valid JSON here
    row.update({
        'l3_verdict': verdict['verdict'],
        'l3_groundedness': min(max(verdict['groundedness'], 0.0), 1.0),    # schema can't enforce 0..1
        'l3_relevance': min(max(verdict['relevance'], 0.0), 1.0),
        'l3_unsupported_span': verdict['unsupported_span'] or None,         # "" -> NULL: nothing unsupported
        'l3_rationale': verdict['rationale'],
    })
    return row


def judge_rows(query_ids):
    """Judge the given query_log rows, skipping any already judged with this prompt version."""
    connection = sqlite3.connect(query_logger.DB_PATH)
    placeholders = ', '.join('?' for _ in query_ids)
    rows = connection.execute(
        f"SELECT query_id, query_text, retrieved_text, answer_text FROM query_log "
        f"WHERE query_id IN ({placeholders}) AND error IS NULL "
        f"AND (l3_prompt_version IS NULL OR l3_prompt_version != ?)",    # never pay twice for one answer
        (*query_ids, PROMPT_VERSION)).fetchall()

    for number, (query_id, question, context, answer) in enumerate(rows, start=1):
        try:
            result = judge(question, context, answer)
        except budget.BudgetExceeded as exc:
            print(f"STOPPED: {exc}")
            break
        assignments = ', '.join(f"{column} = :{column}" for column in result)
        connection.execute(f"UPDATE query_log SET {assignments} WHERE query_id = :query_id",
                           {**result, 'query_id': query_id})
        connection.commit()    # per row: a stop halfway keeps everything already paid for
        print(f"  {number:>3}/{len(rows)}  {result['l3_verdict'] or 'NO VERDICT':<10} "
              f"${result['cost_l3_usd']:.4f}  {result['latency_l3_ms'] / 1000:.1f} s", flush=True)
    connection.close()
    print(f"Total spent so far: ${budget.spent_so_far():.4f} of the ${budget.SPEND_CAP_USD:.2f} cap")


def pick_sample(run_id):
    """5 TRAIN answers of different kinds, one per rubric rule, for testing the prompt cheaply."""
    import pandas as pd
    from l1_signals import answer_behaviour
    connection = sqlite3.connect(query_logger.DB_PATH)
    logged = pd.read_sql_query("SELECT query_id, question_id, induction_mode, answer_text, retrieval_hit "
                               "FROM query_log WHERE run_id = ? AND error IS NULL", connection, params=(run_id,))
    connection.close()
    seed = pd.read_csv(BASE_DIR / 'data' / 'seed_set.csv', usecols=['question_id', 'split'])
    train = logged.merge(seed, on='question_id')
    train = train[train['split'] == 'train'].copy()    # test rows are never used to tune anything
    train['behaviour'] = train['answer_text'].map(answer_behaviour)

    wanted = [
        ('natural, right chunk retrieved, answered', (train.induction_mode == 'natural') & (train.retrieval_hit == 1) & (train.behaviour == 'answered')),
        ('distractor, answered (planted clause)',    (train.induction_mode == 'distractor') & (train.behaviour == 'answered')),
        ('unanswerable, answered anyway',            (train.induction_mode == 'unanswerable') & (train.behaviour == 'answered')),
        ('refused, then guessed',                    train.behaviour == 'refused_then_guessed'),
        ('clean refusal',                            train.behaviour == 'clean_refusal'),
    ]
    picks = []
    for label, mask in wanted:
        row = train[mask].sort_values('question_id').iloc[0]    # sorted: the same pick every time
        print(f"  sample: {label:<42} {row.query_id[:8]}")
        picks.append(row.query_id)
    return picks


if __name__ == '__main__':
    from l1_signals import latest_seed_run
    run_id = latest_seed_run()
    if len(sys.argv) == 3 and sys.argv[1] == '--sample':
        query_ids = pick_sample(run_id)[:int(sys.argv[2])]
    elif len(sys.argv) == 2 and sys.argv[1] == '--all':
        connection = sqlite3.connect(query_logger.DB_PATH)
        query_ids = [row[0] for row in connection.execute(
            "SELECT query_id FROM query_log WHERE run_id = ? AND error IS NULL", (run_id,))]
        connection.close()
    else:
        sys.exit("Usage: python src/judge.py --sample N   or   python src/judge.py --all")
    print(f"Judging {len(query_ids)} answers from {run_id} with {JUDGE_MODEL} ({PROMPT_VERSION})")
    judge_rows(query_ids)
