import sys
from pathlib import Path

# Lets this script import rag.py from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rag

# Smoke test for the whole RAG pipeline (T6 Step 6). Two kinds of check:
#   1. Automatic: the plumbing works, i.e. every field is filled and sensible.
#   2. By eye: a human reads each answer next to the clause that should answer it.
# Answer QUALITY is deliberately not asserted. This is the test bench; it is supposed to
# get some answers wrong, because the evaluator needs wrong answers to catch.

CONTRACT = 'EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.txt'

# Same five questions as check_retrieval.py, with the clause that answers each.
ANSWERABLE = [
    ("Which state's law governs the agreement?",
     "This Agreement shall be interpreted and enforced in accordance with the laws of the State of California"),
    ("Does the agreement renew automatically?",
     "This agreement shall automatically renew for additional successive terms of twelve (12) months"),
    ("How is a change in control defined?",
     'For purposes of this Agreement, "Change in Control" means'),
    ("Is either party barred from challenging the other's trademarks?",
     "At no time during the term of the Agreement or thereafter shall eDiets attack, challenge or file any application with respect to any Women.com Mark."),
    ("Can the fees be increased, and with how much notice?",
     "Following the Initial Term, Women.com shall have the right, upon no fewer than sixty (60) days prior written notice to eDiets, to increase"),
]

# Facts a co-branding agreement has no reason to contain.
UNANSWERABLE = [
    "Who is the CEO of Women.com?",
    "How many employees does Women.com have?",
    "What is the eDiets office address?",
]

# Columns answer() must always fill (they are NOT NULL in schema.sql, minus the ones T7 adds).
REQUIRED_FIELDS = [
    'pipeline_config_id', 'created_at', 'query_text', 'embedding_model', 'retrieval_k',
    'retrieved_chunk_ids', 'retrieved_scores', 'retrieved_text', 'generator_model', 'answer_text',
]


def check_plumbing(result):
    """Fail loudly if any part of the pipeline returned something empty or malformed."""
    for field in REQUIRED_FIELDS:
        assert result.get(field) not in (None, '', []), f"{field} is empty"
    assert len(result['retrieved_chunk_ids']) == len(result['retrieved_scores']) == result['retrieval_k'], \
        "chunk ids, scores and k don't line up"
    assert result['prompt_tokens'] > 0 and result['completion_tokens'] > 0, "Ollama returned no token counts"


exact_refusals = 0

print("=== Answerable: compare each answer with the clause ===\n")
for question, gold_clause in ANSWERABLE:
    result = rag.answer(question, CONTRACT)
    check_plumbing(result)
    print(f"Q:      {question}")
    print(f"Clause: {gold_clause}")
    print(f"Answer: {result['answer_text']}")
    if result['answer_text'].startswith(rag.NOT_FOUND_ANSWER):
        print("        ^ refused a question the contract CAN answer")
    print()

print("=== Unanswerable: should get the exact refusal sentence ===\n")
for question in UNANSWERABLE:
    result = rag.answer(question, CONTRACT)
    check_plumbing(result)
    is_exact = result['answer_text'] == rag.NOT_FOUND_ANSWER
    exact_refusals += is_exact
    print(f"Q:      {question}")
    print(f"Answer: {result['answer_text']}")
    print(f"        exact refusal: {is_exact}\n")

print(f"Plumbing checks passed for all {len(ANSWERABLE) + len(UNANSWERABLE)} questions.")
print(f"Exact refusals on unanswerable questions: {exact_refusals}/{len(UNANSWERABLE)}")
