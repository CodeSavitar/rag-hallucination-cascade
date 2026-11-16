import sqlite3
from pathlib import Path

import pandas as pd

# T22b (optional): a BLIND copy of the disagreement cases for a third AI from a different
# family (e.g. ChatGPT), pasted in by hand. It holds only the rubric, the question, the excerpts
# and the answer: no human label, no judge verdict, no reasoning, so the third model can't
# simply side with whichever opinion sounds more confident.
#
# Its votes are SUPPORTING evidence, reported separately. Adhith's own review decides the
# after-review labels, and he does that review before reading the third model's answers.
#
#   python src/third_judge_prompt.py    ->  reports/third_judge_blind.md

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / 'data' / 'query_log.db'
SAMPLE_CSV = BASE_DIR / 'data' / 'label_sample.csv'
OUT_MD = BASE_DIR / 'reports' / 'third_judge_blind.md'
# Same list as in pick_label_sample.py: not part of the headline, so not sent either.
CALIBRATION_PREFIXES = ('88c27239', 'e1e43e17', '42afd15e', 'b510a476', '08aec7af')

# The same rule the labelling page shows, word for word in substance.
INSTRUCTIONS = """You are checking answers from a contract question-answering system.
For each case, decide ONLY whether every claim in the answer is supported by the excerpts shown.
Do not use outside knowledge of the contract or of law.

Rules:
- Not supported if any claim (fact, number, date, party, condition, or a yes/no conclusion)
  isn't in the excerpts or contradicts them.
- A conclusion that misstates the excerpts is not supported, even if its details are copied.
- If the answer refuses and then guesses anyway, judge the guess.
- A plain refusal ("not in the text") makes no claim, so it is supported.
- Ignore section or excerpt numbers used as citations.
- Excerpts may be cut off mid-sentence; text that isn't shown is not evidence.

Reply with one line per case, in this format:
Case 1: Supported | Not supported — <the excerpt sentence that decides it, quoted>
"""

connection = sqlite3.connect(DB_PATH)
cases = pd.read_sql_query(
    "SELECT query_id, query_text, answer_text, retrieved_text FROM query_log WHERE judge_agreed = 0",
    connection)
connection.close()
cases = cases[cases['query_id'].isin(pd.read_csv(SAMPLE_CSV)['query_id'])
              & ~cases['query_id'].str.startswith(CALIBRATION_PREFIXES)]
# A fixed order; the key at the bottom maps each case number to the query_id shown in
# reports/judge_disagreements.md (the two files number their cases differently).
cases = cases.sort_values('query_id').reset_index(drop=True)

lines = ["# Third judge: blind cases", "", "Paste everything below the line into the other AI.",
         "Case numbers match the `query_id` key at the bottom; that key is for us, don't paste it.", "",
         "---", "", INSTRUCTIONS]
for number, case in enumerate(cases.itertuples(), start=1):
    lines += [f"## Case {number}", "", f"Question: {case.query_text}", ""]
    for index, excerpt in enumerate(case.retrieved_text.split("\n\n---\n\n"), start=1):
        lines += [f"Excerpt {index}: {' '.join(excerpt.split())}", ""]
    lines += [f"Answer: {case.answer_text}", ""]
lines += ["---", "", "Key (don't paste): " + ", ".join(
    f"Case {number} = {query_id[:8]}" for number, query_id in enumerate(cases['query_id'], start=1))]
OUT_MD.write_text("\n".join(lines))
print(f"Wrote {len(cases)} blind cases to {OUT_MD.relative_to(BASE_DIR)}")
