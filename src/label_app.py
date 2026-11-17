import json
import os
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

# T20: the blind labelling page. Run with:
#   .venv/bin/streamlit run src/label_app.py
#
# BLIND on purpose (decided 2026-10-09): the page never shows the judge's verdict, L1's
# verdict, the failure mode or the train/test split. Seeing any of them would nudge the label,
# and the judge-vs-human agreement measured in T22 would come out inflated.

BASE_DIR = Path(__file__).resolve().parent.parent
# LABEL_DB_PATH lets a test run point at a throwaway copy instead of the real database.
DB_PATH = Path(os.environ.get('LABEL_DB_PATH', BASE_DIR / 'data' / 'query_log.db'))
SEED_CSV = BASE_DIR / 'data' / 'seed_set.csv'
GUIDE_PATH = BASE_DIR / 'src' / 'label_guide.md'
SAMPLE_CSV = BASE_DIR / 'data' / 'label_sample.csv'
SEED_RUN_ID = 'run-20261007T150409Z-360c'    # same as query_logger.SEED_RUN_ID
BATCH_SIZE = 20
SHUFFLE_SEED = 7    # fixed, so the order is the same every time the page opens

st.set_page_config(page_title="Label answers", layout="wide")


@st.cache_data
def load_rows(sample_version):
    """The sampled answers of the seed-set run, in a fixed shuffled order.
    sample_version (the sample file's modified time) makes the cache reload when the sample changes."""
    connection = sqlite3.connect(DB_PATH)
    run_id = SEED_RUN_ID
    # Only the columns the labeller may see are selected: nothing here reveals a verdict or a mode.
    rows = pd.read_sql_query(
        "SELECT query_id, question_id, query_text, source_contract_id, retrieved_text, answer_text "
        "FROM query_log WHERE run_id = ? AND error IS NULL", connection, params=(run_id,))
    connection.close()
    seed = pd.read_csv(SEED_CSV, usecols=['question_id', 'gold_spans'])
    rows = rows.merge(seed, on='question_id', how='left')
    # Only the 60 answers picked by src/pick_label_sample.py are labelled (decided 2026-11-14).
    rows = rows[rows['query_id'].isin(pd.read_csv(SAMPLE_CSV)['query_id'])]
    order = sorted(rows['query_id'])
    random.Random(SHUFFLE_SEED).shuffle(order)    # mixes modes and contracts together
    return rows.set_index('query_id').loc[order].reset_index(), run_id


def saved_labels():
    """query_id -> (label, notes) for every answer labelled so far. Read fresh each time."""
    connection = sqlite3.connect(DB_PATH)
    labels = {query_id: (label, notes) for query_id, label, notes in connection.execute(
        "SELECT query_id, human_label, human_label_notes FROM query_log WHERE human_label IS NOT NULL")}
    connection.close()
    return labels


def save_label(query_id, label, correct_for_contract, unhelpful, not_sure, comment):
    notes = json.dumps({'correct_for_contract': correct_for_contract, 'unhelpful': unhelpful,
                        'not_sure': not_sure, 'comment': comment.strip()})
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        "UPDATE query_log SET human_label = ?, human_label_notes = ?, human_labeled_at = ? WHERE query_id = ?",
        (label, notes, datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), query_id))
    connection.commit()
    connection.close()


rows, run_id = load_rows(SAMPLE_CSV.stat().st_mtime)
labels = saved_labels()
done = sum(query_id in labels for query_id in rows['query_id'])

# Start at the first unlabelled answer; the Back/Next buttons move from there.
if 'position' not in st.session_state:
    unlabelled = [i for i, query_id in enumerate(rows['query_id']) if query_id not in labels]
    st.session_state.position = unlabelled[0] if unlabelled else 0
position = st.session_state.position
row = rows.iloc[position]
batch = position // BATCH_SIZE + 1

# --- Sidebar: progress and the rubric -----------------------------------------------------
with st.sidebar:
    st.header("Progress")
    st.progress(done / len(rows), text=f"{done} of {len(rows)} labelled")
    st.write(f"Answer **{position + 1}** of {len(rows)} · batch **{batch}** of {-(-len(rows) // BATCH_SIZE)}")
    if done and done % BATCH_SIZE == 0 and row['query_id'] not in labels:
        st.success(f"Batch finished. A good moment for a break.")
    st.header("Rubric")
    st.markdown("""
**Question 1 decides the label. Judge it against the EXCERPTS only.**

- **Not supported** if any claim (fact, number, date, party, condition, or a yes/no
  conclusion) isn't in the excerpts or contradicts them.
- A conclusion that **misstates** the excerpts is *not supported*, even if its details are
  copied. ("No assignment except on a merger" → "Yes, you can assign" = not supported.)
- Refused, then guessed anyway → judge **the guess**.
- A plain refusal ("not in the text") makes no claim → **supported**. If the excerpts *did*
  contain the answer, also tick **unhelpful**.
- Ignore section or excerpt numbers used as citations.

**Question 2 is optional and separate.** It uses the real clause from CUAD and never changes question 1.
""")

# --- Main: the answer to label ------------------------------------------------------------
st.title("Label one answer")
# General word meanings only, written from the clause types and never from the answers.
with st.expander("Plain-English guide to contract wording"):
    st.markdown(GUIDE_PATH.read_text())
st.caption(f"Contract: {row['source_contract_id'].split('_')[0]}  ·  {row['source_contract_id'][:90]}")
st.subheader("Question")
st.write(row['query_text'])
st.subheader("Answer")
st.info(row['answer_text'])

st.subheader("Excerpts the system was given")
for number, excerpt in enumerate(row['retrieved_text'].split("\n\n---\n\n"), start=1):
    with st.expander(f"Excerpt {number}", expanded=True):
        st.text(" ".join(excerpt.split()))    # whitespace collapsed for reading; meaning unchanged

with st.expander("Real clause from CUAD (for question 2 only)"):
    spans = json.loads(row['gold_spans']) if isinstance(row['gold_spans'], str) else []
    if spans:
        for span in spans:
            st.text(" ".join(span.split()))
    else:
        st.write("CUAD's lawyers found **no such clause** in this contract.")

# Pre-fill only with YOUR earlier answer for this row (when going back); never a default.
previous_label, previous_notes = labels.get(row['query_id'], (None, None))
previous_notes = json.loads(previous_notes) if previous_notes else {}
LABEL_OPTIONS = ["Supported: every claim is in the excerpts", "Not supported: at least one claim isn't"]
CONTRACT_OPTIONS = ["yes", "no", "can't tell"]

st.subheader("Your label")
label_choice = st.radio("1. Is every claim in the answer supported by the excerpts?", LABEL_OPTIONS,
                        index=None if previous_label is None else int(previous_label), key=f"label_{position}")
not_sure = st.checkbox("Not sure: the contract wording was too hard to be confident about question 1",
                       value=previous_notes.get('not_sure', False), key=f"not_sure_{position}")
contract_choice = st.radio("2. (Optional) Is the answer correct for the real contract?", CONTRACT_OPTIONS,
                           index=CONTRACT_OPTIONS.index(previous_notes['correct_for_contract'])
                           if previous_notes.get('correct_for_contract') in CONTRACT_OPTIONS else None,
                           horizontal=True, key=f"contract_{position}")
unhelpful = st.checkbox("Unhelpful: it refused, but the excerpts did contain the answer",
                        value=previous_notes.get('unhelpful', False), key=f"unhelpful_{position}")
comment = st.text_input("Note (optional, e.g. why it's not supported)",
                        value=previous_notes.get('comment', ''), key=f"comment_{position}")

back, save, skip = st.columns(3)
if back.button("← Back", disabled=position == 0):
    st.session_state.position -= 1
    st.rerun()
# Question 2 became optional on 2026-11-14 to save time; only question 1 is measured.
ready = label_choice is not None
if save.button("Save and next →", type="primary", disabled=not ready):
    save_label(row['query_id'], LABEL_OPTIONS.index(label_choice), contract_choice, unhelpful, not_sure, comment)
    st.session_state.position = min(position + 1, len(rows) - 1)
    st.rerun()
if skip.button("Skip for now", disabled=position == len(rows) - 1):
    st.session_state.position += 1
    st.rerun()
if not ready:
    st.caption("Answer question 1 to save.")
