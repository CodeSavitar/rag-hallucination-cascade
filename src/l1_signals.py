import json
import re
import sqlite3
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# Lets this file import from the same folder, wherever it's run from.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import query_logger

# L1: cheap checks on each logged answer (T14), and the pass / fail / uncertain verdict
# built from them (T15). Each check becomes one l1_* column, and in M4 the same numbers
# become the classifier's features.
#
#   python src/l1_signals.py            score the latest full seed-set run
#   python src/l1_signals.py RUN_ID     score a specific run (re-running overwrites)

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FOLDER_PATH = BASE_DIR / 'index'

# --- Refusal behaviour (moved here from run_seed_set.py) ---------------------------------
NOT_FOUND_ANSWER = "The answer is not in the provided contract text."    # same sentence as rag.py
REFUSAL_CORE = NOT_FOUND_ANSWER.rstrip('.').lower()    # without the full stop: Mistral sometimes adds words after "text"
# Words that turn a refusal into a guess: "...not in the text. However, it can be inferred..."
GUESS_MARKERS = re.compile(r"\b(however|but|based on|infer|inferred|appears|likely|suggests|implies)\b", re.IGNORECASE)
# Refusals Mistral writes in its own words instead of the fixed sentence (seen in T6 and T8).
PARAPHRASED_REFUSAL = re.compile(
    r"\b(does not|doesn't|do not) (specify|mention|provide|state|contain|include|say)\b"
    r"|\bnot (specified|mentioned|provided|stated|included|found)\b", re.IGNORECASE)


MAX_LEAD_IN_BEFORE_REFUSAL = 20    # room for "The answer is: " in front of the refusal sentence


def answer_behaviour(answer):
    """clean_refusal / refused_then_guessed / paraphrased_refusal / answered."""
    refusal_at = answer.lower().find(REFUSAL_CORE)
    if refusal_at > MAX_LEAD_IN_BEFORE_REFUSAL:
        # A full answer with the refusal sentence tacked on the END ("...(4) bodily injury.
        # The answer is not in the provided contract text if the question...") is an answer,
        # not a refusal. Found after T19: 3 such answers had been passed as "clean refusals".
        return 'answered'
    if refusal_at != -1:
        # Whatever comes after the refusal sentence decides it. "...text regarding the minimum
        # commitment." just restates the topic: still a clean refusal. A guess marker, or a
        # long continuation, means it refused and then answered anyway.
        leftover = answer.lower().split(REFUSAL_CORE, 1)[1]
        guessed = GUESS_MARKERS.search(leftover) or len(leftover) > 150
        return 'refused_then_guessed' if guessed else 'clean_refusal'
    if PARAPHRASED_REFUSAL.search(answer):
        return 'paraphrased_refusal'
    return 'answered'


# --- Word overlap --------------------------------------------------------------------------
# Common words carry no evidence: "the agreement shall" overlaps with every contract.
STOPWORDS = set("""a an and are as at be been by can could do does for from had has have if in into
is it its may might must no not of on or other our shall should so such than that the their them
then there these they this those to under upon was we were what when where which who will with
would you your any all each either neither party parties agreement""".split())


def content_words(text):
    """Lower-cased words of 3+ letters, minus stopwords."""
    return [word for word in re.findall(r"[a-z]{3,}", text.lower()) if word not in STOPWORDS]


def token_overlap(answer, context):
    """Share of the answer's content words that also appear in the context (0..1)."""
    answer_words = content_words(answer)
    if not answer_words:
        return None    # e.g. a bare refusal: nothing to check
    context_words = set(content_words(context))
    return sum(word in context_words for word in answer_words) / len(answer_words)


# --- Numbers and dates ---------------------------------------------------------------------
UNITS = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
TEENS = {'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
         'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19}
TENS = {'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
SPELLED_NUMBER = re.compile(r"\b(?:(%s)(?:[- ](%s))?|(%s)|(%s))\b" % (
    '|'.join(TENS), '|'.join(UNITS), '|'.join(TEENS), '|'.join(UNITS)), re.IGNORECASE)

# Section references are citations, not claims: "Section 7(b)", "Clauses 1 and 10", "§ 4.2".
# T14 found 4 of 6 "unsupported numbers" were these, so they're removed before checking.
SECTION_REFERENCE = re.compile(
    r"\b(?:sections?|clauses?|articles?|exhibits?|schedules?|paragraphs?|subsections?)\s+"
    r"[\dA-Za-z().]+(?:\s*(?:,|and|or|&)\s*[\dA-Za-z().]+)*|§+\s*[\d.()a-z]+"
    r"|\(\s*\d+(?:\.\d+)+\s+[A-Z][^)]*\)",    # "(9.9 Independent Contracting Parties)"
    re.IGNORECASE)


def numbers_in(text):
    """Every number in the text, normalised: '$4,950.00' -> '4950', '20%' -> '20', 'sixty' -> '60'.

    Words count too: contracts often write "sixty days" with no digits, and a check on digits
    alone would flag a correct "60 days" answer (seen in T14).
    """
    numbers = set()
    for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text):
        number = raw.replace(',', '').rstrip('.')
        if number.endswith('.00'):
            number = number[:-3]
        numbers.add(number)
    for tens, unit, teen, single in SPELLED_NUMBER.findall(text):
        if tens:
            numbers.add(str(TENS[tens.lower()] + (UNITS[unit.lower()] if unit else 0)))
        else:
            numbers.add(str(TEENS[teen.lower()] if teen else UNITS[single.lower()]))
    return numbers


def unsupported_numbers(answer, context):
    """Numbers (amounts, days, percentages, dates) in the answer that never appear in the context.

    Section references are skipped: they are citations, not facts about the contract.
    """
    if REFUSAL_CORE in answer.lower():
        answer = answer.lower().split(REFUSAL_CORE, 1)[1]    # only what comes after a refusal counts
    answer = SECTION_REFERENCE.sub(' ', answer)
    return sorted(numbers_in(answer) - numbers_in(context))


# --- Hedging, PII, toxicity ----------------------------------------------------------------
HEDGES = re.compile(r"\b(it appears|appears to|it seems|seems to|likely|possibly|probably|may be|might|"
                    r"generally|typically|presumably|it can be inferred|can be inferred|suggests|implies|"
                    r"not explicitly|unclear)\b", re.IGNORECASE)
PII = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"                 # email address
                 r"|\b\d{3}-\d{2}-\d{4}\b"                   # US social security number
                 r"|\(?\b\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b")    # US phone number
TOXIC = re.compile(r"\b(idiot|stupid|damn|hell|crap|hate you)\b", re.IGNORECASE)    # deliberately minimal: L0/L1 safety rules are a checkbox, not the project


# --- Weakest sentence ----------------------------------------------------------------------
def split_sentences(text):
    """Split on . ! ? followed by a space. '11.1' and '$4.95' don't split (no space after the dot)."""
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if len(sentence.strip()) > 3]


@lru_cache(maxsize=1)
def resources():
    """Load the embedding model and index once, on first use. Mistral is never loaded."""
    import torch                                       # torch BEFORE faiss (see build_index.py)
    from sentence_transformers import SentenceTransformer
    import faiss

    with open(INDEX_FOLDER_PATH / 'config.json', encoding='utf-8') as file:
        config = json.load(file)
    model = SentenceTransformer(config['embedding_model'],
                                model_kwargs={"dtype": getattr(torch, config['embedding_dtype'])})
    model.encode(["warm-up"])    # keep the start-up cost out of the first row's latency
    index = faiss.read_index(str(INDEX_FOLDER_PATH / 'chunks.faiss'))
    chunk_ids = pd.read_csv(INDEX_FOLDER_PATH / 'chunks.csv', usecols=['chunk_id'])['chunk_id']
    positions = {chunk_id: position for position, chunk_id in enumerate(chunk_ids)}
    return model, index, positions


def min_sentence_similarity(answer, retrieved_chunk_ids):
    """For each answer sentence, its best cosine similarity to any retrieved chunk; return the LOWEST.

    One well-supported sentence can hide one invented one. Scoring the weakest sentence, not
    the average, is what lets a single unsupported claim show up. Chunk vectors come straight
    from the index (already computed in T5); only the answer's sentences are embedded here.
    Both sides are embedded as documents (no "query" prompt): this compares text with text.
    """
    sentences = split_sentences(answer)
    if not sentences or not retrieved_chunk_ids:
        return None
    model, index, positions = resources()
    sentence_vectors = model.encode(sentences, normalize_embeddings=True)
    chunk_vectors = np.stack([index.reconstruct(positions[chunk_id]) for chunk_id in retrieved_chunk_ids])
    best_per_sentence = (sentence_vectors @ chunk_vectors.T).max(axis=1)    # cosine: both sides have length 1
    return float(best_per_sentence.min())


# --- Verdict (T15) -------------------------------------------------------------------------
# PROVISIONAL cut-offs, set 2026-10-08 from the 80 TRAIN answers only, using a stand-in for
# labels: "probably good" = natural question, right chunk retrieved, answered (20 rows);
# "probably bad" = answered although the excerpts couldn't hold the answer (24 rows).
# Revisit once M3's human labels exist. Never tuned on the 40 test rows.
#
# Why L1 never PASSES a real answer: the two groups overlap heavily (good word overlap goes as
# low as 0.67, bad as high as 0.97). The strictest pass rule that still passed anything good
# (overlap >= 0.95 and weakest sentence >= 0.45) passed 5 of 20 good answers AND one planted-
# clause answer, the failure the cascade most needs to catch. A wrongly passed hallucination
# is the expensive error, so every real answer goes on to L2/L3.
FAIL_IF_OVERLAP_BELOW = 0.60          # lowest "probably good" value seen: 0.67
FAIL_IF_WEAKEST_SENTENCE_BELOW = 0.30  # lowest "probably good" value seen: 0.34


def l1_verdict(signals):
    """'fail' on hard evidence, 'pass' only for answers that make no claim, else 'uncertain'."""
    behaviour = signals['l1_features_json']['answer_behaviour']
    overlap = signals['l1_token_overlap']
    weakest = signals['l1_min_sentence_similarity']

    if behaviour == 'refused_then_guessed':
        return 'fail'                     # said "not in the text", then answered anyway
    if signals['l1_unsupported_numerics'] > 0:
        return 'fail'                     # a number the excerpts never mention
    if behaviour == 'clean_refusal':
        return 'pass'                     # the fixed refusal sentence: no claim to be wrong about
    # The overlap rules apply to real answers ONLY. A refusal, fixed or in the model's own
    # words, shares few words with the excerpts by nature; the first version of these rules
    # failed 6 correct refusals for "low overlap" (found in T15).
    if behaviour == 'answered':
        if overlap is not None and overlap < FAIL_IF_OVERLAP_BELOW:
            return 'fail'
        if weakest is not None and weakest < FAIL_IF_WEAKEST_SENTENCE_BELOW:
            return 'fail'
    # Paraphrased refusals ("the contract does not specify...") can carry a claim with them,
    # e.g. "...does not specify a cap, but liability is unlimited", so they're not trusted here.
    return 'uncertain'


# --- All signals for one row ---------------------------------------------------------------
def compute_signals(answer, context, retrieved_chunk_ids, retrieved_scores):
    """Every L1 signal for one answer, keyed by query_log column name."""
    model, _, _ = resources()          # loaded once, BEFORE the timer: loading is not checking time
    start = time.perf_counter()

    behaviour = answer_behaviour(answer)
    hedges = [match.group(0).lower() for match in HEDGES.finditer(answer)]
    unsupported = unsupported_numbers(answer, context)
    answer_tokens = len(model.tokenizer.encode(answer, add_special_tokens=False))    # real tokens, same tokenizer as the embeddings
    context_tokens = len(model.tokenizer.encode(context, add_special_tokens=False))

    signals = {
        'l1_token_overlap': token_overlap(answer, context),
        'l1_max_similarity': max(retrieved_scores),
        'l1_mean_similarity': float(np.mean(retrieved_scores)),
        'l1_score_spread': max(retrieved_scores) - min(retrieved_scores),
        'l1_min_sentence_similarity': min_sentence_similarity(answer, retrieved_chunk_ids),
        'l1_answer_len_tokens': answer_tokens,
        'l1_context_len_tokens': context_tokens,
        'l1_len_ratio': answer_tokens / context_tokens if context_tokens else None,
        'l1_hedging_count': len(hedges),
        'l1_unsupported_numerics': len(unsupported),
        'l1_pii_hits': len(PII.findall(answer)),
        'l1_toxicity_hits': len(TOXIC.findall(answer)),
        # Extra signals with no column of their own, plus the details behind the counts,
        # so any number above can be checked by reading this field.
        'l1_features_json': {
            'answer_behaviour': behaviour,
            'n_sentences': len(split_sentences(answer)),
            'hedges_found': hedges,
            'unsupported_numbers_found': unsupported,
        },
        'cost_l1_usd': 0.0,    # runs locally: $0, recorded as 0 not NULL (NULL = "didn't run")
    }
    signals['l1_verdict'] = l1_verdict(signals)
    signals['latency_l1_ms'] = (time.perf_counter() - start) * 1000
    return signals


def score_run(run_id):
    """Compute L1 signals for every successful row of a run and write them to query_log."""
    connection = sqlite3.connect(query_logger.DB_PATH)
    rows = connection.execute(
        "SELECT query_id, answer_text, retrieved_text, retrieved_chunk_ids, retrieved_scores "
        "FROM query_log WHERE run_id = ? AND error IS NULL", (run_id,)).fetchall()
    assert rows, f"No successful rows for run {run_id}"

    for query_id, answer, context, chunk_ids_json, scores_json in rows:
        signals = compute_signals(answer, context, json.loads(chunk_ids_json), json.loads(scores_json))
        signals['l1_features_json'] = json.dumps(signals['l1_features_json'])
        assignments = ', '.join(f"{column} = :{column}" for column in signals)
        connection.execute(f"UPDATE query_log SET {assignments} WHERE query_id = :query_id",
                           {**signals, 'query_id': query_id})
    connection.commit()
    connection.close()
    return len(rows)


def latest_seed_run():
    """The seed-set run (pinned in query_logger.py; no longer "the most recent big run")."""
    return query_logger.SEED_RUN_ID


if __name__ == '__main__':
    run_id = sys.argv[1] if len(sys.argv) > 1 else latest_seed_run()
    count = score_run(run_id)
    print(f"L1 signals written for {count} rows of {run_id}")
