import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import ollama
import pandas as pd

# ORDER MATTERS: torch must be imported before faiss (see build_index.py).
import torch
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FOLDER_PATH = BASE_DIR / 'index'

# Generation settings, decided 2026-10-06 (see STATUS.md).
GENERATOR_MODEL = 'mistral:latest'
# weak_generator mode: the SAME model (Mistral 7B v0.3) compressed to 2-bit instead of 4-bit,
# so any change in hallucination rate comes from model quality alone.
WEAK_GENERATOR_MODEL = 'mistral:7b-instruct-v0.3-q2_K'
TOP_K = 5              # chunks sent to the generator
NUM_CTX = 8192         # Mistral's context window, set explicitly instead of trusting Ollama's default
TEMPERATURE = 0        # same question -> same answer, so runs are reproducible
PROMPT_VERSION = 'rag-v1'
KEEP_ALIVE = '30m'     # keep Mistral loaded between questions (Ollama unloads after 5 min idle)

# Sent with every call, and logged as generation_params. Warm-up must use the SAME options:
# a different num_ctx makes Ollama reload the model on the first real question.
GENERATION_OPTIONS = {'temperature': TEMPERATURE, 'num_ctx': NUM_CTX}

# The exact sentence the generator must use when the context doesn't hold the answer.
# A fixed phrase makes refusals detectable with a simple string match later (M2/L1).
NOT_FOUND_ANSWER = "The answer is not in the provided contract text."

# rag-v1. Changing this wording means bumping PROMPT_VERSION, or old and new answers
# in query_log can't be told apart.
# The rules come AFTER the excerpts and question on purpose: tested on 5 unanswerable
# questions, Mistral used the exact refusal sentence 4/5 times this way vs 1/3 with the
# rules at the top (small models follow the instructions nearest the answer best).
RAG_PROMPT_TEMPLATE = """You are answering a question about a legal contract, using ONLY the excerpts below.

Contract excerpts:
{context}

Question: {question}

Rules:
- Use only the excerpts above. Do not use outside knowledge or guess.
- If the excerpts do not contain the answer, reply with exactly this sentence and nothing else: {not_found}
- Otherwise, answer in one to three sentences.

Answer:"""


# Load everything ONCE, when this file is imported. Each question then reuses it, instead
# of reloading a 1.2 GB model per question.
with open(INDEX_FOLDER_PATH / 'config.json', encoding='utf-8') as file:
    CONFIG = json.load(file)

INDEX = faiss.read_index(str(INDEX_FOLDER_PATH / 'chunks.faiss'))
CHUNKS_DF = pd.read_csv(INDEX_FOLDER_PATH / 'chunks.csv')

# Queries must be embedded exactly like the chunks were, so the model name and precision
# come from config.json rather than being typed again here.
EMBEDDING_MODEL = SentenceTransformer(
    CONFIG['embedding_model'],
    model_kwargs={"dtype": getattr(torch, CONFIG['embedding_dtype'])},
)

# Warm-up: the first encode() pays a one-off start-up cost (~190 ms vs ~30 ms after).
# Doing it here keeps that cost out of the first question's logged latency.
EMBEDDING_MODEL.encode(["warm-up"], prompt_name="query")

# Same for Mistral: the first call loads the model into memory, which takes seconds.
# The weak model is warmed up the first time it's needed (see generate()), so plain
# runs don't load a second model they never use.
WARMED_UP_MODELS = set()


def warm_up(model):
    if model not in WARMED_UP_MODELS:
        ollama.chat(model=model, messages=[{'role': 'user', 'content': 'Reply OK.'}],
                    options=GENERATION_OPTIONS, keep_alive=KEEP_ALIVE)
        WARMED_UP_MODELS.add(model)


warm_up(GENERATOR_MODEL)

# Checks that can fail: the three files belong together. A mismatch means the index
# was half-rebuilt, and every search result would point at the wrong chunk.
assert INDEX.ntotal == len(CHUNKS_DF) == CONFIG['num_chunks'], "Index, chunk table and config disagree on chunk count"
assert INDEX.d == CONFIG['dimensions'], "Index dimensions don't match config.json"

# chunk_id -> its row number, which is also its vector's position in the index
CHUNK_POSITIONS = {chunk_id: position for position, chunk_id in enumerate(CHUNKS_DF['chunk_id'])}


def retrieve(question, contract, k=TOP_K, exclude_chunk_ids=(), inject_chunk_ids=()):
    """Find the k chunks of one contract that best match the question.

    contract is a filename as it appears in chunks.csv. Returns a dict whose keys are
    query_log column names, so T7 can write it straight to the table.

    The two optional arguments are the sabotage switches for M2's failure modes:
      exclude_chunk_ids: never return these (starved_retrieval: hide the answer)
      inject_chunk_ids:  always include these, even from another contract (distractor)
    """
    # Only this contract's chunks are searched (decided 2026-10-06: global search across
    # 30 contracts returns other contracts' clauses).
    contract_positions = np.where(CHUNKS_DF['filename'] == contract)[0]
    assert len(contract_positions) > 0, f"Contract not in the index: {contract}"
    excluded = {CHUNK_POSITIONS[chunk_id] for chunk_id in exclude_chunk_ids}
    positions = np.array([p for p in contract_positions if p not in excluded], dtype='int64')
    n_search = k - len(inject_chunk_ids)    # injected chunks take some of the k slots
    assert n_search <= len(positions), f"k={k} but only {len(positions)} chunks are searchable"
    search_params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(positions))

    # Timed from question text to search result, so the cost of embedding the
    # question counts as retrieval time.
    start = time.perf_counter()
    query_vector = EMBEDDING_MODEL.encode([question], prompt_name="query", normalize_embeddings=True)
    scores, hits = INDEX.search(query_vector, n_search, params=search_params)
    results = list(zip(scores[0].tolist(), hits[0].tolist()))

    # Injected chunks get their REAL similarity to the question (dot product of two length-1
    # vectors = cosine), then everything is re-sorted by score. A distractor pinned to rank 1
    # with a fake score would leave a pattern in the logged scores that the evaluator could
    # learn, instead of learning to spot wrong content.
    for chunk_id in inject_chunk_ids:
        position = CHUNK_POSITIONS[chunk_id]
        score = float(query_vector[0] @ INDEX.reconstruct(position))
        results.append((score, position))
    results.sort(key=lambda pair: pair[0], reverse=True)    # best match first
    latency_ms = (time.perf_counter() - start) * 1000

    top_chunks = CHUNKS_DF.iloc[[position for _, position in results]]

    return {
        "embedding_model": CONFIG['embedding_model'],
        "retrieval_k": k,
        "retrieved_chunk_ids": top_chunks['chunk_id'].tolist(),
        "retrieved_scores": [float(score) for score, _ in results],    # all k, same order as the ids
        # The exact context the generator will see. The separator keeps chunks visibly apart.
        "retrieved_text": "\n\n---\n\n".join(top_chunks['text']),
        "latency_retrieval_ms": latency_ms,
        # Runs locally, so $0. Recorded as 0.0, not left NULL: in query_log NULL means
        # "this step didn't run", and the cost comparison in M5 must include this step.
        "cost_embedding_usd": 0.0,
    }


def build_prompt(question, context):
    """Fill the rag-v1 template with the question and the retrieved context."""
    return RAG_PROMPT_TEMPLATE.format(
        not_found=NOT_FOUND_ANSWER,
        context=context,
        question=question,
    )


def generate(prompt, model=GENERATOR_MODEL):
    """Send one prompt to Mistral. Returns a dict keyed by query_log column names."""
    warm_up(model)    # before the timer: loading a model is not answering time
    start = time.perf_counter()
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}],
        options=GENERATION_OPTIONS,
        keep_alive=KEEP_ALIVE,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "generator_model": model,
        "generation_params": GENERATION_OPTIONS,
        "prompt_version": PROMPT_VERSION,
        "answer_text": response.message.content.strip(),
        "prompt_tokens": response.prompt_eval_count,      # counted by Ollama, not estimated
        "completion_tokens": response.eval_count,
        "latency_generation_ms": latency_ms,
        "cost_generation_usd": 0.0,    # runs locally: no per-call charge
    }


INDUCTION_MODES = {'natural', 'unanswerable', 'starved_retrieval', 'distractor', 'weak_generator'}


def answer(question, contract, k=TOP_K, induction_mode='natural',
           gold_chunk_ids=(), distractor_chunk_ids=()):
    """Run one question through the whole RAG pipeline: retrieve -> prompt -> generate.

    induction_mode picks a deliberate failure (M2, decided 2026-10-07):
      natural, unanswerable  the normal pipeline (unanswerable questions fail by themselves)
      starved_retrieval      the answer's chunks are hidden; the next k are sent instead
      distractor             the answer's chunks are hidden; another contract's version of the
                             same clause takes one slot
      weak_generator         normal retrieval, but the 2-bit model writes the answer

    Returns one dict keyed by query_log column names, holding everything the RAG side
    knows. The caller (T7) adds the run-level fields: query_id, run_id, question_id,
    induction_mode, gold_chunk_ids, shadow_mode.
    """
    assert induction_mode in INDUCTION_MODES, f"Unknown induction_mode: {induction_mode}"
    if induction_mode in ('starved_retrieval', 'distractor'):
        assert gold_chunk_ids, f"{induction_mode} needs gold_chunk_ids to know what to hide"
    if induction_mode == 'distractor':
        assert distractor_chunk_ids, "distractor needs distractor_chunk_ids"

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ALL gold chunks are hidden, including ones that only touch the clause's edge, so no
    # fragment of the real answer reaches the model.
    hide = gold_chunk_ids if induction_mode in ('starved_retrieval', 'distractor') else ()
    plant = distractor_chunk_ids if induction_mode == 'distractor' else ()
    model = WEAK_GENERATOR_MODEL if induction_mode == 'weak_generator' else GENERATOR_MODEL

    retrieval = retrieve(question, contract, k, exclude_chunk_ids=hide, inject_chunk_ids=plant)
    prompt = build_prompt(question, retrieval["retrieved_text"])
    generation = generate(prompt, model)

    return {
        "pipeline_config_id": CONFIG['pipeline_config_id'],
        "created_at": created_at,
        "query_text": question,
        "source_contract_id": contract,
        **retrieval,     # ** unpacks a dict: copies all its keys and values in here
        **generation,
    }


if __name__ == '__main__':
    print(f"Loaded {INDEX.ntotal} chunks from {CHUNKS_DF['filename'].nunique()} contracts "
          f"(config {CONFIG['pipeline_config_id']})")
