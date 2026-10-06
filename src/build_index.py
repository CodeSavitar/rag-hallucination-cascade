import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from pathlib import Path

# ORDER MATTERS: torch must be imported before faiss. Both bundle their own OpenMP
# library (libomp.dylib); with faiss first, loading the model segfaults on macOS.
import torch
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
final_csv_path = BASE_DIR / 'data' / 'selected_contracts.csv'

CHUNK_SIZE = 1000
OVERLAP_SIZE = 200
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-0.6B'
CONTRACTS_INPUT_PATH_FOLDER = BASE_DIR / 'CUAD_v1' / 'full_contract_txt'
# Built from the settings above, so it can never drift out of sync with them.
# e.g. 'qwen3-embedding-0.6b_c1000_o200'
PIPELINE_CONFIG_ID = f"{EMBEDDING_MODEL.split('/')[-1].lower()}_c{CHUNK_SIZE}_o{OVERLAP_SIZE}"

INDEX_FOLDER_PATH = BASE_DIR / 'index'
FAISS_INDEX_PATH = INDEX_FOLDER_PATH / 'chunks.faiss'    # the vectors
CHUNKS_CSV_PATH = INDEX_FOLDER_PATH / 'chunks.csv'       # row n = vector n: contract, offsets, text
CONFIG_JSON_PATH = INDEX_FOLDER_PATH / 'config.json'     # the settings that built the two files above

# Validate settings before touching the disk
if CHUNK_SIZE <= OVERLAP_SIZE:
    raise ValueError("CHUNK_SIZE must be greater than OVERLAP_SIZE")

# index/ is gitignored, so a fresh clone won't have it. exist_ok=True makes re-runs safe.
INDEX_FOLDER_PATH.mkdir(parents=True, exist_ok=True)


def chunk_text(text):
    """Split one contract into overlapping fixed-size windows.

    Returns a list of dicts: {"text", "start_char", "end_char"}.
    The offsets point into the raw text, so text[start_char:end_char] == chunk["text"].
    """
    step = CHUNK_SIZE - OVERLAP_SIZE    # how far each window moves forward (800)
    chunks = []
    start = 0

    while True:
        end = min(start + CHUNK_SIZE, len(text))    # the last window may be shorter
        chunks.append({
            "text": text[start:end],
            "start_char": start,
            "end_char": end,
        })

        # Stop once a window reaches the end of the text. Without this check the
        # loop would add one more chunk made only of overlap.
        if end == len(text):
            break

        start += step

    return chunks


df = pd.read_csv(final_csv_path, encoding="utf-8")

textfiles_dict = {}

for i, row in df.iterrows():
    textfile_path = CONTRACTS_INPUT_PATH_FOLDER / row['Filename']
    with open(textfile_path, "r", encoding="utf-8") as file:
        text = file.read()
    
    textfiles_dict[row['Filename']] = text


# One row per chunk. Row order matters: row n of this table becomes vector n in the
# FAISS index, so this table is how a search result is traced back to its contract.
rows = []

for filename, text in textfiles_dict.items():
    for chunk_index, chunk in enumerate(chunk_text(text)):
        rows.append({
            # Readable and stable across rebuilds, e.g. 'EdietsComInc_..._Agreement::007'
            "chunk_id": f"{Path(filename).stem}::{chunk_index:03d}",
            "filename": filename,
            "chunk_index": chunk_index,
            "start_char": chunk["start_char"],
            "end_char": chunk["end_char"],
            "text": chunk["text"],
        })

chunks_df = pd.DataFrame(rows)

# Checks that can fail: every contract produced chunks, and no chunk_id repeats
assert chunks_df["filename"].nunique() == len(textfiles_dict), "Some contract produced no chunks"
assert chunks_df["chunk_id"].is_unique, "Duplicate chunk_id found"

print(f"{len(chunks_df)} chunks from {chunks_df['filename'].nunique()} contracts")


# Embed every chunk. The first run downloads the model (~1.2 GB) into the Hugging Face cache.
# Load in float32. The model's default is bfloat16 (~3 significant digits), which left
# "normalised" vectors with lengths of 0.998-1.004 and cosine scores above 1.0.
# Queries in T6 must be embedded with this same setting.
model = SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"dtype": torch.float32})

# Chunks are encoded WITHOUT a prompt. Qwen expects only questions to carry the
# "query" prompt (that happens at query time, in T6).
# normalize_embeddings=True scales every vector to length 1, so the inner product
# FAISS computes in Step 6 is exactly cosine similarity.
embeddings = model.encode(
    chunks_df["text"].tolist(),    # same order as the table: row n -> vector n
    batch_size=16,
    normalize_embeddings=True,
    show_progress_bar=True,
)

# Checks that can fail: one vector per row, and every vector has length 1
assert embeddings.shape[0] == len(chunks_df), "Number of vectors doesn't match number of chunks"
assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-4), "Vectors are not normalised"

print(f"Embeddings: {embeddings.shape[0]} vectors x {embeddings.shape[1]} dimensions")


# Build the index. IndexFlatIP scores by inner product; on length-1 vectors that is
# cosine similarity (higher = closer). "Flat" = exact search over every vector, which
# is instant at ~1,700 vectors, so no approximate index is needed.
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)    # vector n gets position n, matching row n of chunks_df

# Final checks, all BEFORE anything is written, so a bad build never reaches the disk
assert index.ntotal == len(chunks_df), "Index size doesn't match the chunk table"
assert chunks_df["filename"].nunique() == len(df), "Not every selected contract is in the index"

# A vector's closest match must be itself, with a cosine score of 1.0
scores, positions = index.search(embeddings[:1], 1)
assert positions[0][0] == 0 and abs(scores[0][0] - 1.0) < 1e-4, "Index search is broken"

config = {
    "pipeline_config_id": PIPELINE_CONFIG_ID,
    "embedding_model": EMBEDDING_MODEL,
    "embedding_dtype": "float32",
    "normalized": True,
    "similarity": "cosine (IndexFlatIP on normalised vectors)",
    "dimensions": embeddings.shape[1],
    "chunk_size": CHUNK_SIZE,
    "overlap_size": OVERLAP_SIZE,
    "num_chunks": len(chunks_df),
    "num_contracts": chunks_df["filename"].nunique(),
    "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}

faiss.write_index(index, str(FAISS_INDEX_PATH))    # faiss needs a plain string path
chunks_df.to_csv(CHUNKS_CSV_PATH, index=False)
with open(CONFIG_JSON_PATH, "w", encoding="utf-8") as file:
    json.dump(config, file, indent=2)

print(f"Wrote {FAISS_INDEX_PATH.name}, {CHUNKS_CSV_PATH.name}, {CONFIG_JSON_PATH.name} to {INDEX_FOLDER_PATH}")

