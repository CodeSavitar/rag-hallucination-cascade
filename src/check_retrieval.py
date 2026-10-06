import json
from pathlib import Path

import numpy as np
import pandas as pd

# ORDER MATTERS: torch must be imported before faiss (see build_index.py).
import torch
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_FOLDER_PATH = BASE_DIR / 'index'
CONTRACTS_INPUT_PATH_FOLDER = BASE_DIR / 'CUAD_v1' / 'full_contract_txt'
TOP_K = 5

# Hand-written questions about one contract, each paired with the start of the clause
# that answers it (copied from CUAD's annotations in master_clauses.csv).
CONTRACT = 'EdietsComInc_20001030_10QSB_EX-10.4_2606646_EX-10.4_Co-Branding Agreement.txt'
TEST_CASES = [
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

# Load the three files build_index.py wrote. The model settings come from config.json,
# so queries are embedded exactly the way the chunks were.
with open(INDEX_FOLDER_PATH / 'config.json', encoding='utf-8') as file:
    config = json.load(file)
index = faiss.read_index(str(INDEX_FOLDER_PATH / 'chunks.faiss'))
chunks_df = pd.read_csv(INDEX_FOLDER_PATH / 'chunks.csv')

model = SentenceTransformer(
    config['embedding_model'],
    model_kwargs={"dtype": getattr(torch, config['embedding_dtype'])},    # "float32" -> torch.float32
)

contract_text = (CONTRACTS_INPUT_PATH_FOLDER / CONTRACT).read_text(encoding='utf-8')

# Search only this contract's chunks. Across all 30 contracts, "the agreement" is
# ambiguous: every contract has a governing-law clause, so global search returns
# other contracts' clauses first.
contract_positions = np.where(chunks_df['filename'] == CONTRACT)[0].astype('int64')
assert len(contract_positions) > 0, f"{CONTRACT} is not in the index"
search_params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(contract_positions))

failures = []

for question, gold_span in TEST_CASES:
    # Where the answer sits in the raw text. The gold chunk is one that contains all of it.
    span_start = contract_text.find(gold_span)
    assert span_start != -1, f"Gold span not found in the contract: {gold_span[:60]!r}"
    span_end = span_start + len(gold_span)

    # Questions get Qwen's "query" prompt; chunks were embedded without one.
    query_vector = model.encode([question], prompt_name="query", normalize_embeddings=True)
    scores, positions = index.search(query_vector, TOP_K, params=search_params)

    gold_rank = None
    for rank, position in enumerate(positions[0], start=1):
        chunk = chunks_df.iloc[position]
        if chunk['start_char'] <= span_start and chunk['end_char'] >= span_end:
            gold_rank = rank
            break

    result = f"rank {gold_rank}" if gold_rank else f"MISSING from top {TOP_K}"
    print(f"{result:<20} top score {scores[0][0]:.2f}   {question}")
    if gold_rank is None:
        failures.append(question)

assert not failures, f"{len(failures)} question(s) did not retrieve their gold chunk: {failures}"
print(f"All {len(TEST_CASES)} questions retrieved their gold chunk in the top {TOP_K}.")
