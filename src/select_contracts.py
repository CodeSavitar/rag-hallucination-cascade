from pathlib import Path
import pandas as pd

# Path to the project root directory
BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / 'CUAD_v1' / 'master_clauses.csv'

df = pd.read_csv(csv_path)
df_copy = df.copy()
df_copy["size"] = 0.0

for i, x in enumerate(df['Filename']):
    filename = Path(x).stem
    txt_file = f"{filename}.txt"
    search_path = BASE_DIR / 'CUAD_v1' / 'full_contract_txt' / txt_file

    if not search_path.is_file():
        df_copy = df_copy.drop(index=i)

    else:
        df_copy.loc[i, 'size'] = (search_path.stat().st_size)/1024
        df_copy.loc[i, 'Filename'] = txt_file
        if df_copy['size'][i] < 16 or df_copy['size'][i] > 65:
            df_copy = df_copy.drop(index=i)

df_copy = df_copy.drop(columns=[col for col in df_copy.columns if "-Answer" in col or "- Answer" in col])
df_copy = df_copy.reset_index(drop=True)
df_copy["clause_count"] = 0

for i, row in df_copy.iterrows():
    count = 0

    for j, column in enumerate(df_copy.columns):
        if df_copy[column][i] != "[]" and column not in ["Filename", "size", "clause_count"]:
            count += 1

    df_copy.loc[i, 'clause_count'] = count
            

df_sorted = df_copy.sort_values(by="clause_count", ascending=False)
df_sorted = df_sorted.reset_index(drop=True)
df_top = df_sorted.head(30)

# Every clause type must appear in at least one of the selected contracts
for column in df_top.columns:
    if column not in ["Filename", "size", "clause_count"]:
        assert (df_top[column] != "[]").any(), f"No selected contract has the clause type '{column}'"

df_final = df_top[["Filename", "size", "clause_count"]]

output_path = Path(f"{BASE_DIR}/data/selected_contracts.csv")

df_final.to_csv(output_path, index=False)
