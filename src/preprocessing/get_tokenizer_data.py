from datasets import load_dataset, Dataset
import pandas as pd

def extract_cobol_fortran_pascal(df):
    langs = ['COBOL', 'Fortran', 'Free Pascal']
    filtered = df[df['language_name'].isin(langs)]
    filtered = filtered[filtered['task_name'] != 'Comments']
    code_snippets = filtered['code'].tolist()
    print(f"Extracted {len(code_snippets)} code snippets from {langs}")
    return code_snippets


def save_tokenizer_dataset(code_snippets, filename='cobol_fortran_pascal_code.json'):
    import json, os
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(data_dir, 'data'), exist_ok=True)
    path = os.path.join(data_dir, 'data', filename)
    with open(path, 'w') as f:
        json.dump(code_snippets, f, indent=2)
    print(f"Saved tokenizer training dataset: {path}")

DS_NAME = "iNeil77/CodeNet"

langs = ['COBOL', 'Fortran', 'Pascal']

code_list = []

for l in langs:

    ds = load_dataset(DS_NAME, l)
    df = ds["train"].to_pandas()

    accepted = df[df["status"] == "Accepted"].reset_index(drop=True)
    
    c = [accepted["code"].iloc[i] for i in range(len(accepted)) if int(accepted["code_size"].iloc[i]) == len(accepted["code"].iloc[i])]
    
    code_list.extend(c)
    
    print(f"Extracted {len(c)} code snippets from {l} in CodeNet")

# Rosetta code
ds = load_dataset("christopher/rosetta-code", split='train')
df = Dataset.to_pandas(ds)

code_list.extend(extract_cobol_fortran_pascal(df))
save_tokenizer_dataset(code_list, filename='cobol_fortran_pascal_code.json')