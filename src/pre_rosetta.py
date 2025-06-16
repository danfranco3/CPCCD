import os
import random
import pandas as pd
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split


ds = load_dataset("christopher/rosetta-code", split='train')
df = Dataset.to_pandas(ds)


def get_language_pairs(df, lang1, lang2):
    df1 = df[df['language_name'] == lang1]
    df2 = df[df['language_name'] == lang2]
    
    common_tasks = set(df1['task_name']).intersection(set(df2['task_name']))
    df1 = df1[df1['task_name'].isin(common_tasks)].copy()
    df2 = df2[df2['task_name'].isin(common_tasks)].copy()
    
    df1 = df1[df1['task_name'] != 'Comments']
    df2 = df2[df2['task_name'] != 'Comments']
    
    return df1[['task_name', 'code']], df2[['task_name', 'code']]


def build_positive_pairs(df1, df2, max_chars=None):
    merged = pd.merge(df1, df2, on='task_name')
    merged = merged.rename(columns={'code_x': 'code1', 'code_y': 'code2'})
    if max_chars is not None:
        merged = merged[
            (merged['code1'].str.len() < max_chars) &
            (merged['code2'].str.len() < max_chars)
        ]
    merged['label'] = 1
    return merged[['task_name', 'code1', 'code2', 'label']]


def build_negative_pairs(df1, df2, task_pool, num_pairs, max_chars=None):
    negatives = []
    task_to_code1 = df1.groupby('task_name').apply(lambda g: g.iloc[0]['code'])
    task_to_code2 = df2.groupby('task_name').apply(lambda g: g.iloc[0]['code'])
    tasks = list(task_pool)

    while len(negatives) < num_pairs:
        t1, t2 = random.sample(tasks, 2)
        c1 = task_to_code1[t1]
        c2 = task_to_code2[t2]

        if max_chars is not None and (len(c1) >= max_chars or len(c2) >= max_chars):
            continue

        negatives.append({
            "task_name": f"{t1}--{t2}",
            "code1": c1,
            "code2": c2,
            "label": 0
        })

    return pd.DataFrame(negatives)



def generate_clone_dataset(df, lang1, lang2, test_size=0.3, seed=42, max_chars=None):
    df1, df2 = get_language_pairs(df, lang1, lang2)
    tasks = sorted(set(df1['task_name']) & set(df2['task_name']))
    train_tasks, test_tasks = train_test_split(tasks, test_size=test_size, random_state=seed)

    def subset(df_lang, task_list):
        return df_lang[df_lang['task_name'].isin(task_list)]

    train_df1 = subset(df1, train_tasks)
    train_df2 = subset(df2, train_tasks)
    test_df1 = subset(df1, test_tasks)
    test_df2 = subset(df2, test_tasks)

    train_pos = build_positive_pairs(train_df1, train_df2, max_chars=max_chars)
    test_pos = build_positive_pairs(test_df1, test_df2, max_chars=max_chars)

    train_neg = build_negative_pairs(train_df1, train_df2, train_tasks, len(train_pos), max_chars=max_chars)
    test_neg = build_negative_pairs(test_df1, test_df2, test_tasks, len(test_pos), max_chars=max_chars)

    train_data = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_data = pd.concat([test_pos, test_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_data, test_data



def save_dataset(train_data, test_data, prefix='tasksplit_clones'):
    train_json = train_data[['code1', 'code2', 'label']].to_dict(orient='records')
    test_json = test_data[['code1', 'code2', 'label']].to_dict(orient='records')
    data_dir = os.path.dirname(os.path.abspath(__file__))
    
    if not os.path.exists(os.path.join(data_dir, 'data')):
        os.makedirs(os.path.join(data_dir, 'data'))
    
    with open(os.path.join(data_dir, f'data/{prefix}_train.json'), 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(data_dir, f'data/{prefix}_test.json'), 'w') as f:
        json.dump(test_json, f, indent=2)
    print(f"Saved: {prefix}_train.json ({len(train_json)} examples), {prefix}_test.json ({len(test_json)} examples)")


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


def main():
    max_chars = 1000  # Filter snippet pairs where both are under 1000 chars

    train, test = generate_clone_dataset(df, 'Java', 'Fortran', test_size=0.3, max_chars=max_chars)
    save_dataset(train, test, prefix='java_fortran_clones')

    train, test = generate_clone_dataset(df, 'Python', 'COBOL', test_size=0.3, max_chars=max_chars)
    save_dataset(train, test, prefix='python_cobol_clones')

    train, test = generate_clone_dataset(df, 'JavaScript', 'Free Pascal', test_size=0.3, max_chars=max_chars)
    save_dataset(train, test, prefix='js_pascal_clones')

    code_snippets = extract_cobol_fortran_pascal(df)
    save_tokenizer_dataset(code_snippets, filename='cobol_fortran_pascal_code.json')


if __name__ == "__main__":
    main()
