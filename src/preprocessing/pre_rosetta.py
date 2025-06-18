import os
import random
import pandas as pd
import json
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

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
    
    df1 = df1[~df1["code"].str.contains("$ sudo", na=False)].reset_index(drop=True)
    df2 = df2[~df2["code"].str.contains("$ sudo", na=False)].reset_index(drop=True)
    
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



def save_dataset(train_data, test_data, prefix='tasksplit_clones', data_folder='rosetta'):
    train_json = train_data[['code1', 'code2', 'label']].to_dict(orient='records')
    test_json = test_data[['code1', 'code2', 'label']].to_dict(orient='records')
    p = 'src' if os.getcwd().endswith("CPCCD") else os.path.abspath(__file__)
    
    os.makedirs(os.path.join(p, 'data'), exist_ok=True)
    folder = os.path.join(p, 'data', data_folder)
    os.makedirs(folder, exist_ok=True)
    
    with open(os.path.join(folder, f'{prefix}_train.json'), 'w') as f:
        json.dump(train_json, f, indent=2)
    with open(os.path.join(folder, f'{prefix}_test.json'), 'w') as f:
        json.dump(test_json, f, indent=2)
    print(f"Saved: {prefix}_train.json ({len(train_json)} examples), {prefix}_test.json ({len(test_json)} examples)")

def print_clone_stats(train, test, label):
    print(f"\nStats for {label}:")
    print(f"  Train clones:     {sum(train['label'] == 1)}")
    print(f"  Train non-clones: {sum(train['label'] == 0)}")
    print(f"  Test clones:      {sum(test['label'] == 1)}")
    print(f"  Test non-clones:  {sum(test['label'] == 0)}")

def main():
    max_chars = 650  # Filter snippet pairs where both are under 1000 chars
    
    ds = load_dataset("christopher/rosetta-code", split='train')
    df = Dataset.to_pandas(ds)

    train, test = generate_clone_dataset(df, 'Java', 'Fortran', test_size=0.3, max_chars=max_chars)
    print_clone_stats(train, test, "Java-Fortran")
    save_dataset(train, test, prefix='java_fortran')

    train, test = generate_clone_dataset(df, 'Python', 'COBOL', test_size=0.3, max_chars=max_chars)
    print_clone_stats(train, test, "Python-COBOL")    
    save_dataset(train, test, prefix='python_cobol')

    train, test = generate_clone_dataset(df, 'JavaScript', 'Free Pascal', test_size=0.3, max_chars=max_chars)
    print_clone_stats(train, test, "JavaScript-Pascal")
    save_dataset(train, test, prefix='js_pascal')
    

if __name__ == "__main__":
    main()
