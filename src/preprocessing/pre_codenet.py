import json
from datasets import load_dataset
import pandas as pd
import random
from typing import List, Tuple
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_accepted_subset(dataset_name: str, subset_name: str) -> pd.DataFrame:
    """Load dataset subset and return dataframe with status == 'Accepted'."""
    ds = load_dataset(dataset_name, subset_name)
    df = ds["train"].to_pandas()
    return df[df["status"] == "Accepted"].reset_index(drop=True)

def stream_and_match_both(
    dataset_name: str,
    source_subset: str,
    target_subset: str,
    max_matches: int = 150,
    max_code_len: int = 500  # your 'n' value here
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match target samples against accepted source by p_id and return both aligned subsets."""
    source_df = get_accepted_subset(dataset_name, source_subset)
    accepted_p_ids = set(source_df["p_id"])
    source_df = source_df[source_df["p_id"].isin(accepted_p_ids)].drop_duplicates("p_id").set_index("p_id")

    target_stream = load_dataset(dataset_name, target_subset, split="train", streaming=True)

    matched_target_rows = []
    matched_source_rows = []
    selected_p_ids = set()

    pbar = tqdm(total=max_matches, desc="Matching Accepted Samples")

    for sample in target_stream:
        pid = sample["p_id"]
        if (
            sample["status"] == "Accepted"
            and pid in accepted_p_ids
            and pid not in selected_p_ids
            and len(sample["code"]) <= max_code_len
            and len(sample["code"]) == int(sample.get("code_size", -1))
        ):
            source_row = source_df.loc[pid]
            if (
                len(source_row["code"]) <= max_code_len
                and len(source_row["code"]) == int(source_row.get("code_size", -1))
            ):
                matched_target_rows.append(sample)
                matched_source_rows.append(source_row.to_dict())
                selected_p_ids.add(pid)
                pbar.update(1)

        if len(selected_p_ids) >= max_matches:
            break

    pbar.close()

    df_target = pd.DataFrame(matched_target_rows)
    df_source = pd.DataFrame(matched_source_rows)
    return df_target.reset_index(drop=True), df_source.reset_index(drop=True)


def deranged_shuffle(arr: List[str], seed: int = 42) -> List[str]:
    """Returns a shuffled list where no element remains in its original position."""
    rng = random.Random(seed)
    while True:
        shuffled = arr[:]
        rng.shuffle(shuffled)
        if all(orig != shuffled[i] for i, orig in enumerate(arr)):
            return shuffled

def create_balanced_clone_dataset_with_test_limit(
    df_target: pd.DataFrame,
    df_source: pd.DataFrame,
    max_test_matches: int,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split matched samples into test/train using fixed number of test matches,
    and build balanced datasets (positive + deranged negative) for both sets.
    """

    df_target = df_target[["code"]].dropna().reset_index(drop=True)
    df_source = df_source[["code"]].dropna().reset_index(drop=True)

    df_target_test = df_target.iloc[:max_test_matches].reset_index(drop=True)
    df_source_test = df_source.iloc[:max_test_matches].reset_index(drop=True)

    df_target_train = df_target.iloc[max_test_matches:].reset_index(drop=True)
    df_source_train = df_source.iloc[max_test_matches:].reset_index(drop=True)

    def build_pos_neg(df1, df2, label, seed):
        pos_df = pd.DataFrame({
            "code1": df1["code"],
            "code2": df2["code"],
            "label": label
        })
        
        deranged_codes = deranged_shuffle(df2["code"].tolist(), seed=seed)
        neg_df = pd.DataFrame({
            "code1": df1["code"],
            "code2": deranged_codes,
            "label": 1 - label
        })
        return pos_df, neg_df

    
    test_pos, test_neg = build_pos_neg(df_target_test, df_source_test, 1, seed + 100)
    test_df = pd.concat([test_pos, test_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

   
    train_pos, train_neg = build_pos_neg(df_target_train, df_source_train, 1, seed + 200)
    train_df = pd.concat([train_pos, train_neg]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, test_df


def split_clone_dataset(
    df: pd.DataFrame,
    test_size: float = 0.4,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the clone dataset into training and test sets.
    
    Returns:
        train_df, test_df
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def save_dataset(train_data, test_data, prefix='tasksplit_clones', data_folder='codeNet'):
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


def main():
    
    MAX_LENGTH = 400
    MAX_MATCHES = 24
    MAX_TEST_MATCHES = 8
    DS_NAME = "iNeil77/CodeNet"
    
    code1_df, code2_df = stream_and_match_both(DS_NAME, "COBOL", "Python", max_code_len=MAX_LENGTH, max_matches=MAX_MATCHES)
    train_df, test_df = create_balanced_clone_dataset_with_test_limit(
        code1_df, code2_df, max_test_matches=MAX_TEST_MATCHES
    )
    save_dataset(train_df, test_df, 'python_cobol', data_folder='codeNet')
    
    
    code1_df, code2_df = stream_and_match_both(DS_NAME, "Fortran", "Java", max_code_len=MAX_LENGTH, max_matches=MAX_MATCHES)
    train_df, test_df = create_balanced_clone_dataset_with_test_limit(
        code1_df, code2_df, max_test_matches=MAX_TEST_MATCHES
    )

    save_dataset(train_df, test_df, 'java_fortran', data_folder='codeNet')


    code1_df, code2_df = stream_and_match_both(DS_NAME, "Pascal", "JavaScript", max_code_len=MAX_LENGTH, max_matches=MAX_MATCHES)
    train_df, test_df = create_balanced_clone_dataset_with_test_limit(
        code1_df, code2_df, max_test_matches=MAX_TEST_MATCHES
    )
    save_dataset(train_df, test_df, 'js_pascal', data_folder='codeNet')
    
    
if __name__ == "__main__":
    main()

