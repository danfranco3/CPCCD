import os
import json
from datasets import Dataset, DatasetDict

# Input and output directories
DATA_DIR = "src/data"  # change this if your JSON files are elsewhere
SAVE_DIR = "src/hf_data"  # where to save the Hugging Face datasets

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Get all pairs from available files
files = os.listdir(DATA_DIR)
train_files = [f for f in files if f.endswith("_train.json")]

for train_file in train_files:
    pair_name = train_file.replace("_train.json", "")
    test_file = f"{pair_name}_test.json"

    train_path = os.path.join(DATA_DIR, train_file)
    test_path = os.path.join(DATA_DIR, test_file)

    # Skip if test file doesn't exist
    if not os.path.exists(test_path):
        print(f"⚠️ Test file for {pair_name} not found, skipping...")
        continue

    # Load JSON data
    with open(train_path, "r") as f:
        train_data = json.load(f)
    with open(test_path, "r") as f:
        test_data = json.load(f)

    # Convert to Hugging Face DatasetDict
    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(test_data)
    })

    # Optional: Add validation split from train (e.g., 10%)
    split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)
    dataset = DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"],
        "test": dataset["test"]
    })

    # Save to disk
    save_path = os.path.join(SAVE_DIR, pair_name)
    dataset.save_to_disk(save_path)
    print(f"Saved {pair_name} dataset to: {save_path}")
