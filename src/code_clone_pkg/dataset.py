import json
from torch.utils.data import Dataset
import torch

class CodeCloneDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        with open(path) as f:
            raw_data = json.load(f)
        self.data = [
            {
                "input": f"Are the following two code snippets functionally equivalent?\n\nCode1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer:",
                "output": ex["label"] == 1
            } for ex in raw_data
        ]
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        enc = self.tokenizer(
            sample["input"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        labels = self.tokenizer(
            sample["output"],
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )["input_ids"]
        enc["labels"] = labels
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }