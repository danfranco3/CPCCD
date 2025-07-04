import json
from torch.utils.data import Dataset
import torch

class CodeCloneDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, seq=True):
        with open(path) as f:
            raw_data = json.load(f)
        self.data = [
            {
                "input": f"Are the following two code snippets functionally equivalent?\n\nCode1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer:",
                "output": "Yes" if ex["label"] == 1 else "No"
            } for ex in raw_data
        ]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seq = seq

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.seq:
            sample = self.data[idx]
            enc = self.tokenizer(
                sample["input"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            label = 1 if "Yes" in sample["output"] else 0  # or just store `ex["label"]` directly earlier
            return {
                "input_ids": enc["input_ids"].squeeze(),
                "attention_mask": enc["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            }
        else:
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