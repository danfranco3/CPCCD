import json
from torch.utils.data import Dataset
import torch

class CodeCloneDataset(Dataset):
    def __init__(self, path, tokenizer, max_length):
        with open(path) as f:
            raw_data = json.load(f)
        self.data = [
            {
                "code1": ex['code1'],
                "code2": ex["code2"],
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
            sample["code1"],
            sample["code2"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(sample["output"]), dtype=torch.long)
        }
