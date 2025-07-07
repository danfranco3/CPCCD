import json
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

class CodeCloneDataset(Dataset):
    def __init__(self, path, tokenizer, max_length, model_type="bert"):
        with open(path) as f:
            raw_data = json.load(f)
        self.data = [
            {
                "code1": ex['code1'],
                "code2": ex["code2"],
                "output": ex["label"]
            } for ex in raw_data
        ]
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.model_type = model_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.model_type == "codet5":
        
            combined_code = sample["code1"] + " </s> " + sample["code2"]

            enc = self.tokenizer(
                combined_code,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(sample["output"], dtype=torch.long)
            }
            
        else:
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
                "labels": torch.tensor(sample["output"], dtype=torch.long)
            }
        
