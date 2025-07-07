import json
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F

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
        

def get_code_embedding(model, tokenizer, code1, code2, model_type="codet5", max_length=512, device="cuda"):
    """
    Returns a single embedding vector for a pair of code snippets from either CodeBERT or CodeT5+.
    
    Args:
        model: Hugging Face model instance (CodeBERT or CodeT5+).
        tokenizer: Corresponding tokenizer.
        code1, code2: Strings (code snippets).
        model_type: 'codebert' or 'codet5'.
        max_length: Token length cap.
        device: "cuda" or "cpu".
        
    Returns:
        Tensor: Normalized [batch_size, hidden_dim] embedding.
    """
    model.eval()

    tokens = tokenizer(
        code1,
        code2,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        if model_type in ["codebert", "graphcodebert"]:
            outputs = model(**tokens)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token


        elif model_type == "codet5":
            # Mean pooling over encoder output (no [CLS] token)
            encoder_outputs = model.base_model.encoder(
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"]
            )
            hidden = encoder_outputs.last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).expand(hidden.size())
            emb = (hidden * mask).sum(1) / mask.sum(1)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    # L2-normalize for consistency
    return F.normalize(emb, p=2, dim=1)
