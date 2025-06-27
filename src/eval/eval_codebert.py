import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score
import os
from code_clone_pkg.data_utils import load_multiple_datasets
from code_clone_pkg.dataset import CodeCloneDataset

# Configuration
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 1400
BATCH_SIZE = 3
THRESHOLD = 0.9  # Cosine similarity threshold to decide "clone" vs "non-clone"
OUTPUT_DIR = "results/codebert"
CLONE_DATASETS = [
    'python_cobol',
    'java_fortran',
    'js_pascal'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_embedding(text, tokenizer, model):
    encoded = tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state  # (1, seq_len, hidden_size)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)
        mean_embedding = summed / count
    return mean_embedding


def evaluate_model(model, tokenizer, test_examples, output_path):
    preds = []
    targets = []
    outputs = []

    for ex in test_examples:
        code1 = ex["code1"]
        code2 = ex["code2"]
        label = ex["label"]

        emb1 = get_mean_embedding(code1, tokenizer, model)
        emb2 = get_mean_embedding(code2, tokenizer, model)

        sim = F.cosine_similarity(emb1, emb2).item()
        pred = 1 if sim >= THRESHOLD else 0

        preds.append(pred)
        targets.append(label)
        outputs.append({
            "code1": code1,
            "code2": code2,
            "cosine_similarity": sim,
            "pred_label": pred,
            "true_label": label
        })

    print("\nEvaluation Results:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Detailed results saved to {output_path}")


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    for code_set in CLONE_DATASETS:
        print(f"\n=== Running for dataset: {code_set} ===")
        test_path = f"src/data/rosetta/{code_set}_test.json"

        with open(test_path) as f:
            test_examples = json.load(f)

        output_path = f"{OUTPUT_DIR}/{code_set}_cosine_mean_pooling.json"
        evaluate_model(model, tokenizer, test_examples, output_path)


if __name__ == "__main__":
    run()
