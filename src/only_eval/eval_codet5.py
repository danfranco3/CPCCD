import json
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report, accuracy_score
from code_clone_pkg.data_utils import load_multiple_datasets, sample_few_shot_examples

# CONFIG
MODEL_NAME = "Salesforce/codet5p-220m"
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLONE_DATASETS = ["python_cobol", "java_fortran", "js_pascal"]
SUPPORT_PATHS = ["src/data/codeNet/ruby_go_test.json"]
OUTPUT_DIR = "results/similarity_eval"
THRESHOLD = 0.85  # Cosine similarity threshold

# Load model and tokenizer 
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
encoder = model.get_encoder()
model.eval()

# Embed code
def embed_code(code: str) -> torch.Tensor:
    tokens = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        hidden = encoder(**tokens).last_hidden_state  # (1, seq_len, hidden_size)
    return hidden[:, 0, :]  # use first token (like [CLS]) → shape: (1, hidden_size)

# Zero-shot evaluation 
def evaluate_zero_shot(test_examples, dataset_name):
    preds, targets = [], []
    for ex in test_examples:
        emb1 = embed_code(ex["code1"])
        emb2 = embed_code(ex["code2"])
        sim = F.cosine_similarity(emb1, emb2).item()
        pred = 1 if sim >= THRESHOLD else 0
        preds.append(pred)
        targets.append(ex["label"])
    report_results(targets, preds, f"{dataset_name}_zero_shot")

# Few-shot (1 or 2) using prototype averaging 
def evaluate_few_shot(test_examples, support_set, dataset_name, shots=1):
    preds, targets = [], []
    # Create average embedding (prototype) for each class
    prototypes = {}
    for label in [0, 1]:
        support = sample_few_shot_examples(support_set, n=shots, label=label, seed=42)
        support_embs = [embed_code(s["code1"] + "\n" + s["code2"]) for s in support]
        prototypes[label] = torch.stack(support_embs).mean(dim=0)  # shape: (hidden_size)

    for ex in test_examples:
        query = embed_code(ex["code1"] + "\n" + ex["code2"])  # (1, hidden_size)
        sims = {l: F.cosine_similarity(query, prototypes[l].unsqueeze(0)).item() for l in [0, 1]}
        pred = 1 if sims[1] > sims[0] else 0
        preds.append(pred)
        targets.append(ex["label"])
    report_results(targets, preds, f"{dataset_name}_{shots}shot")

# Print + save metrics 
def report_results(y_true, y_pred, tag):
    print(f"\nEvaluation: {tag}")
    print(classification_report(y_true, y_pred, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json"), "w") as f:
        json.dump({"preds": y_pred, "targets": y_true}, f)

# Main
def main():
    support_set = load_multiple_datasets(SUPPORT_PATHS)

    for code_set in CLONE_DATASETS:
        test_path = f"src/data/rosetta/{code_set}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        print(f"\nDataset: {code_set}")
        evaluate_zero_shot(test_examples, code_set)
        evaluate_few_shot(test_examples, support_set, code_set, shots=1)
        evaluate_few_shot(test_examples, support_set, code_set, shots=2)

if __name__ == "__main__":
    main()
