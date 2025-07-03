import json
import os
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets

MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "results/codebert_eval_only"
MAX_LENGTH = 1400
THRESHOLD = 0.85
CLONE_DATASETS = ['python_cobol', 'java_fortran', 'js_pascal']
SUPPORT_PATHS = ["src/data/codeNet/ruby_go_test.json"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mean_embedding(text, tokenizer, model):
    encoded = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)
        mean_embedding = summed / count
    return mean_embedding


def evaluate_model_cosine(model, tokenizer, test_examples, output_path):
    model.eval()
    preds, targets, outputs = [], [], []

    for ex in tqdm(test_examples, desc="Zero-shot evaluation"):
        emb1 = get_mean_embedding(ex["code1"], tokenizer, model)
        emb2 = get_mean_embedding(ex["code2"], tokenizer, model)
        sim = F.cosine_similarity(emb1, emb2).item()
        pred = 1 if sim >= THRESHOLD else 0

        preds.append(pred)
        targets.append(ex["label"])
        outputs.append({
            "code1": ex["code1"],
            "code2": ex["code2"],
            "cosine_similarity": sim,
            "pred_label": pred,
            "true_label": ex["label"]
        })

    print("\nZero-shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)


def predict_one_or_two_shot(model, tokenizer, support_set, test_examples, output_path, shots=1):
    model.eval()
    preds, targets, outputs = [], [], []

    for i, ex in tqdm(enumerate(test_examples), total=len(test_examples), desc=f"{shots}-shot evaluation"):
        if shots == 1:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i)
        elif shots == 2:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i) + \
                      sample_few_shot_examples(support_set, n=1, label=0, seed=i + 100)

        query_emb = get_mean_embedding(ex["code1"] + ex["code2"], tokenizer, model)

        sims = []
        for s in support:
            support_emb = get_mean_embedding(s["code1"] + s["code2"], tokenizer, model)
            sims.append(F.cosine_similarity(query_emb, support_emb).item())

        avg_sim = sum(sims) / len(sims)
        pred = 1 if avg_sim >= THRESHOLD else 0

        preds.append(pred)
        targets.append(ex["label"])
        outputs.append({
            "support_examples": support,
            "query": ex,
            "avg_similarity": avg_sim,
            "pred_label": pred,
            "true_label": ex["label"]
        })

    print(f"\n{shots}-shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)


def run_evaluation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    support_set = load_multiple_datasets(SUPPORT_PATHS)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Evaluating dataset: {code_set} ===")
        test_path = f"src/data/rosetta/{code_set}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        # Zero-shot
        zero_shot_output = f"{OUTPUT_DIR}/{code_set}_zero_shot.json"
        evaluate_model_cosine(model, tokenizer, test_examples, zero_shot_output)

        # One-shot
        one_shot_output = f"{OUTPUT_DIR}/{code_set}_one_shot.json"
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, one_shot_output, shots=1)

        # Two-shot
        two_shot_output = f"{OUTPUT_DIR}/{code_set}_two_shot.json"
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, two_shot_output, shots=2)


if __name__ == "__main__":
    run_evaluation()
