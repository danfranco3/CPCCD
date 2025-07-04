# evaluate_pretrained_graphcodebert.py

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, accuracy_score
import os

from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets

MODEL_NAME = "microsoft/graphcodebert-base"
MAX_LENGTH = 512
THRESHOLD = 0.9
CLONE_DATASETS = ['python_cobol', 'java_fortran', 'js_pascal']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_pair_embedding(code1, code2, tokenizer, model):
    inputs1 = tokenizer(code1, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH).to(device)
    inputs2 = tokenizer(code2, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        h1 = model(**inputs1).last_hidden_state[:, 0]
        h2 = model(**inputs2).last_hidden_state[:, 0]
        return torch.cat([h1, h2], dim=-1)

def evaluate_cosine(model, tokenizer, test_examples, output_path):
    model.eval()
    preds, targets, outputs = [], [], []

    for ex in test_examples:
        emb = get_pair_embedding(ex["code1"], ex["code2"], tokenizer, model)
        sim = F.cosine_similarity(emb[:, :emb.shape[1] // 2], emb[:, emb.shape[1] // 2:]).item()
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

def few_shot_eval(model, tokenizer, support_set, test_examples, output_path, shots=1):
    model.eval()
    preds, targets, outputs = [], [], []

    for i, ex in enumerate(test_examples):
        if shots == 1:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i)
        else:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i) + \
                      sample_few_shot_examples(support_set, n=1, label=0, seed=i+100)

        query_emb = get_pair_embedding(ex["code1"], ex["code2"], tokenizer, model)

        sim_by_class = {0: [], 1: []}
        for s in support:
            support_emb = get_pair_embedding(s["code1"], s["code2"], tokenizer, model)
            sim = F.cosine_similarity(query_emb, support_emb).item()
            sim_by_class[s["label"]].append(sim)

        avg_sim_0 = sum(sim_by_class[0]) / len(sim_by_class[0]) if sim_by_class[0] else -1
        avg_sim_1 = sum(sim_by_class[1]) / len(sim_by_class[1]) if sim_by_class[1] else -1
        pred = 1 if avg_sim_1 > avg_sim_0 else 0

        preds.append(pred)
        targets.append(ex["label"])
        outputs.append({
            "support_examples": support,
            "query": ex,
            "avg_sim_class_0": avg_sim_0,
            "avg_sim_class_1": avg_sim_1,
            "pred_label": pred,
            "true_label": ex["label"]
        })

    print(f"\n{shots}-shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)

def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Dataset: {code_set} ===")
        with open(f"src/data/rosetta/{code_set}_test.json") as f:
            test_examples = json.load(f)

        evaluate_cosine(model, tokenizer, test_examples, f"results/graphcodebert_pretrained/{code_set}_zero_shot.json")

        support_set = load_multiple_datasets(["src/data/codeNet/ruby_go_test.json"])
        few_shot_eval(model, tokenizer, support_set, test_examples, f"results/graphcodebert_pretrained/{code_set}_one_shot.json", shots=1)
        few_shot_eval(model, tokenizer, support_set, test_examples, f"results/graphcodebert_pretrained/{code_set}_two_shot.json", shots=2)

if __name__ == "__main__":
    run()
