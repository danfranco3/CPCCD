# eval_only.py

import json
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report, accuracy_score
import os

MODEL_NAME = "Salesforce/codet5p-220m"
CLONE_DATASETS = [
    'python_cobol',
    'java_fortran',
    'js_pascal'
]
MAX_LENGTH = 2800
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeCloneDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=MAX_LENGTH):
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

def few_shot_prompt(example_pairs, target_pair):
    prompt = "Are the following code snippet pairs functionally equivalent?\n\n"
    for ex in example_pairs:
        label = "Yes" if ex["label"] == 1 else "No"
        prompt += f"Code1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer: {label}\n\n"
    prompt += f"Code1:\n{target_pair['code1']}\n\nCode2:\n{target_pair['code2']}\n\nAnswer:"
    return prompt

def predict(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def predict_few_shot(example_pairs, target_pair, model, tokenizer):
    prompt = few_shot_prompt(example_pairs, target_pair)
    return predict(prompt, model, tokenizer)

def evaluate_model(model, tokenizer, test_data, raw_examples, output_path):
    model.eval()
    preds = []
    targets = []
    outputs = []

    loader = DataLoader(test_data, batch_size=1)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            generated_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=10)

            pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
            true = tokenizer.decode(labels[0], skip_special_tokens=True).strip().lower()

            pred_label = 1 if 'yes' in pred else 0
            true_label = 1 if 'yes' in true else 0

            preds.append(pred_label)
            targets.append(true_label)

            # Save detailed output
            outputs.append({
                "code1": raw_examples[i]["code1"],
                "code2": raw_examples[i]["code2"],
                "model_output": pred,
                "pred_label": pred_label,
                "true_label": true_label
            })

    print("\nEvaluation Results:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    # Save JSON output
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Results saved to {output_path}")


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/codet5", exist_ok=True)
    
    for code_set in CLONE_DATASETS:
        print(f"\n=== Evaluating dataset: {code_set} ===")
        test_path = f"src/data/{code_set}_test.json"
        output_path = f"results/codet5/{code_set}_eval_output.json"

        with open(test_path) as f:
            test_examples = json.load(f)

        test_dataset = CodeCloneDataset(test_path, tokenizer)

        # Run evaluation and save detailed outputs
        evaluate_model(model, tokenizer, test_dataset, test_examples, output_path)

        # Optional: one-shot and few-shot demo
        print("\nOne-shot and Few-shot:")
        for i, ex in enumerate(test_examples[:3]):
            one_shot = predict(few_shot_prompt([], ex), model, tokenizer)
            few_shot = predict_few_shot(test_examples[:2], ex, model, tokenizer)
            print(f"One-shot [{i}]: {one_shot}")
            print(f"Few-shot [{i}]: {few_shot}")


if __name__ == "__main__":
    run()
