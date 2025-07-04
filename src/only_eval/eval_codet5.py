import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import classification_report, accuracy_score
import os
from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets
from code_clone_pkg.dataset import CodeCloneDataset
from code_clone_pkg.prompts import few_shot_prompt
from code_clone_pkg.utils import extend_tokenizer_and_resize_model

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"
OUTPUT_DIR = "results/eval"
MAX_LENGTH = 1024
CLONE_DATASETS = [
    'python_cobol',
    'java_fortran',
    'js_pascal'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=3,
        temperature=0.0,
        do_sample=False,
        num_beams=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id or tokenizer.convert_tokens_to_ids(["</s>"])[0],
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()

    # Just return "yes" or "no" if possible
    if "yes" in decoded:
        return "Yes"
    elif "no" in decoded:
        return "No"
    else:
        return decoded  # fallback (can help debug edge cases)


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

            outputs.append({
                "code1": raw_examples[i]["code1"],
                "code2": raw_examples[i]["code2"],
                "model_output": pred,
                "pred_label": pred_label,
                "true_label": true_label
            })

    print("\nZero-Shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved to {output_path}")

def eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, neg=False):
    targets, preds, outputs = [], [], []

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            few_shot_support = sample_few_shot_examples(
                support_dataset, n=1, label=0 if neg else 1, seed=i + (100 if neg else 0)
            )
            prompt = few_shot_prompt(few_shot_support, ex)
            output_text = predict(prompt, model, tokenizer)
            pred_label = 1 if 'yes' in output_text.strip().lower() else 0

            targets.append(ex["label"])
            preds.append(pred_label)
            outputs.append({
                "input": prompt,
                "prediction": output_text,
                "pred_label": pred_label,
                "true_label": ex["label"],
                "code1": ex.get("code1"),
                "code2": ex.get("code2"),
            })

    mode = 'Negative' if neg else 'Positive'
    print(f"\n{mode} One-Shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    suffix = "neg" if neg else "pos"
    output_path = f"{OUTPUT_DIR}/codet5/{code_set}_{suffix}_one_shot.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved to {output_path}")

def eval_two_shot(code_set, support_dataset, test_examples, model, tokenizer):
    targets, preds, outputs = [], [], []

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            few_shot_support = (
                sample_few_shot_examples(support_dataset, n=1, label=1, seed=i) +
                sample_few_shot_examples(support_dataset, n=1, label=0, seed=i + 100)
            )
            prompt = few_shot_prompt(few_shot_support, ex)
            output_text = predict(prompt, model, tokenizer)
            pred_label = 1 if 'yes' in output_text.strip().lower() else 0

            targets.append(ex["label"])
            preds.append(pred_label)
            outputs.append({
                "input": prompt,
                "prediction": output_text,
                "pred_label": pred_label,
                "true_label": ex["label"],
                "code1": ex.get("code1"),
                "code2": ex.get("code2"),
            })

    print("\nTwo-Shot Evaluation:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    output_path = f"{OUTPUT_DIR}/codet5/{code_set}_few_shot.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Saved to {output_path}")

def run_evaluation():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Evaluating on dataset: {code_set} ===")
        test_path = f"src/data/rosetta/{code_set}_test.json"
        test_dataset = CodeCloneDataset(test_path, tokenizer, MAX_LENGTH)

        with open(test_path) as f:
            test_examples = json.load(f)

        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device)
        if torch.cuda.is_available():
            model.gradient_checkpointing_enable()

        tokenizer, model = extend_tokenizer_and_resize_model(
            model,
            tokenizer,
            custom_tokenizer_path="bpe_cobol_fortran_pascal.json"
        )

        evaluate_model(model, tokenizer, test_dataset, test_examples,
                       output_path=f"{OUTPUT_DIR}/codet5/{code_set}_zero_shot.json")

        paths = [
            "src/data/codeNet/ruby_go_test.json",
        ]
        
        support_dataset = load_multiple_datasets(paths)

        eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, neg=False)
        eval_one_shot(code_set, support_dataset, test_examples, model, tokenizer, neg=True)
        eval_two_shot(code_set, support_dataset, test_examples, model, tokenizer)

if __name__ == "__main__":
    run_evaluation()
