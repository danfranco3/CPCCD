import json
import os
import torch
from torch.utils.data import Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from code_clone_pkg.dataset import CodeCloneDataset

# CONFIG
MODEL_NAME   = "microsoft/codebert-base"
OUTPUT_DIR   = "results/codebert_finetune"
MAX_LENGTH   = 512
EPOCHS       = 8
BATCH_SIZE   = 2
CLONE_DATASETS = ['python_cobol', 'java_fortran', 'js_pascal']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def evaluate_zero_shot(model, test_examples, dataset_name):
    preds, targets = [], []

    for ex in test_examples:
        tokens = tokenizer(
            ex["code1"],
            ex["code2"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**tokens)
            logits = outputs.logits
            pred = torch.argmax(logits, dim=-1).item()

        preds.append(pred)
        targets.append(ex["label"])

    return report_results(targets, preds, f"{dataset_name}_zero_shot_cls")


def report_results(y_true, y_pred, tag):
    print(f"\nEvaluation: {tag}")
    print(classification_report(y_true, y_pred, target_names=["Non-clone", "Clone"]))
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, f"{tag}.json"), "w") as f:
        json.dump({"preds": y_pred, "targets": y_true}, f)
    return acc

def run():
    # Load full dataset
    with open("src/data/combined_train.json") as f:
        full_data = json.load(f)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    train_full = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)
    labels = [ex["label"] for ex in full_data]

    train_idx, val_idx = train_test_split(
        list(range(len(full_data))),
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    train_ds = Subset(train_full, train_idx)
    val_ds   = Subset(train_full, val_idx)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        gradient_accumulation_steps=16,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/codebert_cls")

    best_model = trainer.model
    best_model.eval()

    for code_set in CLONE_DATASETS:
        print(f"\nDataset: {code_set}")
        with open(f"src/data/rosetta/{code_set}_test.json") as f:
            test_examples = json.load(f)
        evaluate_zero_shot(best_model, test_examples, code_set)

if __name__ == "__main__":
    run()
