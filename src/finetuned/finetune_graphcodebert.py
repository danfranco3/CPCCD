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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from code_clone_pkg.dataset import CodeCloneDataset

# Configuration
MODEL_NAME = "microsoft/graphcodebert-base"
OUTPUT_DIR = "results/graphcodebert_finetune"
MAX_LENGTH = 512
EPOCHS = 9
BATCH_SIZE = 2
CLONE_DATASETS = ['python_cobol', 'java_fortran', 'js_pascal']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

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
    full_dataset = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
    
    with open("src/data/combined_train.json") as f:
        raw_data = json.load(f)
    labels = [ex["label"] for ex in raw_data]

    train_indices, val_indices = train_test_split(
        list(range(len(raw_data))),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        gradient_accumulation_steps=16,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/graphcodebert_finetuned")
    
    best_model = trainer.model
    best_model.eval()

    for dataset_name in CLONE_DATASETS:
        test_path = f"src/data/rosetta/{dataset_name}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        print(f"\nEvaluating zero-shot on dataset: {dataset_name}")
        evaluate_zero_shot(best_model, test_examples, dataset_name)

if __name__ == "__main__":
    run()
