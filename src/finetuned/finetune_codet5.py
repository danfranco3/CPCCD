import json
import torch
from torch.utils.data import random_split, Subset
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score
import os
from sklearn.model_selection import train_test_split
from code_clone_pkg.data_utils import load_multiple_datasets, sample_few_shot_examples
from code_clone_pkg.dataset import CodeCloneDataset

SUPPORT_PATHS = ["src/data/codeNet/ruby_go_test.json"]

# CONFIG
MODEL_NAME   = "Salesforce/codet5p-220m"
OUTPUT_DIR   = "results/codetp5"
MAX_LENGTH   = 512
EPOCHS       = 20
BATCH_SIZE   = 1
CLONE_DATASETS = ["python_cobol", "java_fortran", "js_pascal"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
encoder = model.base_model.encoder

def embed_code(code1: str, code2:str) -> torch.Tensor:
    tokens = tokenizer(
        code1,
        code2,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.base_model(**tokens)
        hidden = outputs.last_hidden_state
        emb = hidden[:, 0, :]  # CLS token
    return emb


def evaluate_zero_shot(test_examples, dataset_name):
    preds, targets = [], []

    for ex in test_examples:
        tokens = embed_code(ex["code1"], ex["code2"])

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
    # Load raw data
    with open("src/data/combined_train.json") as f:
        full_data = json.load(f)

    # Tokenized full dataset
    train_full = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)

    # Extract labels
    labels = [ex["label"] for ex in full_data]

    # Create stratified split
    train_indices, val_indices = train_test_split(
        list(range(len(full_data))),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # Create Subset datasets
    train_ds = Subset(train_full, train_indices)
    val_ds   = Subset(train_full, val_indices)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_dir=f"{OUTPUT_DIR}/logs",
        load_best_model_at_end=True,
        gradient_accumulation_steps=16,
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/codet5p_cls")

    model.eval()
    
    for code_set in CLONE_DATASETS:
        test_path = f"src/data/rosetta/{code_set}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        print(f"\nDataset: {code_set}")
        evaluate_zero_shot(test_examples, code_set)

if __name__ == "__main__":
    run()
