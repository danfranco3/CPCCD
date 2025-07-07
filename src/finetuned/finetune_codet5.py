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
EPOCHS       = 8
BATCH_SIZE   = 1
CLONE_DATASETS = ["python_cobol", "java_fortran", "js_pascal"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(DEVICE)
encoder = model.base_model.encoder

def embed_code(code: str) -> torch.Tensor:
    tokens = tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        hidden = encoder(
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"]
        ).last_hidden_state
    return hidden[:, 0, :]  # (1, hidden_size)


def evaluate_zero_shot(test_examples, dataset_name, threshold=0.75):
    preds, targets = [], []
    for ex in test_examples:
        emb1 = embed_code(ex["code1"])
        emb2 = embed_code(ex["code2"])
        sim = F.cosine_similarity(emb1, emb2).item()
        pred = 1 if sim >= threshold else 0
        preds.append(pred)
        targets.append(ex["label"])
    return report_results(targets, preds, f"{dataset_name}_zero_shot")


def evaluate_one_shot(test_examples, support_set, dataset_name, label, threshold=None):
    """
    One-shot with either a positive (1) or negative (0) support example.
    """
    preds, targets = [], []
    support = sample_few_shot_examples(support_set, n=1, label=label, seed=42)
    support_emb = embed_code(support[0]["code1"] + "\n" + support[0]["code2"]).squeeze(0)

    for ex in test_examples:
        query_emb = embed_code(ex["code1"] + "\n" + ex["code2"]).squeeze(0)
        sim = F.cosine_similarity(query_emb, support_emb, dim=0).item()
        pred = label if sim >= 0.5 else 1 - label  # compare to support class
        preds.append(pred)
        targets.append(ex["label"])
    
    tag = f"{dataset_name}_1shot_label{label}"
    return report_results(targets, preds, tag)


def evaluate_two_shot(test_examples, support_set, dataset_name, threshold=None):
    """
    Two-shot: one positive and one negative support example.
    """
    preds, targets = [], []
    support_pos = sample_few_shot_examples(support_set, n=1, label=1, seed=42)
    support_neg = sample_few_shot_examples(support_set, n=1, label=0, seed=42)

    pos_emb = embed_code(support_pos[0]["code1"] + "\n" + support_pos[0]["code2"]).squeeze(0)
    neg_emb = embed_code(support_neg[0]["code1"] + "\n" + support_neg[0]["code2"]).squeeze(0)

    for ex in test_examples:
        query = embed_code(ex["code1"] + "\n" + ex["code2"]).squeeze(0)
        sims = {
            0: F.cosine_similarity(query, neg_emb, dim=0).item(),
            1: F.cosine_similarity(query, pos_emb, dim=0).item()
        }
        pred = 1 if sims[1] > sims[0] else 0
        preds.append(pred)
        targets.append(ex["label"])

    return report_results(targets, preds, f"{dataset_name}_2shot")


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

    # Also get raw validation examples for threshold tuning
    val_examples = [full_data[i] for i in val_indices]



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
    support_set = load_multiple_datasets(SUPPORT_PATHS)

    thresholds = [0.7, 0.75, 0.8, 0.85, 0.9]
    best_threshold, best_avg_val_acc = None, -1

    for t in thresholds:
        val_preds, val_targets = [], []
        for ex in val_examples:
            emb1 = embed_code(ex["code1"])
            emb2 = embed_code(ex["code2"])
            sim = F.cosine_similarity(emb1, emb2).item()
            pred = 1 if sim >= t else 0
            val_preds.append(pred)
            val_targets.append(ex["label"])
        acc = accuracy_score(val_targets, val_preds)
        print(f"Threshold {t}: Validation Accuracy = {acc:.4f}")
        if acc > best_avg_val_acc:
            best_avg_val_acc = acc
            best_threshold = t

    print(f"\nBest threshold selected: {best_threshold:.2f} with validation accuracy {best_avg_val_acc:.4f}")

    # Final evaluation using best threshold
    overall_acc = 0
    for code_set in CLONE_DATASETS:
        test_path = f"src/data/rosetta/{code_set}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        print(f"\nDataset: {code_set}")
        acc1 = evaluate_zero_shot(test_examples, code_set, threshold=best_threshold)
        acc2a = evaluate_one_shot(test_examples, support_set, code_set, label=0)
        acc2b = evaluate_one_shot(test_examples, support_set, code_set, label=1)
        acc3 = evaluate_two_shot(test_examples, support_set, code_set)

        overall_acc = (acc1 + acc2a + acc2b + acc3) / 4


    final_avg = overall_acc / len(CLONE_DATASETS)
    print(f"\nFinal Average Accuracy with threshold {best_threshold}: {final_avg:.4f}")

if __name__ == "__main__":
    run()
