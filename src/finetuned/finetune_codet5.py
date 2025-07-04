import json
import torch
from torch.utils.data import DataLoader, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.metrics import classification_report, accuracy_score
import os

from code_clone_pkg.dataset import CodeCloneDataset

# CONFIG
MODEL_NAME   = "Salesforce/codet5p-220m"
OUTPUT_DIR   = "results/finetune_cls"
MAX_LENGTH   = 512
EPOCHS       = 8
BATCH_SIZE   = 8
CLONE_DATASETS = [
    "python_cobol",
    "java_fortran",
    "js_pascal"
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# METRICS 
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["Non-clone","Clone"], output_dict=True, zero_division=0)
    # return overall plus clone-class metrics
    return {
        "accuracy": acc,
        "precision_clone": report["Clone"]["precision"],
        "recall_clone":    report["Clone"]["recall"],
        "f1_clone":        report["Clone"]["f1-score"],
    }


# EVALUATION
def evaluate_model(model, tokenizer, test_dataset, raw_examples, output_path):
    model.eval()
    preds, targets = [], []
    outputs = []
    loader = DataLoader(test_dataset, batch_size=1)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = {k:v.to(device) for k,v in batch.items()}
            logits = model(**batch).logits
            pred_label = logits.argmax(dim=1).item()
            true_label = batch["labels"].item()

            preds.append(pred_label)
            targets.append(true_label)
            outputs.append({
                "code1":      raw_examples[i]["code1"],
                "code2":      raw_examples[i]["code2"],
                "pred_label": pred_label,
                "true_label": true_label
            })

    print("\nEvaluation Results:")
    print(classification_report(targets, preds, target_names=["Non-clone","Clone"], zero_division=0))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(outputs, f, indent=2)
    print(f"Detailed results saved to {output_path}")


# MAIN
def run():
    # Load tokenizer & model (classification head attached)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    ).to(device)

    # Prepare datasets
    train_full = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)
    train_size = int(0.8 * len(train_full))
    val_size   = len(train_full) - train_size
    train_ds, val_ds = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

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
        fp16=torch.cuda.is_available(),
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Fine-tune
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/codet5p_cls")

    # Evaluate on each test set
    for code_set in CLONE_DATASETS:
        print(f"\n=== Evaluating on {code_set} ===")
        test_path = f"src/data/rosetta/{code_set}_test.json"
        test_ds = CodeCloneDataset(test_path, tokenizer, MAX_LENGTH)
        with open(test_path) as f:
            raw = json.load(f)

        out_path = f"{OUTPUT_DIR}/codet5p_cls/{code_set}_eval.json"
        evaluate_model(model, tokenizer, test_ds, raw, out_path)


if __name__ == "__main__":
    run()
