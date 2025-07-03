import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, get_scheduler
from torch.optim import AdamW
from sklearn.metrics import classification_report, accuracy_score
import os
from tqdm import tqdm

from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets
from code_clone_pkg.dataset import CodeCloneDataset

# Configuration
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "results/codebert_finetune"
MAX_LENGTH = 512
EPOCHS = 8
BATCH_SIZE = 2
LR = 2e-5
THRESHOLD = 0.9
CLONE_DATASETS = ['python_cobol', 'java_fortran', 'js_pascal']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CodeBERTClassifier(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 2)  # only one encoding now

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        logits = self.classifier(out)
        return logits


def tokenize_pair(tokenizer, code1, code2):
    code1_inputs = tokenizer(code1, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding='max_length')
    code2_inputs = tokenizer(code2, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding='max_length')
    return {k: v.squeeze(0).to(device) for k, v in code1_inputs.items()}, {k: v.squeeze(0).to(device) for k, v in code2_inputs.items()}


def get_pair_embedding(code1, code2, tokenizer, model):
    inputs1 = tokenizer(code1, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH).to(device)
    inputs2 = tokenizer(code2, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LENGTH).to(device)
    with torch.no_grad():
        h1 = model(**inputs1).last_hidden_state[:, 0]
        h2 = model(**inputs2).last_hidden_state[:, 0]
        return torch.cat([h1, h2], dim=-1)


def finetune(model, train_dataset, val_dataset):
    model.train()
    optimizer = AdamW(model.parameters(), lr=LR)
    num_training_steps = EPOCHS * len(train_dataset) // BATCH_SIZE
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                logits = model(input_ids, attention_mask)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "codebert_finetuned.pt"))



def evaluate_model_cosine(model, tokenizer, test_examples, output_path):
    model.eval()
    preds, targets, outputs = [], [], []

    for ex in test_examples:
        emb1 = get_pair_embedding(ex["code1"], ex["code2"], tokenizer, model.encoder)
        sim = F.cosine_similarity(emb1[:, :emb1.shape[1] // 2], emb1[:, emb1.shape[1] // 2:]).item()
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

    for i, ex in enumerate(test_examples):
        if shots == 1:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i)
        elif shots == 2:
            support = sample_few_shot_examples(support_set, n=1, label=1, seed=i) + \
                      sample_few_shot_examples(support_set, n=1, label=0, seed=i+100)

        query_emb = get_pair_embedding(ex["code1"], ex["code2"], tokenizer, model.encoder)

        # === Average similarity by class ===
        sim_by_class = {0: [], 1: []}
        for s in support:
            support_emb = get_pair_embedding(s["code1"], s["code2"], tokenizer, model.encoder)
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
    model = CodeBERTClassifier(MODEL_NAME).to(device)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Running for dataset: {code_set} ===")

        train_dataset = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)
        with open(f"src/data/rosetta/{code_set}_test.json") as f:
            test_examples = json.load(f)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_data, val_data = random_split(train_dataset, [train_size, val_size])

        finetune(model, train_data, val_data)

        evaluate_model_cosine(model, tokenizer, test_examples, f"{OUTPUT_DIR}/{code_set}_zero_shot.json")

        support_set = load_multiple_datasets(["src/data/codeNet/ruby_go_test.json"])
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, f"{OUTPUT_DIR}/{code_set}_one_shot.json", shots=1)
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, f"{OUTPUT_DIR}/{code_set}_two_shot.json", shots=2)


if __name__ == "__main__":
    run()
