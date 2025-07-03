import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer, AutoModel, AdamW, get_scheduler
from sklearn.metrics import classification_report, accuracy_score
import os
from tqdm import tqdm

from code_clone_pkg.data_utils import sample_few_shot_examples, load_multiple_datasets
from code_clone_pkg.dataset import CodeCloneDataset
from code_clone_pkg.prompts import few_shot_prompt

# Configuration
MODEL_NAME = "microsoft/codebert-base"
OUTPUT_DIR = "results/codebert_finetune"
MAX_LENGTH = 1400
EPOCHS = 10
BATCH_SIZE = 3
LR = 2e-5
THRESHOLD = 0.9
CLONE_DATASETS = [
    'python_cobol',
    'java_fortran',
    'js_pascal'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Finetunable model with classification head
class CodeBERTClassifier(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size * 2, 2)  # binary classifier

    def forward(self, code1_inputs, code2_inputs):
        out1 = self.encoder(**code1_inputs).last_hidden_state[:, 0]  # [CLS]
        out2 = self.encoder(**code2_inputs).last_hidden_state[:, 0]
        combined = torch.cat([out1, out2], dim=1)
        logits = self.classifier(combined)
        return logits

def tokenize_pair(tokenizer, code1, code2):
    code1_inputs = tokenizer(code1, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding='max_length')
    code2_inputs = tokenizer(code2, return_tensors="pt", max_length=MAX_LENGTH, truncation=True, padding='max_length')
    return {k: v.squeeze(0).to(device) for k, v in code1_inputs.items()}, {k: v.squeeze(0).to(device) for k, v in code2_inputs.items()}

def get_mean_embedding(text, tokenizer, model):
    encoded = tokenizer(
        text,
        return_tensors='pt',
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length'
    ).to(device)

    with torch.no_grad():
        output = model(**encoded)
        token_embeddings = output.last_hidden_state
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        count = attention_mask.sum(dim=1)
        mean_embedding = summed / count
    return mean_embedding


def finetune(model, tokenizer, train_dataset, val_dataset):
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
            code1_inputs, code2_inputs = tokenize_pair(tokenizer, batch['code1'], batch['code2'])
            labels = batch['labels'].to(device)

            logits = model(code1_inputs, code2_inputs)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

        # === Validation phase ===
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                code1_inputs, code2_inputs = tokenize_pair(tokenizer, batch['code1'], batch['code2'])
                labels = batch['labels'].to(device)

                logits = model(code1_inputs, code2_inputs)
                loss = loss_fn(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().tolist())
                val_targets.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_targets, val_preds)
        print(f"[Epoch {epoch+1}] Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "codebert_finetuned.pt"))


def evaluate_model_cosine(model, tokenizer, test_examples, output_path):
    model.eval()
    preds, targets, outputs = [], [], []

    for ex in test_examples:
        emb1 = get_mean_embedding(ex["code1"], tokenizer, model.encoder)
        emb2 = get_mean_embedding(ex["code2"], tokenizer, model.encoder)

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


def predict_one_or_two_shot(model, tokenizer, support_set, test_examples, code_set, output_path, shots=1):
    model.eval()
    preds, targets, outputs = [], [], []

    with torch.no_grad():
        for i, ex in enumerate(test_examples):
            if shots == 1:
                support = sample_few_shot_examples(support_set, n=1, label=1, seed=i)
            elif shots == 2:
                support = sample_few_shot_examples(support_set, n=1, label=1, seed=i) + \
                          sample_few_shot_examples(support_set, n=1, label=0, seed=i+100)

            emb_ex = get_mean_embedding(ex["code1"] + ex["code2"], tokenizer, model.encoder)

            sims = []
            for s in support:
                emb_s = get_mean_embedding(s["code1"] + s["code2"], tokenizer, model.encoder)
                sim = F.cosine_similarity(emb_ex, emb_s).item()
                sims.append(sim)

            avg_sim = sum(sims) / len(sims)
            pred = 1 if avg_sim >= THRESHOLD else 0

            preds.append(ex["label"])
            targets.append(pred)
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


def run():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = CodeBERTClassifier(MODEL_NAME).to(device)

    for code_set in CLONE_DATASETS:
        print(f"\n=== Running for dataset: {code_set} ===")

        train_dataset = CodeCloneDataset("src/data/combined_train.json", tokenizer, MAX_LENGTH)
        test_path = f"src/data/rosetta/{code_set}_test.json"
        with open(test_path) as f:
            test_examples = json.load(f)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_data, val_data = random_split(train_dataset, [train_size, val_size])

        finetune(model, tokenizer, train_data, val_data)

        zero_shot_output = f"{OUTPUT_DIR}/{code_set}_zero_shot.json"
        evaluate_model_cosine(model, tokenizer, test_examples, zero_shot_output)

        support_paths = [
            "src/data/codeNet/ruby_go_test.json",
        ]
        
        support_set = load_multiple_datasets(support_paths)

        one_shot_output = f"{OUTPUT_DIR}/{code_set}_one_shot.json"
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, code_set, one_shot_output, shots=1)

        two_shot_output = f"{OUTPUT_DIR}/{code_set}_two_shot.json"
        predict_one_or_two_shot(model, tokenizer, support_set, test_examples, code_set, two_shot_output, shots=2)


if __name__ == "__main__":
    run()
