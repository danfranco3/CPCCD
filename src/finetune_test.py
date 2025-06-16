import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from sklearn.metrics import classification_report, accuracy_score

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"
OUTPUT_DIR = "results"
MAX_LENGTH = 700
EPOCHS = 3
BATCH_SIZE = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = torch.cuda.is_available()  # Enable fp16 only if CUDA is available

# Dataset definition
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
            "input_ids": enc["input_ids"].squeeze(),  # [seq_len]
            "attention_mask": enc["attention_mask"].squeeze(),  # [seq_len]
            "labels": labels.squeeze()  # [seq_len]
        }



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(device)

if torch.cuda.is_available():
    model.gradient_checkpointing_enable()

# Load datasets
code_set = 'python_cobol_clones'

train_dataset = CodeCloneDataset(f"src/data/{code_set}_train.json", tokenizer)
test_dataset = CodeCloneDataset(f"src/data/{code_set}_test.json", tokenizer)

# Split train into train/val
train_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,  # optional but helpful
    padding=True,  # dynamic padding
    return_tensors="pt",
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_accumulation_steps=4,
    predict_with_generate=True,
    logging_dir="logs",
    load_best_model_at_end=True,
    fp16=use_fp16,
    no_cuda=not torch.cuda.is_available(),
)


# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save final model
trainer.save_model(f"{OUTPUT_DIR}/final_model")

# Prediction function
def predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Few-shot prediction prompt builder
def few_shot_prompt(example_pairs, target_pair):
    prompt = "Are the following code snippet pairs functionally equivalent?\n\n"
    for ex in example_pairs:
        label = "Yes" if ex["label"] == 1 else "No"
        prompt += f"Code1:\n{ex['code1']}\n\nCode2:\n{ex['code2']}\n\nAnswer: {label}\n\n"
    prompt += f"Code1:\n{target_pair['code1']}\n\nCode2:\n{target_pair['code2']}\n\nAnswer:"
    return prompt

# Few-shot predictor
def predict_few_shot(example_pairs, target_pair):
    prompt = few_shot_prompt(example_pairs, target_pair)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LENGTH).to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def evaluate_model(model, tokenizer, test_loader, max_new_tokens=10):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)  # shape: (1, seq_len)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            labels = batch["labels"].to(device)

            # Generate prediction
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )

            pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip().lower()
            true = tokenizer.decode(labels[0], skip_special_tokens=True).strip().lower()

            # Normalize to 'yes'/'no'
            pred_label = 1 if 'yes' in pred else 0
            true_label = 1 if 'yes' in true else 0

            preds.append(pred_label)
            targets.append(true_label)

    # Report
    print("\nEvaluation Results:")
    print(classification_report(targets, preds, target_names=["Non-clone", "Clone"]))
    print(f"Accuracy: {accuracy_score(targets, preds):.4f}")

evaluate_model(model, tokenizer, test_loader)

