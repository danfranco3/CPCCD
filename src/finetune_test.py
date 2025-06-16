import json
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments

# Configuration
MODEL_NAME = "Salesforce/codet5p-220m"
OUTPUT_DIR = "results"
MAX_LENGTH = 2048
EPOCHS = 3
BATCH_SIZE = 1

device = torch.device("cuda")
torch.cuda.is_available = lambda: True  # Override to disable fp16 logic in Hugging Face
use_fp16 = True  # Make sure fp16 is disabled

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
        return {k: v.squeeze(0) for k, v in enc.items()}


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
model.to(device)

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
    predict_with_generate=True,
    logging_dir="logs",
    load_best_model_at_end=True,
    fp16=use_fp16,  # ensure fp16 is disabled
    no_cuda=True,   # force CPU usage
)


# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
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


# Example usage
example_code1 = "var mac = \"88:53:2E:67:07:BE\";\nfunction findmac(){\n\twindow.open(\"http://api.macvendors.com/\" + mac);\n}\nfindmac();"
example_code2 = "program rosettaCodeSHA256;\nuses\n  SysUtils, DCPsha256;\nvar\n  ros: String;\n  sha256 : TDCP_sha256;\n  digest : array[0..63] of byte;\n  i: Integer;\n  output: String;\nbegin\n  ros := 'Rosetta code';\n  sha256 := TDCP_sha256.Create(nil);\n  sha256.init;\n  sha256.UpdateStr(ros);\n  sha256.Final(digest);\n  output := '';\n  for i := 0 to 31 do begin\n    output := output + intToHex(digest[i], 2);\n  end;\n  writeln(lowerCase(output));\nend."
example_prompt = f"Are the following two code snippets functionally equivalent?\n\nCode1:\n{example_code1}\n\nCode2:\n{example_code2}\n\nAnswer:"
print("Model prediction:", predict_few_shot([test_dataset[0], test_dataset[1]], example_prompt))
