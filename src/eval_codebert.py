import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score
import json
from tokenizers import Tokenizer

# Configs
CHECKPOINT = "microsoft/codebert-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-2
TRAIN_RATIO = 0.8

code_2_tokenizer = AutoTokenizer.from_pretrained("tokenizer_ckpt")

# Load tokenizer and embedding model
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
embedding_model = AutoModel.from_pretrained(CHECKPOINT).to(DEVICE)
embedding_model.eval()  # freeze the model

# Dataset class
class CloneDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s["code1"], s["code2"], torch.tensor(float(s["label"]))

# Get frozen embedding
def get_embedding(code, tokenizer):
    if not code.strip():
        code = "<pad>"
    inputs = tokenizer(code, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    inputs = {k: v.long() for k, v in inputs.items()}
    with torch.no_grad():
        outputs = embedding_model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :]  # CLS token
    return emb.squeeze(0).cpu()  # shape: [768]

# Classifier head
class CloneClassifier(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, emb1, emb2):
        combined = torch.cat((emb1, emb2), dim=1)
        return self.classifier(combined)

code_set = 'python_cobol_clones'

# Load datasets
train_dataset = CloneDataset(f"src/data/{code_set}_train.json")
test_dataset = CloneDataset(f"src/data/{code_set}_test.json")

# Split train into train/val
train_size = int(len(train_dataset) * TRAIN_RATIO)
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Initialize model
classifier = CloneClassifier().to(DEVICE)
optimizer = optim.AdamW(classifier.parameters(), lr=LR)
loss_fn = nn.BCEWithLogitsLoss()

# Training loop
for epoch in range(EPOCHS):
    classifier.train()
    running_loss = 0
    for code1s, code2s, labels in train_loader:
        emb1s = torch.stack([get_embedding(c, tokenizer) for c in code1s]).to(DEVICE)
        emb2s = torch.stack([get_embedding(c, code_2_tokenizer) for c in code2s]).to(DEVICE)
        logits = classifier(emb1s, emb2s).squeeze(1)
        loss = loss_fn(logits, labels.to(DEVICE))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Train Loss: {running_loss / len(train_loader):.4f}")

    # Validation
    classifier.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for code1s, code2s, labels in val_loader:
            emb1s = torch.stack([get_embedding(c, tokenizer) for c in code1s]).to(DEVICE)
            emb2s = torch.stack([get_embedding(c, code_2_tokenizer) for c in code2s]).to(DEVICE)
            logits = classifier(emb1s, emb2s).squeeze(1)
            preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
            val_preds.extend(preds)
            val_labels.extend(labels.tolist())

    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")

# Test evaluation
classifier.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for code1s, code2s, labels in test_loader:
        emb1s = torch.stack([get_embedding(c, tokenizer) for c in code1s]).to(DEVICE)
        emb2s = torch.stack([get_embedding(c, code_2_tokenizer) for c in code2s]).to(DEVICE)
        logits = classifier(emb1s, emb2s).squeeze(1)
        preds = (torch.sigmoid(logits) > 0.5).long().cpu().tolist()
        test_preds.extend(preds)
        test_labels.extend(labels.tolist())

test_acc = accuracy_score(test_labels, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")
