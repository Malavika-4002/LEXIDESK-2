import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving vocab and label encoder

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "rhetorical", "bilstm")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "rhetorical", "bilstm")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", "bilstm")  # path for model & vocab & encoder

for d in [EVAL_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

# -----------------------------
# DATASET
# -----------------------------
class RhetoricalDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.split()]
        return ids[:self.max_len] + [self.vocab["<PAD>"]] * max(0, self.max_len - len(ids))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encode(self.texts[idx]), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# -----------------------------
# MODEL
# -----------------------------
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return self.fc(context)

# -----------------------------
# LOAD DATA
# -----------------------------
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# Label encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df["Label"])
y_test = le.transform(test_df["Label"])
print("LabelEncoder mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# Build vocab
vocab = {"<PAD>": 0, "<UNK>": 1}
for sent in train_df["Text"]:
    for w in str(sent).split():
        if w not in vocab:
            vocab[w] = len(vocab)

train_ds = RhetoricalDataset(train_df["Text"].tolist(), y_train, vocab)
test_ds = RhetoricalDataset(test_df["Text"].tolist(), y_test, vocab)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMAttention(len(vocab), 128, 128, len(le.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# TRAIN
# -----------------------------
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# -----------------------------
# SAVE MODEL, VOCAB, LABEL ENCODER
# -----------------------------
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "bilstm_rhetorical_model.pt"))
joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"Model, vocab, and LabelEncoder saved in: {MODEL_DIR}")

# -----------------------------
# EVALUATE
# -----------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

metrics = {
    "macro_precision": precision_score(all_labels, all_preds, average="macro"),
    "macro_recall": recall_score(all_labels, all_preds, average="macro"),
    "macro_f1": f1_score(all_labels, all_preds, average="macro"),
    "accuracy": accuracy_score(all_labels, all_preds)
}

# Save metrics
pd.DataFrame([metrics]).to_csv(os.path.join(EVAL_DIR, "metrics.csv"), index=False)
with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Classification report
report = classification_report(all_labels, all_preds, target_names=[str(c) for c in le.classes_], digits=4)
with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("BiLSTM Rhetorical Classifier – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()

# Sample predictions
sample_texts = [
    "The High Court held that the losses could not be deducted.",
    "The appellant contended that the transactions were inter-state sales.",
    "For these reasons, the appeal is dismissed."
]

# Encode sample texts
def encode_samples(texts, vocab, max_len=100):
    encoded = []
    for t in texts:
        ids = [vocab.get(w, vocab["<UNK>"]) for w in t.split()]
        ids = ids[:max_len] + [vocab["<PAD>"]] * max(0, max_len - len(ids))
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)

model.eval()
with torch.no_grad():
    sample_vec = encode_samples(sample_texts, vocab).to(device)
    sample_preds = model(sample_vec).argmax(dim=1).cpu().numpy()
    sample_labels = le.inverse_transform(sample_preds)

pd.DataFrame({
    "sentence": sample_texts,
    "predicted_label": sample_labels
}).to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)

print("BiLSTM rhetorical classifier evaluation complete.")
print("Sample predictions saved in:", os.path.join(OUT_DIR, "sample_predictions.csv"))
