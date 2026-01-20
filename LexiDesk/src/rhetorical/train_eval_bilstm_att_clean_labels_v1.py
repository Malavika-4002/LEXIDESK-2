import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

# ============================================================
# EXPERIMENT CONFIG (CHANGE THIS FOR EVERY NEW RUN)
# ============================================================
EXPERIMENT_NAME = "bilstm_attn_cleanlabels_v1"
EMB_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 100
LR = 1e-3

# ============================================================
# PATHS
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR  = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", "bilstm")
EVAL_DIR  = os.path.join(BASE_DIR, "evaluation", "rhetorical", "bilstm")
OUT_DIR   = os.path.join(BASE_DIR, "outputs", "rhetorical", "bilstm")

for d in [MODEL_DIR, EVAL_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

# ============================================================
# LABEL CLEANING (CRITICAL FIX)
# ============================================================
INVALID_LABELS = {"None", "nan", "NaN", "", "NULL"}

def clean_df(df):
    df["Label"] = df["Label"].astype(str).str.strip()
    df = df[~df["Label"].isin(INVALID_LABELS)]
    return df.reset_index(drop=True)

# ============================================================
# DATASET
# ============================================================
class RhetoricalDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def encode(self, text):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.split()]
        ids = ids[:self.max_len]
        ids += [self.vocab["<PAD>"]] * (self.max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encode(self.texts[idx]), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long)
        )

# ============================================================
# MODEL
# ============================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (attn_weights * lstm_out).sum(dim=1)
        return self.fc(context)

# ============================================================
# LOAD & PREP DATA
# ============================================================
train_df = clean_df(pd.read_csv(TRAIN_CSV))
test_df  = clean_df(pd.read_csv(TEST_CSV))

le = LabelEncoder()
y_train = le.fit_transform(train_df["Label"])
y_test  = le.transform(test_df["Label"])

print("Label mapping:")
print(dict(zip(le.classes_, le.transform(le.classes_))))

# Build vocab (TRAIN ONLY)
vocab = {"<PAD>": 0, "<UNK>": 1}
for sent in train_df["Text"]:
    for w in str(sent).split():
        if w not in vocab:
            vocab[w] = len(vocab)

train_ds = RhetoricalDataset(train_df["Text"].tolist(), y_train, vocab, MAX_LEN)
test_ds  = RhetoricalDataset(test_df["Text"].tolist(), y_test, vocab, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BiLSTMAttention(
    vocab_size=len(vocab),
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    num_classes=len(le.classes_)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# ============================================================
# TRAIN
# ============================================================
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"[{EXPERIMENT_NAME}] Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss/len(train_loader):.4f}")

# ============================================================
# SAVE MODEL ARTIFACTS (NO OVERWRITE)
# ============================================================
torch.save(
    model.state_dict(),
    os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_model.pt")
)

joblib.dump(
    vocab,
    os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_vocab.joblib")
)

joblib.dump(
    le,
    os.path.join(MODEL_DIR, f"{EXPERIMENT_NAME}_label_encoder.joblib")
)

# ============================================================
# EVALUATION
# ============================================================
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y.numpy())

metrics = {
    "experiment": EXPERIMENT_NAME,
    "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
    "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    "accuracy": accuracy_score(all_labels, all_preds)
}

pd.DataFrame([metrics]).to_csv(
    os.path.join(EVAL_DIR, f"{EXPERIMENT_NAME}_metrics.csv"),
    index=False
)

report = classification_report(
    all_labels,
    all_preds,
    target_names=le.classes_,
    digits=4,
    zero_division=0
)

with open(os.path.join(EVAL_DIR, f"{EXPERIMENT_NAME}_classification_report.txt"), "w") as f:
    f.write(report)

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="Blues"
)
plt.title(f"Confusion Matrix – {EXPERIMENT_NAME}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, f"{EXPERIMENT_NAME}_confusion_matrix.png"))
plt.close()

print(f"Experiment completed safely: {EXPERIMENT_NAME}")
