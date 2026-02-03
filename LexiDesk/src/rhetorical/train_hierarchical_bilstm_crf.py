# ============================================================
# IMPORTS
# ============================================================
import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# ============================================================
# CONFIG
# ============================================================
EXPERIMENT_NAME = "hierarchical_bilstm_safe"

EMB_DIM = 128
SENT_HIDDEN = 128
DOC_HIDDEN = 128
MAX_TOKENS = 60
BATCH_SIZE = 1
EPOCHS = 15
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS (UNCHANGED)
# ============================================================
BASE_DIR = "/content/drive/MyDrive/LexiDesk"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", EXPERIMENT_NAME)
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", EXPERIMENT_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

# ============================================================
# TOKENIZATION
# ============================================================
def tokenize(text):
    text = str(text).lower()
    return text.replace(".", " . ").replace(",", " , ").split()

# ============================================================
# DATASET
# ============================================================
class RhetoricalDocDataset(Dataset):
    def __init__(self, csv_path, vocab, le):

        df = pd.read_csv(csv_path)

        self.docs = []

        for _, g in df.groupby("Index"):
            sents = g["Text"].astype(str).tolist()
            labels = le.transform(g["Label"].tolist())
            self.docs.append((sents, labels))

        self.vocab = vocab

    def encode(self, sent):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize(sent)]
        ids = ids[:MAX_TOKENS]
        return ids + [0] * (MAX_TOKENS - len(ids))

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        sents, labels = self.docs[idx]
        X = torch.tensor([self.encode(s) for s in sents])
        y = torch.tensor(labels)
        mask = torch.ones(len(labels), dtype=torch.bool)
        return X, y, mask

def collate_fn(batch):
    max_s = max(x[0].shape[0] for x in batch)
    Xs, ys, ms = [], [], []
    for X, y, m in batch:
        pad = max_s - X.shape[0]
        if pad:
            X = torch.cat([X, torch.zeros(pad, MAX_TOKENS, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])
        Xs.append(X)
        ys.append(y)
        ms.append(m)

    return torch.stack(Xs), torch.stack(ys), torch.stack(ms)

# ============================================================
# MODEL — MEMORY SAFE HIERARCHICAL BiLSTM
# ============================================================
class HierarchicalBiLSTM(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)
        self.sent_lstm = nn.LSTM(EMB_DIM, SENT_HIDDEN, bidirectional=True, batch_first=True)
        self.doc_lstm = nn.LSTM(SENT_HIDDEN * 2, DOC_HIDDEN, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(DOC_HIDDEN * 2, num_labels)


    def forward(self, X):
        B, S, T = X.shape
        sent_vectors = []

        for i in range(S):
            emb = self.embedding(X[:, i])
            out, _ = self.sent_lstm(emb)
            sent_vectors.append(out.mean(dim=1))

        sent_vec = torch.stack(sent_vectors, dim=1)
        doc_out, _ = self.doc_lstm(sent_vec)
        return self.fc(doc_out)

# ============================================================
# BUILD VOCAB & LABELS
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)

vocab = {"<PAD>": 0, "<UNK>": 1}
for t in train_df["Text"]:
    for w in tokenize(t):
        if w not in vocab:
            vocab[w] = len(vocab)

le = LabelEncoder()
le.fit(train_df["Label"])

joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

train_ds = RhetoricalDocDataset(TRAIN_CSV, vocab, le)
val_ds   = RhetoricalDocDataset(VAL_CSV, vocab, le)
test_ds  = RhetoricalDocDataset(TEST_CSV, vocab, le)

train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, BATCH_SIZE, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, BATCH_SIZE, collate_fn=collate_fn)

# ============================================================
# TRAINING (AMP ENABLED)
# ============================================================
model = HierarchicalBiLSTM(len(vocab), len(le.classes_)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

best_f1 = 0.0


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y, m in train_loader:
        X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits = model(X)
            mask = m.view(-1)
            loss = criterion(
                logits.view(-1, logits.size(-1))[mask],
                y.view(-1)[mask]
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    # ---------- VALIDATION ----------
    model.eval()
    gold, pred = [], []

    with torch.no_grad():
        for X, y, m in val_loader:
            X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            p = model(X).argmax(-1)
            for pi, yi, mi in zip(p, y, m):
                gold.extend(yi[mi].cpu())
                pred.extend(pi[mi].cpu())

    f1 = f1_score(gold, pred, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={total_loss:.4f} | Val Macro-F1={f1:.4f}")

# ============================================================
# TEST EVALUATION
# ============================================================
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt")))
model.eval()

gold, pred = [], []

with torch.no_grad():
    for X, y, m in test_loader:
        X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        p = model(X).argmax(-1)
        for pi, yi, mi in zip(p, y, m):
            gold.extend(yi[mi].cpu())
            pred.extend(pi[mi].cpu())

print("\nTEST RESULTS")
print(classification_report(gold, pred, target_names=le.classes_))

cm = confusion_matrix(gold, pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()
t(gold, pred, target_names=le.classes_))

cm = confusion_matrix(gold, pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()

print("Training and evaluation completed successfully.")
