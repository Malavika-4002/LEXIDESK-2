# ============================================================
# Hierarchical BiLSTM + CRF (Resumable, Summarization-Ready)
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
from torchcrf import CRF
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)

from torch.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================

EXPERIMENT_NAME = "hierarchical_bilstm_crf_v3"

EMB_DIM = 128
WORD_HIDDEN = 128
SENT_HIDDEN = 128
MAX_TOKENS = 60

BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 4
EPOCHS = 15
LR = 5e-4

WORD_DROPOUT = 0.2
SENT_DROPOUT = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS (UNCHANGED)
# ============================================================

BASE_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk"
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", EXPERIMENT_NAME)
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", EXPERIMENT_NAME)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pt")

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

    def __init__(self, csv_path, vocab, label_encoder):
        df = pd.read_csv(csv_path)
        self.docs = []

        for _, g in df.groupby("Index"):
            sents = g["Text"].astype(str).tolist()
            labels = label_encoder.transform(g["Label"].tolist())
            self.docs.append((sents, labels))

        self.vocab = vocab

    def encode_sentence(self, sent):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize(sent)]
        ids = ids[:MAX_TOKENS]
        return ids + [0] * (MAX_TOKENS - len(ids))

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        sents, labels = self.docs[idx]
        X = torch.tensor([self.encode_sentence(s) for s in sents])
        y = torch.tensor(labels)
        mask = torch.ones(len(labels), dtype=torch.bool)
        return X, y, mask

def collate_fn(batch):
    max_sents = max(x[0].shape[0] for x in batch)
    Xs, ys, masks = [], [], []

    for X, y, m in batch:
        pad = max_sents - X.shape[0]
        if pad > 0:
            X = torch.cat([X, torch.zeros(pad, MAX_TOKENS, dtype=torch.long)])
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)])
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)])

        Xs.append(X)
        ys.append(y)
        masks.append(m)

    return torch.stack(Xs), torch.stack(ys), torch.stack(masks)

# ============================================================
# MODEL
# ============================================================

class WordAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, H):
        scores = self.attn(H).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(H * weights.unsqueeze(-1), dim=1)

class HierarchicalBiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, num_labels, class_weights):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)

        self.word_lstm = nn.LSTM(
            EMB_DIM, WORD_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.word_attention = WordAttention(WORD_HIDDEN)
        self.word_dropout = nn.Dropout(WORD_DROPOUT)

        self.sent_lstm = nn.LSTM(
            WORD_HIDDEN * 2, SENT_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.sent_dropout = nn.Dropout(SENT_DROPOUT)

        self.hidden2tag = nn.Linear(SENT_HIDDEN * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

        self.register_buffer("class_weights", class_weights)

    def forward(self, X, y, mask):
        B, S, T = X.shape

        X = X.view(B * S, T)
        emb = self.embedding(X)

        word_out, _ = self.word_lstm(emb)
        sent_vec = self.word_attention(word_out)
        sent_vec = self.word_dropout(sent_vec)

        sent_vec = sent_vec.view(B, S, -1)
        doc_out, _ = self.sent_lstm(sent_vec)
        doc_out = self.sent_dropout(doc_out)

        emissions = self.hidden2tag(doc_out)
        emissions = emissions * self.class_weights

        loss = -self.crf(emissions, y, mask=mask, reduction="mean")
        return loss

    def decode(self, X, mask):
        B, S, T = X.shape

        X = X.view(B * S, T)
        emb = self.embedding(X)
        word_out, _ = self.word_lstm(emb)

        sent_vec = self.word_attention(word_out)
        sent_vec = sent_vec.view(B, S, -1)

        doc_out, _ = self.sent_lstm(sent_vec)
        emissions = self.hidden2tag(doc_out)

        return self.crf.decode(emissions, mask=mask)

# ============================================================
# VOCAB & LABELS
# ============================================================

train_df = pd.read_csv(TRAIN_CSV)

vocab = {"<PAD>": 0, "<UNK>": 1}
for txt in train_df["Text"]:
    for w in tokenize(txt):
        if w not in vocab:
            vocab[w] = len(vocab)

label_encoder = LabelEncoder()
label_encoder.fit(train_df["Label"])

label_counts = train_df["Label"].value_counts().sort_index()
class_weights = torch.tensor(
    1.0 / np.sqrt(label_counts.values),
    dtype=torch.float
).to(DEVICE)

joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

# ============================================================
# LOADERS
# ============================================================

train_loader = DataLoader(
    RhetoricalDocDataset(TRAIN_CSV, vocab, label_encoder),
    BATCH_SIZE, shuffle=True, collate_fn=collate_fn
)

val_loader = DataLoader(
    RhetoricalDocDataset(VAL_CSV, vocab, label_encoder),
    BATCH_SIZE, collate_fn=collate_fn
)

test_loader = DataLoader(
    RhetoricalDocDataset(TEST_CSV, vocab, label_encoder),
    BATCH_SIZE, collate_fn=collate_fn
)

# ============================================================
# TRAINING SETUP
# ============================================================

model = HierarchicalBiLSTMCRF(
    len(vocab),
    len(label_encoder.classes_),
    class_weights
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=2, factor=0.5
)

scaler = GradScaler()

start_epoch = 0
best_f1 = 0.0

# ============================================================
# RESUME FROM CHECKPOINT (IF EXISTS)
# ============================================================

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming training from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])

    start_epoch = checkpoint["epoch"] + 1
    best_f1 = checkpoint["best_f1"]

    print(f"Resumed from epoch {start_epoch}, best F1 = {best_f1:.4f}")

# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for step, (X, y, mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        with autocast(device_type="cuda"):
            loss = model(X, y, mask) / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()

    # ================= VALIDATION =================
    model.eval()
    gold, pred = [], []

    with torch.no_grad():
        for X, y, mask in val_loader:
            X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            preds = model.decode(X, mask)

            for p, gt, m in zip(preds, y, mask):
                L = m.sum().item()
                gold.extend(gt[:L].cpu())
                pred.extend(p[:L])

    f1 = f1_score(gold, pred, average="macro")
    scheduler.step(f1)

    print(f"Epoch {epoch+1} | Loss={total_loss:.2f} | Macro-F1={f1:.4f}")

    # ================= SAVE CHECKPOINT =================
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "best_f1": best_f1
    }, CHECKPOINT_PATH)

    # ================= SAVE BEST MODEL =================
    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), BEST_MODEL_PATH)

# ============================================================
# TEST EVALUATION
# ============================================================

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

gold, pred = [], []

with torch.no_grad():
    for X, y, mask in test_loader:
        X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        preds = model.decode(X, mask)

        for p, gt, m in zip(preds, y, mask):
            L = m.sum().item()
            gold.extend(gt[:L].cpu())
            pred.extend(p[:L])

print("\nTEST RESULTS")
print("Macro-F1:", f1_score(gold, pred, average="macro"))
print("Precision:", precision_score(gold, pred, average="macro"))
print("Recall:", recall_score(gold, pred, average="macro"))
print("Accuracy:", accuracy_score(gold, pred))

print("Training, checkpointing, and evaluation completed successfully.")
