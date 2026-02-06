# ============================================================
# train_hierarchical_bilstm_crf.py
# Hierarchical BiLSTM + CRF for Rhetorical Role Classification
# ============================================================

import os
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix
)
from torchcrf import CRF
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================

EXPERIMENT_NAME = "hierarchical_bilstm_crf"

EMB_DIM = 128
WORD_HIDDEN = 128
SENT_HIDDEN = 128
MAX_TOKENS = 60

BATCH_SIZE = 1
EPOCHS = 15
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if DEVICE.type == "cuda":
    torch.cuda.empty_cache()

# ============================================================
# PATHS
# ============================================================

BASE_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", EXPERIMENT_NAME)
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", EXPERIMENT_NAME)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pth")

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
    """
    Each item = one document
    X : [num_sentences, MAX_TOKENS]
    y : [num_sentences]
    mask : [num_sentences]
    """

    def __init__(self, csv_path, vocab, label_encoder):
        df = pd.read_csv(csv_path)
        self.docs = []

        for _, group in df.groupby("Index"):
            sents = group["Text"].astype(str).tolist()
            labels = label_encoder.transform(group["Label"].tolist())
            self.docs.append((sents, labels))

        self.vocab = vocab

    def encode_sentence(self, sentence):
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokenize(sentence)]
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
# MODEL: HIERARCHICAL BiLSTM + CRF
# ============================================================

class HierarchicalBiLSTMCRF(nn.Module):

    def __init__(self, vocab_size, num_labels):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)

        self.word_lstm = nn.LSTM(
            EMB_DIM, WORD_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.sent_lstm = nn.LSTM(
            WORD_HIDDEN * 2, SENT_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.hidden2tag = nn.Linear(SENT_HIDDEN * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, X, y, mask):
        B, S, T = X.shape

        X = X.view(B * S, T)
        emb = self.embedding(X)

        word_out, _ = self.word_lstm(emb)
        sent_vec, _ = torch.max(word_out, dim=1)

        sent_vec = sent_vec.view(B, S, -1)
        doc_out, _ = self.sent_lstm(sent_vec)

        emissions = self.hidden2tag(doc_out)
        loss = -self.crf(emissions, y, mask=mask, reduction="mean")
        return loss

    def decode(self, X, mask):
        B, S, T = X.shape

        X = X.view(B * S, T)
        emb = self.embedding(X)

        word_out, _ = self.word_lstm(emb)
        sent_vec, _ = torch.max(word_out, dim=1)

        sent_vec = sent_vec.view(B, S, -1)
        doc_out, _ = self.sent_lstm(sent_vec)

        emissions = self.hidden2tag(doc_out)
        return self.crf.decode(emissions, mask=mask)

# ============================================================
# EXECUTION (TRAIN / EVAL ONLY)
# ============================================================

if __name__ == "__main__":

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # ---------------- BUILD VOCAB & LABELS ----------------

    train_df = pd.read_csv(TRAIN_CSV)

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for txt in train_df["Text"]:
        for w in tokenize(txt):
            if w not in vocab:
                vocab[w] = len(vocab)

    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["Label"])

    joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))
    joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.joblib"))

    # ---------------- DATA LOADERS ----------------

    train_ds = RhetoricalDocDataset(TRAIN_CSV, vocab, label_encoder)
    val_ds   = RhetoricalDocDataset(VAL_CSV, vocab, label_encoder)
    test_ds  = RhetoricalDocDataset(TEST_CSV, vocab, label_encoder)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, BATCH_SIZE, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, BATCH_SIZE, collate_fn=collate_fn)

    # ---------------- TRAINING SETUP ----------------

    model = HierarchicalBiLSTMCRF(len(vocab), len(label_encoder.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler()

    best_f1 = 0.0
    start_epoch = 0

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"]
        best_f1 = checkpoint["best_f1"]
        print(f"Resuming training from epoch {start_epoch + 1}")

    # ---------------- TRAINING LOOP ----------------

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for X, y, mask in progress:
            X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad()

            with autocast():
                loss = model(X, y, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=total_loss)

        # ---------------- VALIDATION ----------------

        model.eval()
        gold, pred = [], []

        with torch.no_grad():
            for X, y, mask in val_loader:
                X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
                preds = model.decode(X, mask)

                for p, gt, m in zip(preds, y, mask):
                    valid_len = m.sum().item()
                    gold.extend(gt[:valid_len].cpu())
                    pred.extend(p[:valid_len])

        f1 = f1_score(gold, pred, average="macro")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_f1": best_f1
        }, CHECKPOINT_PATH)

    print("Training complete.")
