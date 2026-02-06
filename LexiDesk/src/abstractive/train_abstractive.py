# ============================================================
# train_abstractive.py
# From-scratch Abstractive Summarizer Training
# ============================================================

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from src.abstractive.vocab import Vocabulary
from src.abstractive.dataset_abstractive import AbstractiveDataset
from src.abstractive.model_seq2seq import Seq2Seq  # STEP 1 MODEL

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk"

TRAIN_JSONL = os.path.join(
    BASE_DIR, "data/processed/abstractive/train_abstractive.jsonl"
)

MODEL_DIR = os.path.join(BASE_DIR, "models/abstractive")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.json")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "checkpoint.pt")

BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-3

MAX_SRC_LEN = 800
MAX_TGT_LEN = 200

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# BUILD VOCAB
# ============================================================

print("🔹 Building vocabulary...")

texts = []
with open(TRAIN_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        texts.append(obj["input"])
        texts.append(obj["target"])

vocab = Vocabulary(min_freq=2)
vocab.build(texts)

os.makedirs(MODEL_DIR, exist_ok=True)
with open(VOCAB_PATH, "w", encoding="utf-8") as f:
    json.dump(vocab.word2idx, f)

print(f"✅ Vocabulary size: {vocab.size}")

# ============================================================
# DATASET & LOADER
# ============================================================

train_dataset = AbstractiveDataset(
    TRAIN_JSONL,
    vocab,
    max_src_len=MAX_SRC_LEN,
    max_tgt_len=MAX_TGT_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# ============================================================
# MODEL
# ============================================================

model = Seq2Seq(
    vocab_size=vocab.size,
    pad_idx=vocab.pad_idx
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)

scaler = GradScaler()

start_epoch = 0

if os.path.exists(CHECKPOINT_PATH):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    start_epoch = ckpt["epoch"]
    print(f"🔁 Resuming from epoch {start_epoch}")

# ============================================================
# TRAINING LOOP
# ============================================================

print("\n🚀 Starting Abstractive Training...\n")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    total_loss = 0.0

    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in progress:
        src = batch["src"].to(DEVICE)
        src_len = batch["src_len"].to(DEVICE)
        tgt = batch["tgt"].to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            output = model(src, src_len, tgt[:, :-1])
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=total_loss)

    torch.save({
        "epoch": epoch + 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }, CHECKPOINT_PATH)

    print(f"✅ Epoch {epoch+1} completed. Loss: {total_loss:.4f}")

print("\n🎯 Abstractive training complete.")
