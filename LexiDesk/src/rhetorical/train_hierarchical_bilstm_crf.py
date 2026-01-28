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
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# ============================================================
# EXPERIMENT CONFIG
# ============================================================
EXPERIMENT_NAME = "hierarchical_bilstm_crf_v1"

EMB_DIM = 128
SENT_HIDDEN = 128
DOC_HIDDEN = 128
MAX_TOKENS = 80
BATCH_SIZE = 2          # documents per batch
EPOCHS = 15
LR = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# PATHS (VS CODE - WINDOWS)
# ============================================================
BASE_DIR = r"C:\Users\acer\Downloads\LexiDesk-2\LexiDesk"

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")

MODEL_DIR = os.path.join(
    BASE_DIR, "models", "rhetorical", EXPERIMENT_NAME
)

EVAL_DIR = os.path.join(
    BASE_DIR, "evaluation", EXPERIMENT_NAME
)

for d in [MODEL_DIR, EVAL_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

# ============================================================
# TOKENIZATION (SIMPLE, STABLE)
# ============================================================
def tokenize(text):
    text = str(text).lower()
    text = text.replace(".", " . ").replace(",", " , ")
    return text.split()

# ============================================================
# DATASET (DOCUMENT-LEVEL)
# ============================================================
class RhetoricalDocDataset(Dataset):
    def __init__(self, csv_path, vocab, label_encoder):
        df = pd.read_csv(csv_path)
        df["Text"] = df["Text"].astype(str)

        self.docs = []
        for _, group in df.groupby("Index"):
            sentences = group["Text"].tolist()
            labels = label_encoder.transform(group["Label"].tolist())
            self.docs.append((sentences, labels))

        self.vocab = vocab

    def encode_sentence(self, sent):
        tokens = tokenize(sent)
        ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in tokens]
        ids = ids[:MAX_TOKENS]
        ids += [self.vocab["<PAD>"]] * (MAX_TOKENS - len(ids))
        return ids

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        sentences, labels = self.docs[idx]
        X = torch.tensor([self.encode_sentence(s) for s in sentences], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        mask = torch.ones(len(labels), dtype=torch.bool)
        return X, y, mask

def collate_fn(batch):
    max_sents = max(item[0].shape[0] for item in batch)

    Xs, ys, masks = [], [], []
    for X, y, m in batch:
        pad = max_sents - X.shape[0]
        if pad > 0:
            X = torch.cat([X, torch.zeros(pad, MAX_TOKENS, dtype=torch.long)], dim=0)
            y = torch.cat([y, torch.zeros(pad, dtype=torch.long)], dim=0)
            m = torch.cat([m, torch.zeros(pad, dtype=torch.bool)], dim=0)
        Xs.append(X)
        ys.append(y)
        masks.append(m)

    return torch.stack(Xs), torch.stack(ys), torch.stack(masks)

# ============================================================
# MODEL
# ============================================================
class HierarchicalBiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, EMB_DIM, padding_idx=0)

        self.sent_lstm = nn.LSTM(
            EMB_DIM, SENT_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.doc_lstm = nn.LSTM(
            SENT_HIDDEN * 2, DOC_HIDDEN,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(DOC_HIDDEN * 2, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, X, y=None, mask=None):
        B, S, T = X.shape

        X = X.view(B * S, T)
        emb = self.embedding(X)
        sent_out, _ = self.sent_lstm(emb)
        sent_vec = sent_out.mean(dim=1)
        sent_vec = sent_vec.view(B, S, -1)

        doc_out, _ = self.doc_lstm(sent_vec)
        emissions = self.fc(doc_out)

        if y is not None:
            loss = -self.crf(emissions, y, mask=mask, reduction="mean")
            return loss
        else:
            return self.crf.decode(emissions, mask=mask)

# ============================================================
# BUILD VOCAB & LABELS (TRAIN ONLY)
# ============================================================
train_df = pd.read_csv(TRAIN_CSV)

vocab = {"<PAD>": 0, "<UNK>": 1}
for txt in train_df["Text"]:
    for w in tokenize(txt):
        if w not in vocab:
            vocab[w] = len(vocab)

le = LabelEncoder()
le.fit(train_df["Label"])

joblib.dump(vocab, os.path.join(MODEL_DIR, "vocab.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))

train_ds = RhetoricalDocDataset(TRAIN_CSV, vocab, le)
val_ds   = RhetoricalDocDataset(VAL_CSV, vocab, le)
test_ds  = RhetoricalDocDataset(TEST_CSV, vocab, le)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# ============================================================
# TRAINING WITH FULL CHECKPOINTING
# ============================================================
model = HierarchicalBiLSTM_CRF(len(vocab), len(le.classes_)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_f1 = 0.0
start_epoch = 0
last_ckpt = os.path.join(MODEL_DIR, "checkpoint_last.pt")

if os.path.exists(last_ckpt):
    ckpt = torch.load(last_ckpt)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    best_f1 = ckpt["best_f1"]
    start_epoch = ckpt["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    epoch_loss = 0.0

    for X, y, m in train_loader:
        X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        optimizer.zero_grad()
        loss = model(X, y, m)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y, m in val_loader:
            X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            preds = model(X, mask=m)
            for p, t, mask in zip(preds, y.cpu().numpy(), m.cpu().numpy()):
                all_preds.extend(p)
                all_labels.extend(t[mask])

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Save rolling + epoch checkpoint
    ckpt = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_f1": best_f1
    }

    torch.save(ckpt, last_ckpt)
    torch.save(ckpt, os.path.join(MODEL_DIR, f"checkpoint_epoch_{epoch}.pt"))

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss={epoch_loss:.4f} | Val Macro F1={macro_f1:.4f}")

# Save final model
torch.save(model.state_dict(), os.path.join(MODEL_DIR, "final_model.pt"))

# ============================================================
# TEST EVALUATION
# ============================================================
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt")))
model.eval()

all_preds, all_labels = [], []

with torch.no_grad():
    for X, y, m in test_loader:
        X, y, m = X.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
        preds = model(X, mask=m)
        for p, t, mask in zip(preds, y.cpu().numpy(), m.cpu().numpy()):
            all_preds.extend(p)
            all_labels.extend(t[mask])

# Metrics
labels_no_none = [i for i, l in enumerate(le.classes_) if l != "None"]

metrics = {
    "macro_f1_all": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    "macro_f1_no_none": f1_score(
        all_labels, all_preds,
        labels=labels_no_none,
        average="macro",
        zero_division=0
    )
}

pd.DataFrame([metrics]).to_csv(
    os.path.join(EVAL_DIR, "metrics.csv"),
    index=False
)

# Classification report
report = classification_report(
    all_labels, all_preds,
    target_names=le.classes_,
    digits=4,
    zero_division=0
)

with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d",
    xticklabels=le.classes_,
    yticklabels=le.classes_,
    cmap="Blues"
)
plt.title("Confusion Matrix – Hierarchical BiLSTM + CRF")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()

# Bar plot (F1 per class)
f1s = f1_score(all_labels, all_preds, average=None, zero_division=0)

plt.figure(figsize=(10, 5))
plt.bar(le.classes_, f1s)
plt.xticks(rotation=45)
plt.ylabel("F1-score")
plt.title("Per-class F1 Score")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "f1_bar_plot.png"))
plt.close()

print("Training, checkpointing, and evaluation completed.")
