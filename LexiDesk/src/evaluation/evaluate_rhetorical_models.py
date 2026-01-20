import os
import sys
import json
import joblib
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, accuracy_score

# =====================================================
# PATH FIX
# =====================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, PROJECT_ROOT)

# =====================================================
# PATHS
# =====================================================
DATA_PATH = "data/processed/rhetorical/test_rhetorical.csv"
OUTPUT_DIR = "outputs/rhetorical_evaluation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Baseline paths
BASELINE_MODEL_PATH = "models/rhetorical/baseline/baseline_rhetorical_model.joblib"
TFIDF_PATH = "models/rhetorical/baseline/tfidf_vectorizer.joblib"
BASELINE_LABEL_ENCODER_PATH = "models/rhetorical/baseline/label_encoder.joblib"

# ---- BiLSTM paths
BILSTM_MODEL_PATH = "models/rhetorical/bilstm/bilstm_rhetorical_model.pt"
BILSTM_VOCAB_PATH = "models/rhetorical/bilstm/vocab.joblib"
BILSTM_LABEL_ENCODER_PATH = "models/rhetorical/bilstm/label_encoder.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 100
BATCH_SIZE = 512  # Safe batch size to avoid memory issues

# =====================================================
# LOAD & CLEAN DATA
# =====================================================
df = pd.read_csv(DATA_PATH)
initial_size = len(df)
df = df.dropna(subset=["Label"]).reset_index(drop=True)
removed = initial_size - len(df)

texts = df["Text"].astype(str).tolist()
labels_str = df["Label"].astype(str).tolist()

print(f"Loaded {len(texts)} samples for evaluation")
if removed > 0:
    print(f"Removed {removed} samples with missing labels")

# =====================================================
# HELPER: PER-CLASS BAR PLOT
# =====================================================
def plot_per_class_metrics(report, labels, title, filename):
    precision = [report[l]["precision"] for l in labels]
    recall = [report[l]["recall"] for l in labels]
    f1 = [report[l]["f1-score"] for l in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(14, 6))
    plt.bar(x - width, precision, width, label="Precision")
    plt.bar(x, recall, width, label="Recall")
    plt.bar(x + width, f1, width, label="F1-score")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# =====================================================
# 1. BASELINE EVALUATION
# =====================================================
print("\nEvaluating Baseline Model...")

baseline_model = joblib.load(BASELINE_MODEL_PATH)
vectorizer = joblib.load(TFIDF_PATH)
baseline_le = joblib.load(BASELINE_LABEL_ENCODER_PATH)

baseline_labels = list(baseline_le.classes_)
y_true_baseline = baseline_le.transform(labels_str)

X_tfidf = vectorizer.transform(texts)
y_pred_baseline = baseline_model.predict(X_tfidf)

baseline_report = classification_report(
    y_true_baseline,
    y_pred_baseline,
    target_names=baseline_labels,
    output_dict=True,
    digits=4
)

baseline_acc = accuracy_score(y_true_baseline, y_pred_baseline)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "baseline_report.json"), "w") as f:
    json.dump(baseline_report, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "baseline_summary.txt"), "w") as f:
    f.write(f"Accuracy: {baseline_acc:.4f}\n")

print(f"Baseline Accuracy: {baseline_acc:.4f}")

plot_per_class_metrics(
    baseline_report,
    baseline_labels,
    "Baseline Rhetorical Classifier – Per Class Metrics",
    "baseline_per_class_metrics.png"
)

# =====================================================
# 2. BiLSTM MODEL DEFINITION
# =====================================================
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            emb_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = (weights * lstm_out).sum(dim=1)
        return self.fc(context)

# =====================================================
# 3. BiLSTM EVALUATION (BATCHED)
# =====================================================
print("\nEvaluating BiLSTM Model...")

vocab = joblib.load(BILSTM_VOCAB_PATH)
bilstm_le = joblib.load(BILSTM_LABEL_ENCODER_PATH)

PAD_ID = vocab["<PAD>"]
UNK_ID = vocab["<UNK>"]

def encode_text(text):
    ids = [vocab.get(w, UNK_ID) for w in text.split()]
    ids = ids[:MAX_LEN] + [PAD_ID] * max(0, MAX_LEN - len(ids))
    return ids

# Convert all texts to integer sequences
X_seq_all = torch.tensor([encode_text(t) for t in texts], dtype=torch.long)
y_true_bilstm = bilstm_le.transform(labels_str)

# Create DataLoader for batching
dataset = TensorDataset(X_seq_all, torch.tensor(y_true_bilstm, dtype=torch.long))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

# Load trained BiLSTM model
model = BiLSTMAttention(
    vocab_size=len(vocab),
    emb_dim=128,
    hidden_dim=128,
    num_classes=len(bilstm_le.classes_)
).to(DEVICE)
model.load_state_dict(torch.load(BILSTM_MODEL_PATH, map_location=DEVICE))
model.eval()

# Evaluate in batches to avoid memory issues
all_preds = []

with torch.no_grad():
    for batch_x, _ in dataloader:
        batch_x = batch_x.to(DEVICE)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)

y_pred_bilstm = np.array(all_preds)
bilstm_labels = list(bilstm_le.classes_)

bilstm_report = classification_report(
    y_true_bilstm,
    y_pred_bilstm,
    target_names=bilstm_labels,
    output_dict=True,
    digits=4
)

bilstm_acc = accuracy_score(y_true_bilstm, y_pred_bilstm)

# Save metrics
with open(os.path.join(OUTPUT_DIR, "bilstm_report.json"), "w") as f:
    json.dump(bilstm_report, f, indent=4)

with open(os.path.join(OUTPUT_DIR, "bilstm_summary.txt"), "w") as f:
    f.write(f"Accuracy: {bilstm_acc:.4f}\n")

print(f"BiLSTM Accuracy: {bilstm_acc:.4f}")

plot_per_class_metrics(
    bilstm_report,
    bilstm_labels,
    "BiLSTM Rhetorical Classifier – Per Class Metrics",
    "bilstm_per_class_metrics.png"
)

# =====================================================
# 4. COMBINED F1-SCORE COMPARISON PLOT
# =====================================================
def plot_combined_f1(baseline_report, bilstm_report, labels, filename):
    baseline_f1 = [baseline_report[l]["f1-score"] for l in labels]
    bilstm_f1 = [bilstm_report[l]["f1-score"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, baseline_f1, width, label="Baseline F1")
    plt.bar(x + width/2, bilstm_f1, width, label="BiLSTM F1")

    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("F1-score")
    plt.title("Baseline vs BiLSTM – Per Class F1-score Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_combined_f1(
    baseline_report,
    bilstm_report,
    baseline_labels,  # assumes both models have same class labels
    "baseline_vs_bilstm_f1_comparison.png"
)

# =====================================================
print("\nUnified evaluation complete.")
print("Outputs saved to:", OUTPUT_DIR)
