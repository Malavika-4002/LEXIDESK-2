# src/sbd/evaluate_sbd.py

import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

from src.sbd.cnn_model import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(BASE_DIR, "models", "sbd")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "sbd")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
PLOTS_DIR = os.path.join(EVAL_DIR, "plots")
SEGMENTED_DIR = os.path.join(EVAL_DIR, "segmented_sentences")

TEST_CSV = os.path.join(DATA_DIR, "test_data.csv")

CNN_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
CRF_PATH = os.path.join(MODEL_DIR, "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_vocab.json")

# Auto-create folders
for d in [EVAL_DIR, RESULTS_DIR, PLOTS_DIR, SEGMENTED_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# MODEL LOADING
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(VOCAB_PATH, "r") as f:
    char_to_idx = json.load(f)

cnn_model = LegalSBD_CNN(
    vocab_size=len(char_to_idx),
    embedding_dim=128,
    num_filters=6,
    kernel_size=5,
    hidden_dim=250,
    dropout_prob=0.2
).to(device)

cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=device))
cnn_model.eval()

crf_model = joblib.load(CRF_PATH)

DELIMITERS = {".", "?", "!", ";", ":"}
CONTEXT_WINDOW = 6

# -----------------------------
# CNN HELPER
# -----------------------------
def cnn_boundary_probability(text, idx):
    if text[idx] not in DELIMITERS:
        return 0.0

    left = max(0, idx - CONTEXT_WINDOW)
    window = text[left:idx] + text[idx] + text[idx+1:idx+1+CONTEXT_WINDOW]

    pad_idx = char_to_idx["<PAD>"]
    encoded = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in window]
    encoded = encoded[: (2 * CONTEXT_WINDOW + 1)]
    encoded += [pad_idx] * ((2 * CONTEXT_WINDOW + 1) - len(encoded))

    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        return cnn_model(tensor).item()

# -----------------------------
# LOAD TEST DATA
# -----------------------------
df = pd.read_csv(TEST_CSV)

y_true = []
y_pred = []
cnn_scores = []

print("Evaluating Hybrid CNN–CRF Sentence Boundary Detector...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    left = str(row["left_context"])
    delim = str(row["delimiter"])
    right = str(row["right_context"])
    label = int(row["label"])

    text = left + delim + right
    delim_index = len(left)

    # --- Generate features for all tokens ---
    tokens = list(text)
    all_features = []
    for idx, token in enumerate(tokens):
        feat = token_to_features(token, text, idx, idx + 1)
        feat["cnn_prob"] = round(cnn_boundary_probability(text, idx), 4)
        all_features.append(feat)

    all_features = add_neighboring_token_features(all_features)

    # CRF expects a list of sequences
    prediction_seq = crf_model.predict([all_features])[0]

    # Pick the label at delimiter index
    pred_label = prediction_seq[delim_index]

    y_true.append("B" if label == 1 else "O")
    y_pred.append(pred_label)
    cnn_scores.append(all_features[delim_index].get("cnn_prob", 0.0))

# -----------------------------
# METRICS
# -----------------------------
precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", pos_label="B", zero_division=0
)

accuracy = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred, labels=["B", "O"])

metrics = {
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "accuracy": accuracy
}

# Save metrics
pd.DataFrame([metrics]).to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# -----------------------------
# CONFUSION MATRIX PLOT
# -----------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Boundary", "Non-Boundary"],
    yticklabels=["Boundary", "Non-Boundary"]
)
plt.title("Hybrid CNN–CRF SBD Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# PRECISION–RECALL CURVE
# -----------------------------
binary_true = [1 if y == "B" else 0 for y in y_true]
precision_curve, recall_curve, _ = precision_recall_curve(binary_true, cnn_scores)

plt.figure(figsize=(7, 5))
plt.plot(recall_curve, precision_curve, color="darkorange", lw=2, label="Precision-Recall")
plt.fill_between(recall_curve, precision_curve, alpha=0.3, color="darkorange")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("CNN Probability Precision–Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "precision_recall_curve.png"))
plt.close()

# -----------------------------
# BAR GRAPH OF METRICS
# -----------------------------
plt.figure(figsize=(7, 5))
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
sns.barplot(x=metric_names, y=metric_values, palette="viridis")
plt.title("SBD Evaluation Metrics")
plt.ylim(0, 1)
for i, v in enumerate(metric_values):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "metrics_bargraph.png"))
plt.close()

# -----------------------------
# MULTIPLE LEGAL SENTENCES SEGMENTATION
# -----------------------------
legal_texts = [
    "Whereas the Plaintiff claims damages for breach of contract, the Defendant argues that the contract was void ab initio due to lack of consideration; however, the Court observes that performance was partially executed and therefore some remedies may still apply.",
    "The lease agreement between the parties, executed on January 1st, 2020, explicitly states the tenant shall maintain insurance coverage; failure to comply constitutes a material breach of the contract.",
    "If the defendant fails to appear before the court on the scheduled hearing date, a default judgment may be entered against him in accordance with the civil procedure code.",
]

segmented_sentences_dict = {}

for i, text in enumerate(legal_texts):
    tokens = list(text)
    all_features = []
    for idx, token in enumerate(tokens):
        feat = token_to_features(token, text, idx, idx + 1)
        feat["cnn_prob"] = round(cnn_boundary_probability(text, idx), 4)
        all_features.append(feat)
    all_features = add_neighboring_token_features(all_features)
    pred_sequence = crf_model.predict([all_features])[0]

    segmented_sentences = []
    current_sentence = ""
    for token, label in zip(tokens, pred_sequence):
        current_sentence += token
        if label == "B":
            segmented_sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence:
        segmented_sentences.append(current_sentence.strip())

    segmented_sentences_dict[f"legal_text_{i+1}"] = segmented_sentences

# Save segmented sentences
for key, sentences in segmented_sentences_dict.items():
    with open(os.path.join(SEGMENTED_DIR, f"{key}_segmented.txt"), "w", encoding="utf-8") as f:
        for s in sentences:
            f.write(s + "\n")

# -----------------------------
# FINAL REPORT
# -----------------------------
print("\n--- SBD Evaluation Complete ---")
for k, v in metrics.items():
    print(f"{k.upper():12s}: {v:.4f}")

print(f"\nResults saved in: {EVAL_DIR}")
print(f"Segmented multiple legal sentences saved in: {SEGMENTED_DIR}")
