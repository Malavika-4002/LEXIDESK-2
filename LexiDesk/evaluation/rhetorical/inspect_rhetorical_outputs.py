# evaluation/rhetorical/inspect_rhetorical_outputs.py

import os
import sys
import torch
import joblib
import pandas as pd

# -------------------------------------------------
# FIX PYTHON PATH (THIS SOLVES THE src ERROR)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(PROJECT_ROOT)

from src.rhetorical.train_hierarchical_bilstm_crf_correct import (
    HierarchicalBiLSTMCRF,
    RhetoricalDocDataset,
    collate_fn,
    MAX_TOKENS
)

from torch.utils.data import DataLoader

# =========================
# PATHS
# =========================
BASE_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk"

DATA_PATH = os.path.join(BASE_DIR, "data/processed/rhetorical/train_rhetorical.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models/rhetorical/hierarchical_bilstm_crf/best_model.pt")
VOCAB_PATH = os.path.join(BASE_DIR, "models/rhetorical/hierarchical_bilstm_crf/vocab.joblib")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models/rhetorical/hierarchical_bilstm_crf/label_encoder.joblib")

OUTPUT_CSV = os.path.join(BASE_DIR, "outputs/rhetorical/HierarchicalBilstmCRF/train_rhetorical_predictions.csv")

DEVICE = torch.device("cpu")


# =========================
# LOAD RESOURCES
# =========================
vocab = joblib.load(VOCAB_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# =========================
# DATASET (DOCUMENT LEVEL)
# =========================
dataset = RhetoricalDocDataset(DATA_PATH, vocab, label_encoder)
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# =========================
# LOAD MODEL
# =========================
model = HierarchicalBiLSTMCRF(
    vocab_size=len(vocab),
    num_labels=len(label_encoder.classes_)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# =========================
# ORIGINAL DATAFRAME
# =========================
df = pd.read_csv(DATA_PATH)

rows = []
cursor = 0  # tracks sentence rows in CSV

# =========================
# INFERENCE
# =========================
with torch.no_grad():
    for X, y, mask in loader:
        X, y, mask = X.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)

        preds = model.decode(X, mask)[0]
        golds = y[0][:len(preds)].cpu().tolist()

        for gold_id, pred_id in zip(golds, preds):
            rows.append({
                "Index": df.iloc[cursor]["Index"],
                "Text": df.iloc[cursor]["Text"],
                "Gold_Role": label_encoder.inverse_transform([gold_id])[0],
                "Predicted_Role": label_encoder.inverse_transform([pred_id])[0]
            })
            cursor += 1

# =========================
# SAVE OUTPUT
# =========================
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

out_df = pd.DataFrame(rows)
out_df.to_csv(OUTPUT_CSV, index=False)

print(out_df.head(20))
print(f"\nSaved predictions to {OUTPUT_CSV}")
