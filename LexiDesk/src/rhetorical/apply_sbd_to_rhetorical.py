# src/sbd/apply_sbd_to_rhetorical.py

import os
import json
import joblib
import torch
import pandas as pd
from tqdm import tqdm

from src.sbd.cnn_model import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# -----------------------------
# PATH CONFIGURATION
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = os.path.join(BASE_DIR, "models", "sbd")
INPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")

TRAIN_IN = os.path.join(INPUT_DIR, "rhetorical_train.csv")
DEV_IN   = os.path.join(INPUT_DIR, "rhetorical_dev.csv")

TRAIN_OUT = os.path.join(OUTPUT_DIR, "rhetorical_train_sbd.csv")
DEV_OUT   = os.path.join(OUTPUT_DIR, "rhetorical_dev_sbd.csv")

CNN_PATH   = os.path.join(MODEL_DIR, "cnn_model.pth")
CRF_PATH   = os.path.join(MODEL_DIR, "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_vocab.json")

DELIMITERS = {".", "?", "!", ";", ":"}
CONTEXT_WINDOW = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODELS
# -----------------------------
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

# -----------------------------
# CNN PROBABILITY HELPER
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
# SENTENCE SPLITTING FUNCTION
# -----------------------------
def split_sentences(text):
    tokens = list(text)
    features = []

    for idx, token in enumerate(tokens):
        feat = token_to_features(token, text, idx, idx + 1)
        feat["cnn_prob"] = round(cnn_boundary_probability(text, idx), 4)
        features.append(feat)

    features = add_neighboring_token_features(features)
    predictions = crf_model.predict([features])[0]

    sentences = []
    start = 0
    for i, label in enumerate(predictions):
        if label == "B" and i != 0:
            sent = "".join(tokens[start:i]).strip()
            if sent:
                sentences.append(sent)
            start = i

    final = "".join(tokens[start:]).strip()
    if final:
        sentences.append(final)

    return sentences

# -----------------------------
# PROCESS FILE
# -----------------------------
def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    rows = []

    print(f"Processing: {os.path.basename(input_csv)}")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        doc_id = row["doc_id"]
        text = str(row["sentence"])
        label = row["label"]

        sents = split_sentences(text)

        for s in sents:
            rows.append({
                "doc_id": doc_id,
                "sentence": s,
                "label": label
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_csv, index=False)

    print(f"Saved → {output_csv}")
    print(f"Total sentences: {len(out_df)}\n")

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    process_file(TRAIN_IN, TRAIN_OUT)
    process_file(DEV_IN, DEV_OUT)

    print("Rhetorical datasets successfully normalized using Hybrid SBD.")
