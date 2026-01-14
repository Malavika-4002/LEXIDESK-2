# src/rhetorical/build_rhetorical_dataset.py

import os
import json
import joblib
import torch
import pandas as pd
from tqdm import tqdm

from src.sbd.cnn_model import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sbd")

TRAIN_JSON = os.path.join(RAW_DIR, "role_train1.json")
DEV_JSON = os.path.join(RAW_DIR, "role-dev1.json")

CNN_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
CRF_PATH = os.path.join(MODEL_DIR, "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_vocab.json")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD SBD MODELS
# --------------------------------------------------
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

# --------------------------------------------------
# SBD FUNCTIONS
# --------------------------------------------------
def cnn_boundary_probability(text, idx):
    if text[idx] not in DELIMITERS:
        return 0.0

    left = max(0, idx - CONTEXT_WINDOW)
    window = text[left:idx] + text[idx] + text[idx + 1:idx + 1 + CONTEXT_WINDOW]

    pad_idx = char_to_idx["<PAD>"]
    encoded = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in window]
    encoded = encoded[: (2 * CONTEXT_WINDOW + 1)]
    encoded += [pad_idx] * ((2 * CONTEXT_WINDOW + 1) - len(encoded))

    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        return cnn_model(tensor).item()


def segment_sentences(text):
    tokens = list(text)
    features = []

    for i, token in enumerate(tokens):
        feat = token_to_features(token, text, i, i + 1)
        feat["cnn_prob"] = cnn_boundary_probability(text, i)
        features.append(feat)

    features = add_neighboring_token_features(features)
    labels = crf_model.predict([features])[0]

    sentences = []
    start = 0

    for i, lab in enumerate(labels):
        if lab == "B":
            sentences.append((start, i + 1, text[start:i + 1].strip()))
            start = i + 1

    if start < len(text):
        sentences.append((start, len(text), text[start:].strip()))

    return sentences

# --------------------------------------------------
# ANNOTATION ALIGNMENT
# --------------------------------------------------
def get_label_for_sentence(sent_start, sent_end, spans):
    max_overlap = 0
    chosen_label = "NONE"

    for span in spans:
        s_start = span["value"]["start"]
        s_end = span["value"]["end"]
        label = span["value"]["labels"][0]

        overlap = max(0, min(sent_end, s_end) - max(sent_start, s_start))
        if overlap > max_overlap:
            max_overlap = overlap
            chosen_label = label

    return chosen_label

# --------------------------------------------------
# DATASET BUILDER
# --------------------------------------------------
def process_file(json_path, output_csv):
    rows = []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        doc_id = item["id"]
        text = item["data"]["text"]

        spans = []
        for ann in item["annotations"]:
            for res in ann["result"]:
                spans.append(res)

        sentences = segment_sentences(text)

        for s_start, s_end, sentence in sentences:
            if not sentence:
                continue

            label = get_label_for_sentence(s_start, s_end, spans)
            rows.append({
                "doc_id": doc_id,
                "sentence": sentence,
                "label": label
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    process_file(TRAIN_JSON, os.path.join(OUT_DIR, "rhetorical_train.csv"))
    process_file(DEV_JSON, os.path.join(OUT_DIR, "rhetorical_dev.csv"))

    print("\nRhetorical role dataset preparation complete.")
