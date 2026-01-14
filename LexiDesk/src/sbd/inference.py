# src/sbd/inference.py

import os
import json
import torch
import joblib
import re
from typing import List

from src.sbd.cnn_model import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features

# ----------------------------
# FIXED PATHS (EDIT IF NEEDED)
# ----------------------------
MODEL_DIR = r"C:\Users\acer\Downloads\LexiDesk-2\LexiDesk\models\sbd"

CNN_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
CRF_PATH = os.path.join(MODEL_DIR, "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_vocab.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# LOAD MODELS (FAIL LOUDLY)
# ----------------------------
if not (os.path.exists(CNN_PATH) and os.path.exists(CRF_PATH) and os.path.exists(VOCAB_PATH)):
    raise RuntimeError("SBD model files missing. Phase 2 cannot proceed.")

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    CHAR_VOCAB = json.load(f)

VOCAB_SIZE = len(CHAR_VOCAB) + 1

cnn_model = LegalSBD_CNN(VOCAB_SIZE)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn_model.to(DEVICE)
cnn_model.eval()

crf_model = joblib.load(CRF_PATH)

# ----------------------------
# UTILITIES
# ----------------------------
def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text, re.UNICODE)

def _encode(tokens):
    return torch.tensor([
        CHAR_VOCAB.get(tok.lower(), 0) for tok in tokens
    ], dtype=torch.long).unsqueeze(0)

# ----------------------------
# PUBLIC API
# ----------------------------
def get_sentences(text: str) -> List[str]:
    """
    Input: raw legal text
    Output: list of sentence strings
    Uses Hybrid CNN + CRF
    """

    tokens = _tokenize(text)
    if len(tokens) < 5:
        return [text.strip()]

    # CNN probabilities
    with torch.no_grad():
        x = _encode(tokens).to(DEVICE)
        cnn_probs = cnn_model(x).squeeze(0).cpu().numpy()

    # CRF features
    features = []
    for i in range(len(tokens)):
        f = token_to_features(tokens, i)
        f["cnn_prob"] = float(cnn_probs[i])
        features.append(f)

    # CRF decode
    labels = crf_model.predict([features])[0]

    # Sentence reconstruction
    sentences = []
    current = []

    for tok, lab in zip(tokens, labels):
        current.append(tok)
        if lab == "B":
            sentence = " ".join(current).strip()
            if len(sentence) > 3:
                sentences.append(sentence)
            current = []

    if current:
        sentences.append(" ".join(current).strip())

    return sentences
