import os
import json
import joblib
import torch
import pandas as pd
import re
from tqdm import tqdm

# Import your existing SBD components
from src.sbd.cnn_model import LegalSBD_CNN
from src.sbd.feature_extractor import token_to_features, add_neighboring_token_features

# --------------------------------------------------
# 1. PATH CONFIGURATION
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
MODEL_DIR = os.path.join(BASE_DIR, "models", "sbd")

# Ensure your raw files are named exactly like this in data/raw/
TRAIN_JSON = os.path.join(RAW_DIR, "role_train1.json")
DEV_JSON = os.path.join(RAW_DIR, "role-dev1.json")

CNN_PATH = os.path.join(MODEL_DIR, "cnn_model.pth")
CRF_PATH = os.path.join(MODEL_DIR, "crf_hybrid_model.joblib")
VOCAB_PATH = os.path.join(MODEL_DIR, "char_vocab.json")

os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD SBD MODELS
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SBD Models...")
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
# 3. SEGMENTATION LOGIC (MATCHES SBD PIPELINE)
# --------------------------------------------------
def cnn_boundary_probability(text, idx):
    """Calculates boundary probability from local character context."""
    if text[idx] not in DELIMITERS:
        return 0.0

    left = max(0, idx - CONTEXT_WINDOW)
    # Context window of characters
    window = text[left:idx] + text[idx] + text[idx + 1:idx + 1 + CONTEXT_WINDOW]

    pad_idx = char_to_idx["<PAD>"]
    encoded = [char_to_idx.get(c, char_to_idx["<UNK>"]) for c in window]
    
    max_len = (2 * CONTEXT_WINDOW + 1)
    encoded = encoded[:max_len]
    encoded += [pad_idx] * (max_len - len(encoded))

    tensor = torch.tensor([encoded], dtype=torch.long).to(device)

    with torch.no_grad():
        return cnn_model(tensor).item()


def segment_sentences(text):
    """
    Sentence segmentation using Regex Tokenization + CNN-CRF.
    """
    # Professional regex for word/citation-level tokenization
    tokens_with_spans = [
        (m.group(0), m.start(), m.end())
        for m in re.finditer(r"[\w'-]+|[.,!?;:()]|\S+", text)
    ]
    
    if not tokens_with_spans:
        return []

    sentence_features = []
    for token, start, end in tokens_with_spans:
        # Extract features for the WORD/TOKEN, not character
        feat = token_to_features(token, text, start, end)
        
        if token in DELIMITERS:
            feat["cnn_prob"] = round(cnn_boundary_probability(text, start), 4)
        else:
            feat["cnn_prob"] = 0.0
            
        sentence_features.append(feat)

    # Contextualize features for CRF
    sentence_features = add_neighboring_token_features(sentence_features)
    labels = crf_model.predict([sentence_features])[0]

    sentences = []
    current_sent_start_idx = 0

    for i, lab in enumerate(labels):
        if lab == "B":
            # Safety gate: Don't split if next token starts with lowercase
            if i + 1 < len(tokens_with_spans):
                next_token_text = tokens_with_spans[i+1][0]
                if next_token_text and next_token_text[0].islower():
                    continue

            end_char_pos = tokens_with_spans[i][2]
            start_char_pos = tokens_with_spans[current_sent_start_idx][1]
            
            sent_str = text[start_char_pos:end_char_pos].strip()
            sentences.append((start_char_pos, end_char_pos, sent_str))
            current_sent_start_idx = i + 1

    # Add the last remaining segment
    if current_sent_start_idx < len(tokens_with_spans):
        start_char_pos = tokens_with_spans[current_sent_start_idx][1]
        sentences.append((start_char_pos, len(text), text[start_char_pos:].strip()))

    return sentences

# --------------------------------------------------
# 4. ANNOTATION ALIGNMENT (OVERLAP STRATEGY)
# --------------------------------------------------
def get_label_for_sentence(sent_start, sent_end, spans):
    """
    Finds which annotated label has the highest character overlap 
    with our SBD-generated sentence.
    """
    max_overlap = 0
    chosen_label = "NONE"

    for span in spans:
        # Note: adjust these keys based on your specific JSON structure
        try:
            s_start = span["value"]["start"]
            s_end = span["value"]["end"]
            label = span["value"]["labels"][0]
        except (KeyError, IndexError):
            continue

        # Calculate intersection
        overlap = max(0, min(sent_end, s_end) - max(sent_start, s_start))
        
        if overlap > max_overlap:
            max_overlap = overlap
            chosen_label = label

    # If overlap is very small (e.g. less than 10 chars), default to NONE
    if max_overlap < 5:
        return "NONE"
        
    return chosen_label

# --------------------------------------------------
# 5. DATASET BUILDER
# --------------------------------------------------
def process_file(json_path, output_csv):
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    rows = []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in tqdm(data, desc=f"Processing {os.path.basename(json_path)}"):
        doc_id = item["id"]
        text = item["data"]["text"]

        # Page number cleaning (crucial for legal docs)
        text = re.sub(r'\n\d+\s?\n', ' ', text)

        # Gather all human-labeled spans
        all_spans = []
        for ann in item["annotations"]:
            if "result" in ann:
                all_spans.extend(ann["result"])

        # Step 1: Use your CNN-CRF to split the text
        sentences = segment_sentences(text)

        # Step 2: Map each sentence to a label
        for s_start, s_end, sentence in sentences:
            if len(sentence) < 5: # Ignore very short fragments
                continue

            label = get_label_for_sentence(s_start, s_end, all_spans)
            
            # Clean sentence for CSV
            clean_sent = " ".join(sentence.split())
            
            rows.append({
                "doc_id": doc_id,
                "sentence": clean_sent,
                "label": label
            })

    df = pd.DataFrame(rows)
    
    # Logic to balance the dataset: If there's too much "NONE", we cap it
    # to avoid biasing the RRC classifier.
    none_count = len(df[df['label'] == 'NONE'])
    if none_count > (len(df) * 0.5):
        print(f"Warning: {none_count} 'NONE' labels found. Consider filtering.")

    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} rows to {output_csv}")
    print("Class Distribution:\n", df['label'].value_counts())

# --------------------------------------------------
# 6. EXECUTION
# --------------------------------------------------
if __name__ == "__main__":
    print("Starting Rhetorical Role Dataset Preparation...")
    process_file(TRAIN_JSON, os.path.join(OUT_DIR, "v2_rhetorical_train.csv"))
    process_file(DEV_JSON, os.path.join(OUT_DIR, "v2_rhetorical_dev.csv"))
    print("\nProcessing Complete.")