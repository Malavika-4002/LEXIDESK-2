# role_aware_tfidf_mmr.py
# ============================================================
# Role-Aware Extractive Summarization (TF-IDF + MMR)
# Simple path-based version (NO split logic, NO argparse)
# ============================================================

import os
import json
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# FILE PATHS (EDIT ONLY THESE IF NEEDED)
# ============================================================

INPUT_CSV = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk/outputs/rhetorical/HierarchicalBilstmCRF/train_rhetorical_predictions.csv"
OUTPUT_JSONL = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk/data/processed/extractive/train_extractive.jsonl"

# ============================================================
# CONFIG
# ============================================================

MAX_SENTENCES = 12
MMR_LAMBDA = 0.7

ROLE_WEIGHTS = {
    "Issue": 1.3,
    "Facts": 1.2,
    "Arguments": 1.1,
    "Reasoning": 1.3,
    "Decision": 1.4,
    "Conclusion": 1.4
}

DEFAULT_ROLE_WEIGHT = 1.0

# ============================================================
# MMR FUNCTION
# ============================================================

def mmr(sent_embeddings, doc_embedding, lambda_param, top_k):
    selected = []
    candidates = list(range(len(sent_embeddings)))

    sim_to_doc = cosine_similarity(sent_embeddings, doc_embedding)

    while len(selected) < top_k and candidates:
        scores = []

        for idx in candidates:
            relevance = sim_to_doc[idx][0]

            redundancy = 0.0
            if selected:
                redundancy = max(
                    cosine_similarity(
                        sent_embeddings[idx].reshape(1, -1),
                        sent_embeddings[selected]
                    )[0]
                )

            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            scores.append((idx, score))

        best_idx, _ = max(scores, key=lambda x: x[1])
        selected.append(best_idx)
        candidates.remove(best_idx)

    return sorted(selected)

# ============================================================
# EXTRACTIVE SUMMARIZER
# ============================================================

def role_aware_extractive(df_doc):
    sentences = df_doc["Text"].astype(str).tolist()
    roles = df_doc["Predicted_Role"].fillna("Other").tolist()

    if len(sentences) <= 2:
        return sentences, roles

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(sentences)

    doc_embedding = np.asarray(tfidf.mean(axis=0))

    weighted_embeddings = []
    for i, role in enumerate(roles):
        weight = ROLE_WEIGHTS.get(role, DEFAULT_ROLE_WEIGHT)
        weighted_embeddings.append(tfidf[i].toarray()[0] * weight)

    weighted_embeddings = np.vstack(weighted_embeddings)

    selected_indices = mmr(
        weighted_embeddings,
        doc_embedding,
        MMR_LAMBDA,
        min(MAX_SENTENCES, len(sentences))
    )

    return (
        [sentences[i] for i in selected_indices],
        [roles[i] for i in selected_indices]
    )

# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading CSV...")
    df = pd.read_csv(INPUT_CSV)
    print(f"Rows loaded: {len(df)}")

    # ---- REQUIRED COLUMN CHECK ----
    required_cols = {"Text", "Predicted_Role"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # ---- DOCUMENT ID DETECTION ----
    if "Index" in df.columns:
        doc_col = "Index"
    elif "doc_id" in df.columns:
        doc_col = "doc_id"
    else:
        raise ValueError("CSV must have 'Index' or 'doc_id' column")

    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)

    written = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
        for doc_id, group in df.groupby(doc_col):
            ext_sents, ext_roles = role_aware_extractive(group)

            if not ext_sents:
                continue

            record = {
                "doc_id": int(doc_id),
                "extractive_input": " ".join(
                    f"<s> {s.strip()} </s>" for s in ext_sents
                ),
                "sentences": ext_sents,
                "roles": ext_roles
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 50 == 0:
                print(f"Processed {written} documents...")

    print(f"✅ Done. Documents written: {written}")
    print(f"📄 Output file: {OUTPUT_JSONL}")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
