# ============================================================
# build_abstractive_datasets.py
# ============================================================
# Single-run script to prepare abstractive datasets using:
#   - Role-aware extractive summaries
#   - Gold abstractive summaries
# For train / val / test
# ============================================================

import os
import json
from tqdm import tqdm

# ============================================================
# PATH CONFIG
# ============================================================

BASE_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk"

EXTRACTIVE_DIR = os.path.join(BASE_DIR, "data/processed/extractive")
ABSTRACTIVE_RAW_DIR = os.path.join(BASE_DIR, "data/raw/abstractive")
ABSTRACTIVE_OUT_DIR = os.path.join(BASE_DIR, "data/processed/abstractive")

# ---- Extractive inputs ----
TRAIN_EXTRACTIVE = os.path.join(EXTRACTIVE_DIR, "train_extractive.jsonl")
VAL_EXTRACTIVE   = os.path.join(EXTRACTIVE_DIR, "val_extractive.jsonl")
TEST_EXTRACTIVE  = os.path.join(EXTRACTIVE_DIR, "test_extractive.jsonl")

# ---- Gold abstractive summaries ----
TRAIN_GOLD = os.path.join(ABSTRACTIVE_RAW_DIR, "train.jsonl")
VAL_GOLD   = os.path.join(ABSTRACTIVE_RAW_DIR, "val.jsonl")
TEST_GOLD  = os.path.join(ABSTRACTIVE_RAW_DIR, "test.jsonl")

# ---- Outputs ----
TRAIN_OUT = os.path.join(ABSTRACTIVE_OUT_DIR, "train_abstractive.jsonl")
VAL_OUT   = os.path.join(ABSTRACTIVE_OUT_DIR, "val_abstractive.jsonl")
TEST_OUT  = os.path.join(ABSTRACTIVE_OUT_DIR, "test_abstractive.jsonl")

# ============================================================
# UTILITIES
# ============================================================

def load_jsonl(path, key_field=None):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if key_field:
                data[obj[key_field]] = obj
            else:
                data[len(data)] = obj
    return data


def build_split(extractive_path, gold_path, output_path):
    print(f"\n🔹 Processing split:")
    print(f"   Extractive: {extractive_path}")
    print(f"   Gold:       {gold_path}")

    extractive = load_jsonl(extractive_path, key_field="doc_id")
    gold = load_jsonl(gold_path)

    aligned = []
    missing = 0

    for doc_id, ext in tqdm(extractive.items(), desc="Aligning"):
        if doc_id not in gold:
            missing += 1
            continue

        aligned.append({
            "doc_id": doc_id,
            "input": ext["extractive_input"],
            "target": gold[doc_id]["target"]
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in aligned:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Written: {len(aligned)}")
    print(f"⚠️  Missing gold summaries: {missing}")
    print(f"📄 Output: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n==============================")
    print(" Building Abstractive Datasets")
    print("==============================")

    build_split(TRAIN_EXTRACTIVE, TRAIN_GOLD, TRAIN_OUT)
    build_split(VAL_EXTRACTIVE, VAL_GOLD, VAL_OUT)
    build_split(TEST_EXTRACTIVE, TEST_GOLD, TEST_OUT)

    print("\n🎯 DONE. Train / Val / Test abstractive datasets are ready.")


if __name__ == "__main__":
    main()
