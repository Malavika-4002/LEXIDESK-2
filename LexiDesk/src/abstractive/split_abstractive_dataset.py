# ============================================================
# Split Abstractive Dataset into Train / Val / Test
# ============================================================

import json
import random
import os

INPUT_JSONL = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk/data/raw/abstractive/abstractive_dataset.jsonl"

OUT_DIR = "/home/csnn02/MALAVIKA_PROJECT/LEXIDESK-2/LexiDesk/data/raw/abstractive"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)

    n = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_data = data[:n_train]
    val_data = data[n_train:n_train + n_val]
    test_data = data[n_train + n_val:]

    splits = {
        "train.jsonl": train_data,
        "val.jsonl": val_data,
        "test.jsonl": test_data
    }

    for fname, split_data in splits.items():
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            for item in split_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"✅ Written {len(split_data)} → {path}")

    print("\n🎉 Abstractive dataset successfully split.")

if __name__ == "__main__":
    main()
