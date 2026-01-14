import os

#PROJECT_ROOT = "LexiDesk"

DIRS = [
    "data/raw/judgments",
    "data/raw/metadata",
    "data/processed/sbd",
    "data/processed/rhetorical_roles",
    "data/processed/extractive",
    "data/processed/abstractive",

    "models/sbd",
    "models/rhetorical",
    "models/extractive",
    "models/abstractive",

    "src/sbd",
    "src/rhetorical",
    "src/extractive",
    "src/abstractive",
    "src/preprocessing",
    "src/evaluation",

    "scripts/scrape",
    "scripts/preprocess",
    "scripts/train",
    "scripts/evaluate",

    "configs"
]

FILES = [
    "README.md",
    "requirements.txt"
]

def main():
    os.makedirs(PROJECT_ROOT, exist_ok=True)

    for d in DIRS:
        os.makedirs(os.path.join(PROJECT_ROOT, d), exist_ok=True)

    for f in FILES:
        path = os.path.join(PROJECT_ROOT, f)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()

    print("LexiDesk project structure created successfully.")

if __name__ == "__main__":
    main()
