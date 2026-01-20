import json
import pandas as pd

# Paths to your JSON files
json_files = [
    r"C:\Users\acer\Downloads\LexiDesk-2\LexiDesk\data\raw\role_train1.json",
    r"C:\Users\acer\Downloads\LexiDesk-2\LexiDesk\data\raw\role-dev1.json"
]

for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for doc in data:
        doc_id = doc.get("id")
        annotations = doc.get("annotations", [])
        
        for ann in annotations:
            # "result" is a list of segment annotations
            for result in ann.get("result", []):
                value = result.get("value", {})
                text = value.get("text", "").strip()
                labels = value.get("labels", [])
                
                # Decide how you want to handle labels:
                # here we take the first label if there are multiple
                label = labels[0] if labels else None
                
                rows.append({
                    "doc_id": doc_id,
                    "text": text,
                    "label": label
                })

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Choose output CSV name
    out_csv_path = json_path.replace(".json", "_converted.csv")
    df.to_csv(out_csv_path, index=False, encoding="utf-8")

    print(f"Converted {json_path} → {out_csv_path}")
