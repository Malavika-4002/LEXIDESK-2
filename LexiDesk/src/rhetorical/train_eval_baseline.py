import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # for saving model, vectorizer, encoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "rhetorical")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation", "rhetorical", "baseline")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "rhetorical", "baseline")
MODEL_DIR = os.path.join(BASE_DIR, "models", "rhetorical", "baseline")  # your requested path

for d in [EVAL_DIR, OUT_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

TRAIN_CSV = os.path.join(DATA_DIR, "train_rhetorical.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_rhetorical.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_rhetorical.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
print("Loading data...")
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

X_train = train_df["Text"].astype(str)
y_train = train_df["Label"]

X_test = test_df["Text"].astype(str)
y_test = test_df["Label"]

# -----------------------------
# LABEL ENCODING
# -----------------------------
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
print("LabelEncoder mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# -----------------------------
# MODEL
# -----------------------------
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=40000,
    min_df=2
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print("Training Logistic Regression model...")
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train_vec, y_train_enc)

# -----------------------------
# SAVE MODEL, VECTORIZER & LABEL ENCODER
# -----------------------------
print("Saving model, vectorizer, and LabelEncoder...")
joblib.dump(model, os.path.join(MODEL_DIR, "baseline_rhetorical_model.joblib"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print(f"All saved in: {MODEL_DIR}")

# -----------------------------
# METRICS
# -----------------------------
print("Evaluating model...")
preds = model.predict(X_test_vec)
metrics = {
    "macro_precision": precision_score(y_test_enc, preds, average="macro"),
    "macro_recall": recall_score(y_test_enc, preds, average="macro"),
    "macro_f1": f1_score(y_test_enc, preds, average="macro"),
    "accuracy": accuracy_score(y_test_enc, preds)
}

# Save metrics
pd.DataFrame([metrics]).to_csv(os.path.join(EVAL_DIR, "metrics.csv"), index=False)
with open(os.path.join(EVAL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=4)

# Classification report
report = classification_report(y_test_enc, preds, target_names=[str(c) for c in le.classes_], digits=4)
with open(os.path.join(EVAL_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_test_enc, preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap="Blues")
plt.title("Baseline Rhetorical Classifier – Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "confusion_matrix.png"))
plt.close()

# -----------------------------
# METRIC BAR GRAPH
# -----------------------------
plt.figure()
plt.bar(metrics.keys(), metrics.values())
plt.title("Baseline Model Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(EVAL_DIR, "metrics_bar.png"))
plt.close()

# -----------------------------
# SAMPLE INFERENCE
# -----------------------------
sample_texts = [
    "The High Court held that the losses could not be deducted.",
    "The appellant contended that the transactions were inter-state sales.",
    "For these reasons, the appeal is dismissed."
]

sample_vec = vectorizer.transform(sample_texts)
sample_preds = model.predict(sample_vec)
sample_labels = le.inverse_transform(sample_preds)

pd.DataFrame({
    "sentence": sample_texts,
    "predicted_label": sample_labels
}).to_csv(os.path.join(OUT_DIR, "sample_predictions.csv"), index=False)

print("Baseline rhetorical classifier evaluation complete.")
print("Sample predictions saved in:", os.path.join(OUT_DIR, "sample_predictions.csv"))
