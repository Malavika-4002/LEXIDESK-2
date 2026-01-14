# src/rhetorical/baseline_model.py
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def train_baseline(train_df, dev_df, save_dir='evaluation/rhetorical'):
    os.makedirs(save_dir, exist_ok=True)
    
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train = vectorizer.fit_transform(train_df['sentence'])
    X_dev = vectorizer.transform(dev_df['sentence'])
    
    y_train = train_df['label_enc']
    y_dev = dev_df['label_enc']
    labels = list(train_df['label'].unique())
    
    # Logistic Regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Predictions
    y_pred = clf.predict(X_dev)
    
    # Classification report
    report = classification_report(y_dev, y_pred, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(save_dir, 'baseline_classification_report.csv'), index=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_dev, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Baseline Confusion Matrix")
    plt.savefig(os.path.join(save_dir, 'baseline_confusion_matrix.png'))
    
    # Save model & vectorizer
    joblib.dump(clf, 'models/rhetorical/baseline_lr_model.joblib')
    joblib.dump(vectorizer, 'models/rhetorical/baseline_tfidf_vectorizer.joblib')
    
    print("Baseline training completed. Metrics saved in", save_dir)
    return clf, vectorizer
