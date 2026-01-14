# src/rhetorical/neural_model_advanced.py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

# ---------------------------
# Focal Loss
# ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, target):
        logpt = -self.ce(logits, target)
        pt = torch.exp(logpt)
        loss = ((1 - pt) ** self.gamma) * -logpt
        return loss.mean()

# ---------------------------
# Dataset with optional augmentation
# ---------------------------
class RhetoricalDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, max_len=50, augment=False):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.sentences)

    def synonym_replace(self, sentence):
        # Very simple augmentation: swap random words for "<UNK>" token
        tokens = sentence.split()
        for _ in range(max(1, len(tokens)//5)):
            idx = random.randint(0, len(tokens)-1)
            tokens[idx] = "<UNK>"
        return " ".join(tokens)

    def __getitem__(self, idx):
        sent = self.sentences[idx]
        label = self.labels[idx]
        if self.augment and label in [i for i in range(100)]:  # optional: only minority classes
            sent = self.synonym_replace(sent)
        tokens = [self.word2idx.get(w, 1) for w in sent.split()]  # 1=UNK
        tokens = tokens[:self.max_len]
        tokens += [0]*(self.max_len - len(tokens))
        return torch.tensor(tokens), torch.tensor(label)

# ---------------------------
# BiLSTM + Attention
# ---------------------------
class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim*2, 1)
        self.fc = nn.Linear(hidden_dim*2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        attn_weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        return self.fc(context)

# ---------------------------
# Training function
# ---------------------------
def train_neural_advanced(train_df, dev_df, save_dir='evaluation/rhetorical', 
                          epochs=10, batch_size=32, max_len=50, embed_dim=128, hidden_dim=64):

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('models/rhetorical', exist_ok=True)

    # Build vocab
    vocab = {'<PAD>':0, '<UNK>':1}
    for sent in train_df['sentence']:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = len(vocab)

    # ---------------------------
    # Weighted Random Sampler for minority classes
    # ---------------------------
    class_sample_count = train_df['label_enc'].value_counts().sort_index().to_numpy()
    weights = 1.0 / class_sample_count
    samples_weight = np.array([weights[t] for t in train_df['label_enc']])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    # ---------------------------
    # Datasets
    # ---------------------------
    train_dataset = RhetoricalDataset(train_df['sentence'].tolist(),
                                      train_df['label_enc'].tolist(),
                                      vocab,
                                      max_len=max_len,
                                      augment=True)
    dev_dataset = RhetoricalDataset(dev_df['sentence'].tolist(),
                                    dev_df['label_enc'].tolist(),
                                    vocab,
                                    max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAttention(vocab_size=len(vocab), embed_dim=embed_dim,
                            hidden_dim=hidden_dim, num_classes=len(train_df['label'].unique()))
    model.to(device)

    # ---------------------------
    # Class weights + Focal Loss
    # ---------------------------
    classes = np.unique(train_df['label_enc'])
    class_weights = compute_class_weight('balanced', classes=classes, y=train_df['label_enc'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = FocalLoss(gamma=2.0, weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------
    # Training loop
    # ---------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    # ---------------------------
    # Evaluation
    # ---------------------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in dev_loader:
            x = x.to(device)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.tolist())

    labels = list(train_df['label'].unique())
    report = classification_report(all_labels, all_preds, target_names=labels, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join(save_dir, 'neural_classification_report_advanced.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Neural Confusion Matrix (Advanced)")
    cm_path = os.path.join(save_dir, 'neural_confusion_matrix_advanced.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Save model
    model_path = 'models/rhetorical/bilstm_attention_advanced.pt'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    return model, vocab

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Paths
    train_path = 'data/processed/rhetorical/rhetorical_train_sbd.csv'
    dev_path = 'data/processed/rhetorical/rhetorical_dev_sbd.csv'
    eval_dir = 'evaluation/rhetorical'

    # Load data
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    # Encode labels
    le = LabelEncoder()
    train_df['label_enc'] = le.fit_transform(train_df['label'])
    dev_df['label_enc'] = le.transform(dev_df['label'])

    # Train
    train_neural_advanced(train_df, dev_df, save_dir=eval_dir, epochs=10, batch_size=32)
