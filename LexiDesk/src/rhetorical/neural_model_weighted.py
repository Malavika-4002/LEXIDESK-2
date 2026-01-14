# src/rhetorical/neural_model_weighted.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Dataset
class RhetoricalDataset(Dataset):
    def __init__(self, sentences, labels, word2idx, max_len=50):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        tokens = [self.word2idx.get(w, 1) for w in self.sentences[idx].split()]  # 1=UNK
        tokens = tokens[:self.max_len]
        tokens += [0]*(self.max_len-len(tokens))  # 0=PAD
        return torch.tensor(tokens), torch.tensor(self.labels[idx])

# BiLSTM + Attention
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

# Training function
def train_neural_weighted(train_df, dev_df, save_dir='evaluation/rhetorical', epochs=5, batch_size=32):
    os.makedirs(save_dir, exist_ok=True)
    
    # Build vocab
    vocab = {'<PAD>':0, '<UNK>':1}
    for sent in train_df['sentence']:
        for w in sent.split():
            if w not in vocab:
                vocab[w] = len(vocab)
    
    train_dataset = RhetoricalDataset(train_df['sentence'].tolist(), train_df['label_enc'].tolist(), vocab)
    dev_dataset = RhetoricalDataset(dev_df['sentence'].tolist(), dev_df['label_enc'].tolist(), vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAttention(vocab_size=len(vocab), embed_dim=128, hidden_dim=64, num_classes=len(train_df['label'].unique()))
    model.to(device)
    
    # Compute class weights
    classes = np.unique(train_df['label_enc'])
    class_weights = compute_class_weight('balanced', classes=classes, y=train_df['label_enc'])
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs} done")
    
    # Evaluation
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
    report_df.to_csv(os.path.join(save_dir, 'neural_classification_report_weighted.csv'))
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Neural Confusion Matrix (Weighted)")
    plt.savefig(os.path.join(save_dir, 'neural_confusion_matrix_weighted.png'))
    
    # Save model
    torch.save(model.state_dict(), 'models/rhetorical/bilstm_attention_weighted.pt')
    
    print("Weighted Neural training completed. Metrics saved in", save_dir)
    return model, vocab
if __name__ == "__main__":
    # Paths
    train_path = 'data/processed/rhetorical/rhetorical_train_sbd.csv'
    dev_path = 'data/processed/rhetorical/rhetorical_dev_sbd.csv'
    eval_dir = 'evaluation/rhetorical'

    # Load data
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)

    # Encode labels
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    train_df['label_enc'] = le.fit_transform(train_df['label'])
    dev_df['label_enc'] = le.transform(dev_df['label'])

    # Train neural model
    train_neural_weighted(train_df, dev_df, save_dir=eval_dir, epochs=5, batch_size=32)
