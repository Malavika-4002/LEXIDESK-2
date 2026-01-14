# src/rhetorical/train_rhetorical.py

if __name__ == "__main__":
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    import os

    # Relative imports
    from .baseline_model import train_baseline
    from .neural_model import train_neural

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

    os.makedirs('models/rhetorical', exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    # Train Baseline
    print("=== Training Baseline Model ===")
    train_baseline(train_df, dev_df, save_dir=eval_dir)

    # Train Neural
    print("\n=== Training Neural Model ===")
    train_neural(train_df, dev_df, save_dir=eval_dir, epochs=5, batch_size=32)
