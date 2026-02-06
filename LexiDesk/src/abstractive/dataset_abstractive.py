# ============================================================
# PyTorch Dataset for Abstractive Summarization
# ============================================================

import json
import torch
from torch.utils.data import Dataset


class AbstractiveDataset(Dataset):
    def __init__(self, jsonl_path, vocab, max_src_len=800, max_tgt_len=200):
        self.samples = []
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.samples.append(obj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        src_ids = self.vocab.encode(
            item["input"], self.max_src_len
        )
        tgt_ids = self.vocab.encode(
            item["target"], self.max_tgt_len
        )

        src_len = sum(1 for i in src_ids if i != self.vocab.pad_idx)

        return {
            "src": torch.tensor(src_ids, dtype=torch.long),
            "src_len": torch.tensor(src_len, dtype=torch.long),
            "tgt": torch.tensor(tgt_ids, dtype=torch.long)
        }
