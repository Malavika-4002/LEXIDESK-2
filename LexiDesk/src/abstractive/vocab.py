# ============================================================
# Vocabulary and Tokenization for Legal Abstractive Summarizer
# ============================================================

import re
from collections import Counter


class Vocabulary:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq

        self.pad_token = "<pad>"
        self.sos_token = "<sos>"
        self.eos_token = "<eos>"
        self.unk_token = "<unk>"

        self.special_tokens = [
            self.pad_token,
            self.sos_token,
            self.eos_token,
            self.unk_token
        ]

        self.word2idx = {}
        self.idx2word = {}
        self.freqs = Counter()

    # --------------------------------------------------------
    # Basic legal-text tokenizer
    # --------------------------------------------------------
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)   # remove <s> </s>
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        tokens = text.split()
        return tokens

    # --------------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------------
    def build(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            self.freqs.update(tokens)

        self.word2idx = {tok: i for i, tok in enumerate(self.special_tokens)}
        idx = len(self.word2idx)

        for word, freq in self.freqs.items():
            if freq >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1

        self.idx2word = {i: w for w, i in self.word2idx.items()}

    # --------------------------------------------------------
    # Encode text → indices
    # --------------------------------------------------------
    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        tokens = [self.sos_token] + tokens[: max_len - 2] + [self.eos_token]

        ids = [
            self.word2idx.get(tok, self.word2idx[self.unk_token])
            for tok in tokens
        ]

        pad_len = max_len - len(ids)
        ids += [self.word2idx[self.pad_token]] * pad_len

        return ids

    # --------------------------------------------------------
    # Properties
    # --------------------------------------------------------
    @property
    def pad_idx(self):
        return self.word2idx[self.pad_token]

    @property
    def size(self):
        return len(self.word2idx)
