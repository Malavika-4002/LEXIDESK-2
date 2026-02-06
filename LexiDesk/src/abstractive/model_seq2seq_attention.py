# ============================================================
# Seq2Seq BiLSTM + Attention for Legal Abstractive Summarization
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Attention Module
# ------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        """
        decoder_hidden: (batch, hidden)
        encoder_outputs: (batch, seq_len, hidden*2)
        """

        seq_len = encoder_outputs.size(1)

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy = torch.tanh(
            self.attn(torch.cat((decoder_hidden, encoder_outputs), dim=2))
        )

        scores = self.v(energy).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e10)

        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        return context.squeeze(1), attn_weights


# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        outputs, (hidden, cell) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden, cell


# ------------------------------------------------------------
# Decoder
# ------------------------------------------------------------

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.attention = Attention(hidden_dim)
        self.lstm = nn.LSTM(
            emb_dim + hidden_dim * 2,
            hidden_dim,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim * 3, vocab_size)

    def forward(self, input_token, hidden, cell, encoder_outputs, mask):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)

        context, _ = self.attention(hidden[-1], encoder_outputs, mask)
        context = context.unsqueeze(1)

        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))

        output = output.squeeze(1)
        context = context.squeeze(1)

        prediction = self.fc_out(
            torch.cat((output, context), dim=1)
        )

        return prediction, hidden, cell


# ------------------------------------------------------------
# Seq2Seq Wrapper
# ------------------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, trg_len, vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src, src_lengths)

        input_token = trg[:, 0]

        mask = (src != self.pad_idx)

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input_token, hidden, cell, encoder_outputs, mask
            )

            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            input_token = trg[:, t] if teacher_force else output.argmax(1)

        return outputs
