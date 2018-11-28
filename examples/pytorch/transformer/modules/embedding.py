import torch as T
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = T.zeros(max_len, dim_model, dtype=T.float)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(T.arange(0, dim_model, 2, dtype=T.float) *
                             -(np.log(10000.0) / dim_model))
        pe[:, 0::2] = T.sin(position * div_term)
        pe[:, 1::2] = T.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # Not a parameter but should be in state_dict

    def forward(self, pos):
        return T.index_select(self.pe, 1, pos).squeeze(0)

class Embeddings(nn.Module):
    def __init__(self, vocab_size, dim_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_size, dim_model)
        self.dim_model = dim_model

    def forward(self, x):
        return self.lut(x) * np.sqrt(self.dim_model)
