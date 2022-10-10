import torch as th
import torch.nn as nn
from torch.nn import LayerNorm


class Generator(nn.Module):
    """
    Generate next token from the representation. This part is separated from the decoder, mostly for the convenience of sharing weight between embedding and generator.
    log(softmax(Wx + b))
    """

    def __init__(self, dim_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(dim_model, vocab_size)

    def forward(self, x):
        return th.log_softmax(self.proj(x), dim=-1)


class SubLayerWrapper(nn.Module):
    """
    The module wraps normalization, dropout, residual connection into one equation:
    sublayerwrapper(sublayer)(x) = x + dropout(sublayer(norm(x)))
    """

    def __init__(self, size, dropout):
        super(SubLayerWrapper, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    """
    This module implements feed-forward network(after the Multi-Head Network) equation:
    FFN(x) = max(0, x @ W_1 + b_1) @ W_2 + b_2
    """

    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(dim_model, dim_ff)
        self.w_2 = nn.Linear(dim_ff, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(th.relu(self.w_1(x))))


import copy


def clones(module, k):
    return nn.ModuleList(copy.deepcopy(module) for _ in range(k))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # (key, query, value, mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 2)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn  # (key, query, value, mask)
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SubLayerWrapper(size, dropout), 3)
