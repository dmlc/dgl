
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn import SGConv


class SGC(nn.Module):
    def __init__(self, in_size, out_size,
                 bias=True, k=2):
        super().__init__()
        self.sgc = SGConv(in_size, out_size, k=k, cached=True, bias=bias)

    def forward(self, g, node_feat, edge_feat=None):
        return self.sgc(g, node_feat)
