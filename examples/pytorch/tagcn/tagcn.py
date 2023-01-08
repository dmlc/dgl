"""TAGCN using DGL nn package

References:
- Topology Adaptive Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1710.10370
"""
import torch
import torch.nn as nn

from dgl.nn.pytorch.conv import TAGConv


class TAGCN(nn.Module):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(TAGCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(TAGConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                TAGConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(TAGConv(n_hidden, n_classes))  # activation=None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
