"""
This code was copied from the GCN implementation in DGL examples.
"""
import torch
import torch.nn as nn

from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
