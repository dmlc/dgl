"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self,
                 g, layer,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(layer(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers):
            self.layers.append(layer(n_hidden, n_hidden, activation=activation))
        # output layer
        self.layers.append(layer(n_hidden, n_classes, activation=None))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
