"""GCN using DGL nn package
References:
- Simple and Deep Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1810.05997
- Code: https://github.com/chennnM/GCNII
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConvII

class GCNII(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size,
                 num_layers,
                 alpha,
                 lamda,
                 dropout,
                 norm=True,
                 weight=True,
                 bias=False,
                 activation=None):
        super(GCNII, self).__init__()
        self._activation = activation
        self.dense_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        self.dense_layers.append(nn.Linear(in_size, hidden_size))
        for i in range(num_layers):
            self.conv_layers.append(GraphConvII(hidden_size,
                                                alpha,
                                                lamda,
                                                norm,
                                                weight,
                                                bias,
                                                activation))
        self.dense_layers.append(nn.Linear(hidden_size, out_size))

        self.params_dense = list(self.dense_layers.parameters())
        self.params_conv = list(self.conv_layers.parameters())
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, graph, features):
        h = self.dropout(features)
        h = self.dense_layers[0](h)
        h = self._activation(h)
        initial_feat = h
        h = self.dropout(h)
        for i, layer in enumerate(self.conv_layers):
            h = self.dropout(layer(graph, h, initial_feat, i+1))
        h = self.dense_layers[-1](h)
        return h
