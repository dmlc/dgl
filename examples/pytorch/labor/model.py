import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.functional as F
import torch.nn as nn
import tqdm

import dgl
import dgl.nn as dglnn


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, edge_weight=block.edata['edge_weights'] if 'edge_weights' in block.edata else None)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h
