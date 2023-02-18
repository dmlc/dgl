"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import dgl
import mxnet as mx
from dgl.nn.mxnet import GraphConv
from mxnet import gluon


class GCN(gluon.Block):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = gluon.nn.Sequential()
        # input layer
        self.layers.add(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.add(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.add(GraphConv(n_hidden, n_classes))
        self.dropout = gluon.nn.Dropout(rate=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
