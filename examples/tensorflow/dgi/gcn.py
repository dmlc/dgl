"""
This code was copied from the GCN implementation in DGL examples.
"""
import tensorflow as tf

from dgl.nn.tensorflow import GraphConv
from tensorflow.keras import layers


class GCN(layers.Layer):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layers = []
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = layers.Dropout(dropout)

    def call(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
