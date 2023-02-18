"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import tensorflow as tf

from dgl.nn.tensorflow import GraphConv
from tensorflow.keras import layers


class GCN(tf.keras.Model):
    def __init__(
        self, g, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super(GCN, self).__init__()
        self.g = g
        self.layer_list = []
        # input layer
        self.layer_list.append(
            GraphConv(in_feats, n_hidden, activation=activation)
        )
        # hidden layers
        for i in range(n_layers - 1):
            self.layer_list.append(
                GraphConv(n_hidden, n_hidden, activation=activation)
            )
        # output layer
        self.layer_list.append(GraphConv(n_hidden, n_classes))
        self.dropout = layers.Dropout(dropout)

    def call(self, features):
        h = features
        for i, layer in enumerate(self.layer_list):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
