"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import dgl.function as fn
import tensorflow as tf
from dgl.nn import GATConv
from tensorflow.keras import layers


class GAT(tf.keras.Model):
    def __init__(
        self,
        g,
        num_layers,
        in_dim,
        num_hidden,
        num_classes,
        heads,
        activation,
        feat_drop,
        attn_drop,
        negative_slope,
        residual,
    ):
        super(GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = []
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                num_hidden,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                False,
                self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(
                GATConv(
                    num_hidden * heads[l - 1],
                    num_hidden,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                num_hidden * heads[-2],
                num_classes,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def call(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h)
            h = tf.reshape(h, (h.shape[0], -1))
        # output projection
        logits = tf.reduce_mean(self.gat_layers[-1](self.g, h), axis=1)
        return logits
