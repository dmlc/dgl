"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import mxnet.gluon.nn as nn

from dgl.nn.mxnet.conv import GATConv


class GAT(nn.Block):
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
        alpha,
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
                in_dim, num_hidden, heads[0], feat_drop, attn_drop, alpha, False
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
                    alpha,
                    residual,
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
                alpha,
                residual,
            )
        )
        for i, layer in enumerate(self.gat_layers):
            self.register_child(layer, "gat_layer_{}".format(i))

    def forward(self, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](self.g, h).flatten()
            h = self.activation(h)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits
