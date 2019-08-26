# -*- coding:utf-8 -*-
# pylint: disable=C0103, C0111, W0621
"""Implementation  of MGCN model"""

import dgl
import torch as th
import torch.nn as nn
from .layers import AtomEmbedding, RBFLayer, EdgeEmbedding, \
    MultiLevelInteraction


class MGCNModel(nn.Module):
    """
    MGCN Model from:
    Chengqiang Lu, et al.
    Molecular Property Prediction: A Multilevel
    Quantum Interactions Modeling Perspective. (AAAI'2019)
    """

    def __init__(self,
                 dim=128,
                 output_dim=1,
                 edge_dim=128,
                 cutoff=5.0,
                 width=1,
                 n_conv=3,
                 norm=False,
                 atom_ref=None,
                 pre_train=None):
        """
        Args:
            dim: dimension of feature maps
            out_put_dim: the num of target propperties to predict
            edge_dim: dimension of edge feature
            cutoff: the maximum distance between nodes
            width: width in the RBF layer
            n_conv: number of convolutional layers
            norm: normalization
            atom_ref: atom reference
                      used as the initial value of atom embeddings,
                      or set to None with random initialization
            pre_train: pre_trained node embeddings
        """
        super().__init__()
        self.name = "MGCN"
        self._dim = dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.cutoff = cutoff
        self.width = width
        self.n_conv = n_conv
        self.atom_ref = atom_ref
        self.norm = norm

        self.activation = nn.Softplus(beta=1, threshold=20)

        if atom_ref is not None:
            self.e0 = AtomEmbedding(1, pre_train=atom_ref)
        if pre_train is None:
            self.embedding_layer = AtomEmbedding(dim)
        else:
            self.embedding_layer = AtomEmbedding(pre_train=pre_train)
        self.edge_embedding_layer = EdgeEmbedding(dim=edge_dim)

        self.rbf_layer = RBFLayer(0, cutoff, width)

        self.conv_layers = nn.ModuleList([
            MultiLevelInteraction(self.rbf_layer._fan_out, dim)
            for i in range(n_conv)
        ])

        self.node_dense_layer1 = nn.Linear(dim * (self.n_conv + 1), 64)
        self.node_dense_layer2 = nn.Linear(64, output_dim)

    def set_mean_std(self, mean, std, device):
        self.mean_per_node = th.tensor(mean, device=device)
        self.std_per_node = th.tensor(std, device=device)

    def forward(self, g):

        self.embedding_layer(g, "node_0")
        if self.atom_ref is not None:
            self.e0(g, "e0")
        self.rbf_layer(g)

        self.edge_embedding_layer(g)

        for idx in range(self.n_conv):
            self.conv_layers[idx](g, idx + 1)

        node_embeddings = tuple(g.ndata["node_%d" % (i)]
                                for i in range(self.n_conv + 1))
        g.ndata["node"] = th.cat(node_embeddings, 1)

        # concat multilevel representations
        node = self.node_dense_layer1(g.ndata["node"])
        node = self.activation(node)
        res = self.node_dense_layer2(node)
        g.ndata["res"] = res

        if self.atom_ref is not None:
            g.ndata["res"] = g.ndata["res"] + g.ndata["e0"]

        if self.norm:
            g.ndata["res"] = g.ndata[
                "res"] * self.std_per_node + self.mean_per_node
        res = dgl.sum_nodes(g, "res")
        return res
