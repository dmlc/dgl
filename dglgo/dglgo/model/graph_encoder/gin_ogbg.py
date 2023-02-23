import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import AvgPooling, GINEConv, SumPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class MLP(nn.Module):
    def __init__(self, feat_size: int):
        """Multilayer Perceptron (MLP)"""
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_size, 2 * feat_size),
            nn.BatchNorm1d(2 * feat_size),
            nn.ReLU(),
            nn.Linear(2 * feat_size, feat_size),
            nn.BatchNorm1d(feat_size),
        )

    def forward(self, h):
        return self.mlp(h)


class OGBGGIN(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = 300,
        num_layers: int = 5,
        dropout: float = 0.5,
        virtual_node: bool = False,
    ):
        """Graph Isomorphism Network (GIN) variant introduced in baselines
        for OGB graph property prediction datasets

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            Embedding size.
        num_layers : int
            Number of layers.
        dropout : float
            Dropout rate.
        virtual_node : bool
            Whether to use virtual node.
        """
        super(OGBGGIN, self).__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.virtual_node = virtual_node

        if data_info["name"] in ["ogbg-molhiv", "ogbg-molpcba"]:
            self.node_encoder = AtomEncoder(embed_size)
            self.edge_encoders = nn.ModuleList(
                [BondEncoder(embed_size) for _ in range(num_layers)]
            )
        else:
            # Handle other datasets
            self.node_encoder = nn.Linear(
                data_info["node_feat_size"], embed_size
            )
            self.edge_encoders = nn.ModuleList(
                [
                    nn.Linear(data_info["edge_feat_size"], embed_size)
                    for _ in range(num_layers)
                ]
            )

        self.conv_layers = nn.ModuleList(
            [GINEConv(MLP(embed_size)) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)
        self.pool = AvgPooling()
        self.pred = nn.Linear(embed_size, data_info["out_size"])

        if virtual_node:
            self.virtual_emb = nn.Embedding(1, embed_size)
            nn.init.constant_(self.virtual_emb.weight.data, 0)
            self.mlp_virtual = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.mlp_virtual.append(MLP(embed_size))
            self.virtual_pool = SumPooling()

    def forward(self, graph, node_feat, edge_feat):
        if self.virtual_node:
            virtual_emb = self.virtual_emb.weight.expand(graph.batch_size, -1)

        hn = self.node_encoder(node_feat)

        for layer in range(self.num_layers):

            if self.virtual_node:
                # messages from virtual nodes to graph nodes
                virtual_hn = dgl.broadcast_nodes(graph, virtual_emb)
                hn = hn + virtual_hn

            he = self.edge_encoders[layer](edge_feat)
            hn = self.conv_layers[layer](graph, hn, he)
            if layer != self.num_layers - 1:
                hn = F.relu(hn)
            hn = self.dropout(hn)

            if self.virtual_node and layer != self.num_layers - 1:
                # messages from graph nodes to virtual nodes
                virtual_emb_tmp = self.virtual_pool(graph, hn) + virtual_emb
                virtual_emb = self.mlp_virtual[layer](virtual_emb_tmp)
                virtual_emb = self.dropout(F.relu(virtual_emb))

        hg = self.pool(graph, hn)

        return self.pred(hg)
