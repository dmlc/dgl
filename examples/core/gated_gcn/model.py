import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GatedGCNConv
from dgl.nn.pytorch.glob import AvgPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class GatedGCN(nn.Module):
    def __init__(
        self,
        hid_dim,
        out_dim,
        num_layers,
        dropout=0.2,
        batch_norm=True,
        residual=True,
        activation=F.relu,
    ):
        super(GatedGCN, self).__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.node_encoder = AtomEncoder(hid_dim)
        self.edge_encoder = BondEncoder(hid_dim)

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = GatedGCNConv(
                input_feats=hid_dim,
                edge_feats=hid_dim,
                output_feats=hid_dim,
                dropout=dropout,
                batch_norm=batch_norm,
                residual=residual,
                activation=activation,
            )
            self.layers.append(layer)

        self.pooling = AvgPooling()
        self.output = nn.Linear(hid_dim, out_dim)

    def forward(self, g, edge_feat, node_feat):
        # Encode node and edge feature.
        hv = self.node_encoder(node_feat)
        he = self.edge_encoder(edge_feat)

        # GatedGCNConv layers.
        for layer in self.layers:
            hv, he = layer(g, hv, he)

        # Output project.
        h_g = self.pooling(g, hv)

        return self.output(h_g)
