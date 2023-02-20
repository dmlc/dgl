from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.base import dgl_warning
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = -1,
        num_layers: int = 2,
        hidden_size: int = 8,
        heads: List[int] = [8, 8],
        activation: str = "elu",
        feat_drop: float = 0.6,
        attn_drop: float = 0.6,
        negative_slope: float = 0.2,
        residual: bool = False,
    ):
        """Graph Attention Networks

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            The dimension of created embedding table. -1 means using original node embedding
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of layers.
        norm : str
            GCN normalization type. Can be 'both', 'right', 'left', 'none'.
        activation : str
            Activation function.
        feat_drop : float
            Dropout rate for features.
        attn_drop : float
            Dropout rate for attentions.
        negative_slope : float
            Negative slope for leaky relu in GATConv
        residual : bool
            If true, the GATConv will use residule connection
        """
        super(GAT, self).__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = getattr(torch.nn.functional, activation)

        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]

        for i in range(num_layers):
            in_hidden = hidden_size * heads[i - 1] if i > 0 else in_size
            out_hidden = (
                hidden_size if i < num_layers - 1 else data_info["out_size"]
            )
            activation = None if i == num_layers - 1 else self.activation

            self.gat_layers.append(
                GATConv(
                    in_hidden,
                    out_hidden,
                    heads[i],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    activation,
                )
            )

    def forward(self, graph, node_feat, edge_feat=None):
        if self.embed_size > 0:
            dgl_warning(
                "The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size."
            )
            h = self.embed.weight
        else:
            h = node_feat
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](graph, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits

    def forward_block(self, blocks, node_feat, edge_feat=None):
        h = node_feat
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](blocks[l], h).flatten(1)
        logits = self.gat_layers[-1](blocks[-1], h).mean(1)
        return logits
