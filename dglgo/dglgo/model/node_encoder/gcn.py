import dgl
import torch
import torch.nn as nn
from dgl.base import dgl_warning


class GCN(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = -1,
        hidden_size: int = 16,
        num_layers: int = 1,
        norm: str = "both",
        activation: str = "relu",
        dropout: float = 0.5,
        use_edge_weight: bool = False,
    ):
        """Graph Convolutional Networks

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
        dropout : float
            Dropout rate.
        use_edge_weight : bool
            If true, scale the messages by edge weights.
        """
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.data_info = data_info
        self.embed_size = embed_size
        self.layers = nn.ModuleList()
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]

        for i in range(num_layers):
            in_hidden = hidden_size if i > 0 else in_size
            out_hidden = (
                hidden_size if i < num_layers - 1 else data_info["out_size"]
            )

            self.layers.append(
                dgl.nn.GraphConv(
                    in_hidden, out_hidden, norm=norm, allow_zero_in_degree=True
                )
            )

        self.dropout = nn.Dropout(p=dropout)
        self.act = getattr(torch, activation)

    def forward(self, g, node_feat, edge_feat=None):
        if self.embed_size > 0:
            dgl_warning(
                "The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size."
            )
            h = self.embed.weight
        else:
            h = node_feat
        edge_weight = edge_feat if self.use_edge_weight else None
        for l, layer in enumerate(self.layers):
            h = layer(g, h, edge_weight=edge_weight)
            if l != len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h

    def forward_block(self, blocks, node_feat, edge_feat=None):
        h = node_feat
        edge_weight = edge_feat if self.use_edge_weight else None
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h, edge_weight=edge_weight)
            if l != len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)
        return h
