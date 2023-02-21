import torch.nn as nn
from dgl.base import dgl_warning
from dgl.nn import GINConv


class GIN(nn.Module):
    def __init__(
        self,
        data_info: dict,
        embed_size: int = -1,
        hidden_size=64,
        num_layers=3,
        aggregator_type="sum",
    ):
        """Graph Isomophism Networks

        Edge feature is ignored in this model.

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
        aggregator_type : str
            Aggregator type to use (``sum``, ``max`` or ``mean``), default: 'sum'.
        """
        super().__init__()
        self.data_info = data_info
        self.embed_size = embed_size
        self.conv_list = nn.ModuleList()
        self.num_layers = num_layers
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]
        for i in range(num_layers):
            input_dim = in_size if i == 0 else hidden_size
            mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )

            self.conv_list.append(GINConv(mlp, aggregator_type, 1e-5, True))
        self.out_mlp = nn.Linear(hidden_size, data_info["out_size"])

    def forward(self, graph, node_feat, edge_feat=None):
        if self.embed_size > 0:
            dgl_warning(
                "The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size."
            )
            h = self.embed.weight
        else:
            h = node_feat
        for i in range(self.num_layers):
            h = self.conv_list[i](graph, h)
        h = self.out_mlp(h)
        return h
