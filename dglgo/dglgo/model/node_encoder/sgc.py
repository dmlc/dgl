import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl.base import dgl_warning
from dgl.nn import SGConv


class SGC(nn.Module):
    def __init__(self, data_info: dict, embed_size: int = -1, bias=True, k=2):
        """Simplifying Graph Convolutional Networks

        Edge feature is ignored in this model.

        Parameters
        ----------
        data_info : dict
            The information about the input dataset.
        embed_size : int
            The dimension of created embedding table. -1 means using original node embedding
        bias : bool
            If True, adds a learnable bias to the output. Default: ``True``.
        k : int
            Number of hops :math:`K`. Defaults:``1``.
        """
        super().__init__()
        self.data_info = data_info
        self.out_size = data_info["out_size"]
        self.embed_size = embed_size
        if embed_size > 0:
            self.embed = nn.Embedding(data_info["num_nodes"], embed_size)
            in_size = embed_size
        else:
            in_size = data_info["in_size"]
        self.sgc = SGConv(
            in_size,
            self.out_size,
            k=k,
            cached=True,
            bias=bias,
            norm=self.normalize,
        )

    def forward(self, g, node_feat, edge_feat=None):
        if self.embed_size > 0:
            dgl_warning(
                "The embedding for node feature is used, and input node_feat is ignored, due to the provided embed_size."
            )
            h = self.embed.weight
        else:
            h = node_feat
        return self.sgc(g, h)

    @staticmethod
    def normalize(h):
        return (h - h.mean(0)) / (h.std(0) + 1e-5)
