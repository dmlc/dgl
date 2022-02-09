
import torch
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn import SGConv


class SGC(nn.Module):
    def __init__(self, in_size, out_size,
                 bias=True, k=2):
        """ Simplifying Graph Convolutional Networks

        Parameters
        ----------
        in_size : int 
            Number of input features.
        out_size : int
            Output size.
        bias : bool
            If True, adds a learnable bias to the output. Default: ``True``.
        k : int
            Number of hops :math:`K`. Defaults:``1``.
        """
        super().__init__()
        self.sgc = SGConv(in_size, out_size, k=k, cached=True,
                          bias=bias, norm=self.normalize)

    def forward(self, g, node_feat, edge_feat=None):
        return self.sgc(g, node_feat)

    @staticmethod
    def normalize(h):
        return (h-h.mean(0))/(h.std(0) + 1e-5)
