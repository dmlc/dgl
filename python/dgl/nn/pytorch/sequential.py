"""Torch modules for graph sequential pooling."""
# pylint: disable= no-member, arguments-differ
import torch as th
import torch.nn as nn
import torch.functional as F

from ...utils import get_ndata_name, get_edata_name

class DiffPooling(nn.Module):
    r""" Differentiable Pooling (paper: Hierarchical Graph Representation Learning with
    Differentiable Pooling) layer.
    """
    def __init__(self, k, gnn):
        super(DiffPooling, self).__init__()
        self.k = k
        self.gnn = gnn

    def forward(self, feat, graph):
        # TODO(zihao): finish this
        pass


class TopKPooling(nn.Module):
    r"""Apply Top-K Pooling
    """
    def __init__(self, k):
        super(TopKPooling, self).__init__()
        self.k = k

    def forward(self, *input):
        # TODO(zihao): finish this
        pass
