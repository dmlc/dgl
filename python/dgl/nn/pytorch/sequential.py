"""Torch modules for graph sequential pooling."""
# pylint: disable= no-member, arguments-differ
import torch as th
import torch.nn as nn
import torch.functional as F

from ...utils import get_ndata_name, get_edata_name

class DiffPool(nn.Module):
    r"""Apply Differentiable Pooling
    """
    def __init__(self):
        super(DiffPool, self).__init__()
        pass

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
