import torch as th
import torch.nn as nn
from torch.nn import init

from ... import function as fn
from ...utils import get_ndata_name

class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    def __init__(self):
        super(SumPooling, self).__init__()
        self._feat_name = '_gpool_feat'

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[self._feat_name] = feat
        readout = dgl.sum_nodes(graph, self._feat_name)
        return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the graph.
    """
    def __init__(self):
        super(AvgPooling, self).__init__()
        self._feat_name = '_gpool_avg'

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[self._feat_name] = feat
        readout = dgl.mean_nodes(graph, self._feat_name)
        return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the graph.
    """
    def __init__(self):
        super(MaxPooling, self).__init__()
        self._feat_name = '_gpool_max'

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[self._feat_name] = feat
        readout = dgl.max_nodes(graph, self._feat_name)
        return readout


class TopKPooling(nn.Module):
    r"""Apply top-k pooling (from paper "Graph U-Net" and "Towards Sparse Hierarchical Graph
    Classifiers") over the graph.
    """
    def __init__(self):
        pass

    def forward(self, feat, graph):
        pass

class SortPooling(nn.Module):
    r"""Apply sort pooling (from paper "An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    def __init__(self):
        pass

    def forward(self, feat, graph):
        pass


class GlobAttnPooling(nn.Module):
    r"""Apply global attention pooling over the graph.
    """
    def __init__(self):
        pass


class DiffPool(nn.Module):
    r"""Apply Differentiable Pooling
    """
    def __init__(self):
        pass


class KNNPooling(nn.Module):
    pass


class SpectralClustering(nn.Module):
    pass


class Set2Set(nn.Module):
    r"""Apply
    """
    def __init__(self):
        pass
