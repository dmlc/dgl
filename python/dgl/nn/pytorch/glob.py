""" Global pooling layers for DGL.
Three types:
1. One vector/K vectors per graph as readout.
2. Graph Coarsening, in this case we need to construct a new graph and a bipartite graph to bridge the current graph
   and the pooling graph.
"""

import torch as th
import torch.nn as nn
from torch.nn import init

from python.dgl.heterograph import make_bipartite
from ... import function as fn, BatchedDGLGraph
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
        super(TopKPooling, self).__init__()

    def forward(self, feat, graph):
        pass


class GlobAttnPooling(nn.Module):
    r"""Apply global attention pooling over the graph.
    """
    def __init__(self, gate_nn, nn=None):
        super(GlobAttnPooling, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        # TODO: reset parameters of gate_nn and nn
        pass

    def forward(self, feat, graph):
        feat = feat.unsqueeze(-1) if feat.dim() == 1 else feat
        gate = self.gate_nn(feat)
        feat = self.nn(feat) if self.nn else feat

        # TODO: softmax
        pass

class DiffPool(nn.Module):
    r"""Apply Differentiable Pooling
    """
    def __init__(self):
        super(DiffPool, self).__init__()
        pass

    def forward(self, feat, graph):
        # TODO: compute the features and graph at next level: feat_next, graph_next
        feat_next = None
        graph_next = None

        return feat_next, graph_next, make_bipartite(graph, graph_next)


class Set2Set(nn.Module):
    r"""Apply Set2Set (from paper "Order Matters: Sequence to sequence for sets") over the graph.
    """
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self._feat_name = '_gpool_set2set'

        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers= n_layers
        self.rnn = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)

    def reset_parameters(self):
        # TODO
        pass

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)

        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        # TODO: we also need a graph level softmax here.

class SortPooling(nn.Module):
    r"""Apply sort pooling (from paper "An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    def __init__(self):
        pass

    def forward(self, feat, graph):
        pass


class SpectralClustering(nn.Module):
    pass
