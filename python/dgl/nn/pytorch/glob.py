""" Global pooling layers for DGL.
Three types:
1. One vector/K vectors per graph as readout.
2. Graph Coarsening, in this case we need to construct a new graph and a bipartite graph to bridge the current graph
   and the pooling graph.
"""

import torch as th
import torch.nn as nn
from torch.nn import init

from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes, softmax_nodes


class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    def __init__(self):
        super(SumPooling, self).__init__()
        self._feat_name = '_gpool_feat'

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[self._feat_name] = feat
        readout = sum_nodes(graph, self._feat_name)
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
        readout = mean_nodes(graph, self._feat_name)
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
        readout = max_nodes(graph, self._feat_name)
        return readout


class TopKPooling(nn.Module):
    r"""Apply top-k pooling (from paper "Graph U-Net" and "Towards Sparse Hierarchical Graph
    Classifiers") over the graph.
    """
    def __init__(self):
        super(TopKPooling, self).__init__()

    def forward(self, feat, graph):
        # TODO(zihao): finish this
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
        # TODO(zihao): reset parameters of gate_nn and nn
        pass

    def forward(self, feat, graph):
        feat = feat.unsqueeze(-1) if feat.dim() == 1 else feat
        gate = self.gate_nn(feat)
        feat = self.nn(feat) if self.nn else feat

        feat_name = get_ndata_name(graph, 'gate')
        graph.ndata[feat_name] = gate
        gate = softmax_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        feat_name = get_ndata_name(graph, 'readout')
        graph.ndata[feat_name] = feat * gate
        readout = sum_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        return readout


class DiffPool(nn.Module):
    r"""Apply Differentiable Pooling
    """
    def __init__(self):
        super(DiffPool, self).__init__()
        pass

    def forward(self, feat, graph):
        # TODO(zihao): finish this
        pass

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
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)

    def reset_parameters(self):
        # TODO(zihao): finish this
        pass

    def forward(self, feat, graph):
        self._feat_name = get_ndata_name(graph, self._feat_name)

        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        for i in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            score = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
            feat_name = get_ndata_name(graph, 'score')
            graph.ndata[feat_name] = score
            score = softmax_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            feat_name = get_ndata_name(graph, 'readout')
            graph.ndata[feat_name] = feat * score
            readout = sum_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            q_star = th.cat([q, readout], dim=-1)

        return q_star

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
