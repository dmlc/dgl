"""MXNet modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ
import mxnet as mx
from mxnet import gluon, nd
from mxnet import init
from mxnet.gluon import nn

from ... import function as fn
from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name
from ...base import dgl_warning
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes, softmax_nodes, topk_nodes

__all__ = ['SumPooling', 'AvgPooling', 'MaxPooling', 'SortPooling',
           'GlobAttnPooling', 'Set2Set']

class SumPooling(nn.Block):
    r"""Apply sum pooling over the graph.
    """
    _feat_name = '_gpool_feat'
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = sum_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class AvgPooling(nn.Block):
    r"""Apply average pooling over the graph.
    """
    _feat_name = '_gpool_avg'
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = mean_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class MaxPooling(nn.Block):
    r"""Apply max pooling over the graph.
    """
    _feat_name = '_gpool_max'
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = max_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class SortPooling(nn.Block):
    r"""Apply sort pooling (f"An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    _feat_name = '_gpool_sort'
    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, feat, graph):
        # Sort the feature of each node in ascending order.
        feat = feat.sort(axis=-1)
        graph.ndata[self._feat_name] = feat
        # Sort nodes according to their last features.
        ret = topk_nodes(graph, self._feat_name, self.k).reshape(-1, self.k * feat.shape[-1])
        graph.ndata.pop(self._feat_name)
        return ret


class GlobAttnPooling(nn.Block):
    r"""Apply global attention pooling over the graph.
    """
    _gate_name = '_gpool_attn_gate'
    _readout_name = '_gpool_attn_readout'
    def __init__(self, gate_nn, feat_nn=None):
        super(GlobAttnPooling, self).__init__()
        with self.name_scope():
            self.gate_nn = gate_nn
            self.feat_nn = feat_nn

        self.reset_parameters()

    def reset_parameters(self):
        self.gate_nn.initialize(mx.init.Xavier())
        if self.feat_nn:
            self.feat_nn.initialize(mx.init.Xavier())

    def forward(self, feat, graph):
        gate = self.gate_nn(feat)
        assert gate.shape[-1] == 1, "The output of gate network shoule have size 1 at the last axis."
        feat = self.feat_nn(feat) if self.feat_nn else feat

        feat_name = get_ndata_name(graph, self._gate_name)
        graph.ndata[feat_name] = gate
        gate = softmax_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        feat_name = get_ndata_name(graph, self._readout_name)
        graph.ndata[feat_name] = feat * gate
        readout = sum_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        return readout


class Set2Set(nn.Block):
    r"""Apply Set2Set (f"Order Matters: Sequence to sequence for sets") over the graph.
    """
    _score_name = '_gpool_s2s_score'
    _readout_name = '_gpool_s2s_readout'
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        with self.name_scope():
            self.lstm = gluon.rnn.LSTM(self.input_dim, num_layers=n_layers, input_size=self.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.initialize(mx.init.Xavier())

    def forward(self, feat, graph):
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (nd.zeros((self.n_layers, batch_size, self.input_dim), ctx=feat.context),
             nd.zeros((self.n_layers, batch_size, self.input_dim), ctx=feat.context))
        q_star = nd.zeros((batch_size, self.output_dim), ctx=feat.context)

        for i in range(self.n_iters):
            q, h = self.lstm(q_star.expand_dims(axis=0), h)
            q = q.reshape((batch_size, self.input_dim))

            score = (feat * broadcast_nodes(graph, q)).sum(axis=-1, keepdims=True)
            feat_name = get_ndata_name(graph, self._score_name)
            graph.ndata[feat_name] = score
            score = softmax_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            feat_name = get_ndata_name(graph, self._readout_name)
            graph.ndata[feat_name] = feat * score
            readout = sum_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            q_star = nd.concat(q, readout, dim=-1)

        return q_star

