"""MXNet modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
from mxnet import gluon, nd
from mxnet.gluon import nn

from ...readout import (
    broadcast_nodes,
    max_nodes,
    mean_nodes,
    softmax_nodes,
    sum_nodes,
    topk_nodes,
)

__all__ = [
    "SumPooling",
    "AvgPooling",
    "MaxPooling",
    "SortPooling",
    "GlobalAttentionPooling",
    "Set2Set",
]


class SumPooling(nn.Block):
    r"""Apply sum pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """

    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute sum pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = sum_nodes(graph, "h")
            graph.ndata.pop("h")
            return readout

    def __repr__(self):
        return "SumPooling()"


class AvgPooling(nn.Block):
    r"""Apply average pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k
    """

    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute average pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = mean_nodes(graph, "h")
            graph.ndata.pop("h")
            return readout

    def __repr__(self):
        return "AvgPooling()"


class MaxPooling(nn.Block):
    r"""Apply max pooling over the nodes in the graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i} \left( x^{(i)}_k \right)
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata["h"] = feat
            readout = max_nodes(graph, "h")
            graph.ndata.pop("h")
            return readout

    def __repr__(self):
        return "MaxPooling()"


class SortPooling(nn.Block):
    r"""Pooling layer from `An End-to-End Deep Learning Architecture for Graph Classification
    <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`__

    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.
    """

    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, graph, feat):
        r"""Compute sort pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input node feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, k * D)`, where
            :math:`B` refers to the batch size.
        """
        # Sort the feature of each node in ascending order.
        with graph.local_scope():
            feat = feat.sort(axis=-1)
            graph.ndata["h"] = feat
            # Sort nodes according to their last features.
            ret = topk_nodes(graph, "h", self.k, sortby=-1)[0].reshape(
                -1, self.k * feat.shape[-1]
            )
            return ret

    def __repr__(self):
        return "SortPooling(k={})".format(self.k)


class GlobalAttentionPooling(nn.Block):
    r"""Global Attention Pooling layer from `Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493.pdf>`__

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : gluon.nn.Block
        A neural network that computes attention scores for each feature.
    feat_nn : gluon.nn.Block, optional
        A neural network applied to each feature before combining them
        with attention scores.
    """

    def __init__(self, gate_nn, feat_nn=None):
        super(GlobalAttentionPooling, self).__init__()
        with self.name_scope():
            self.gate_nn = gate_nn
            self.feat_nn = feat_nn

    def forward(self, graph, feat):
        r"""Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input node feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            gate = self.gate_nn(feat)
            assert (
                gate.shape[-1] == 1
            ), "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat) if self.feat_nn else feat

            graph.ndata["gate"] = gate
            gate = softmax_nodes(graph, "gate")

            graph.ndata["r"] = feat * gate
            readout = sum_nodes(graph, "r")

            return readout


class Set2Set(nn.Block):
    r"""Set2Set operator from `Order Matters: Sequence to sequence for sets
    <https://arxiv.org/pdf/1511.06391.pdf>`__

    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Parameters
    ----------
    input_dim : int
        Size of each input sample
    n_iters : int
        Number of iterations.
    n_layers : int
        Number of recurrent layers.
    """

    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        with self.name_scope():
            self.lstm = gluon.rnn.LSTM(
                self.input_dim, num_layers=n_layers, input_size=self.output_dim
            )

    def forward(self, graph, feat):
        r"""Compute set2set pooling.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input node feature with shape :math:`(N, D)` where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (
                nd.zeros(
                    (self.n_layers, batch_size, self.input_dim),
                    ctx=feat.context,
                ),
                nd.zeros(
                    (self.n_layers, batch_size, self.input_dim),
                    ctx=feat.context,
                ),
            )
            q_star = nd.zeros((batch_size, self.output_dim), ctx=feat.context)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.expand_dims(axis=0), h)
                q = q.reshape((batch_size, self.input_dim))
                e = (feat * broadcast_nodes(graph, q)).sum(
                    axis=-1, keepdims=True
                )
                graph.ndata["e"] = e
                alpha = softmax_nodes(graph, "e")
                graph.ndata["r"] = feat * alpha
                readout = sum_nodes(graph, "r")
                q_star = nd.concat(q, readout, dim=-1)

            return q_star

    def __repr__(self):
        summary = "Set2Set("
        summary += "in={}, out={}, " "n_iters={}, n_layers={}".format(
            self.input_dim, self.output_dim, self.n_iters, self.n_layers
        )
        summary += ")"
        return summary
