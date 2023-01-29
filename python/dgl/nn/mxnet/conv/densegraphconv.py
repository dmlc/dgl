"""MXNet Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


class DenseGraphConv(nn.Block):
    """Graph Convolutional layer from `Semi-Supervised Classification with Graph
    Convolutional Networks <https://arxiv.org/abs/1609.02907>`__

    We recommend user to use this module when applying graph convolution on
    dense graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer. If is `'right'`, divide the aggregated messages
        by each node's in-degrees, which is equivalent to averaging the received messages.
        If is `'none'`, no normalization is applied. Default is `'both'`,
        where the :math:`c_{ij}` in the paper is applied.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Notes
    -----
    Zero in-degree nodes will lead to all-zero output. A common practice
    to avoid this is to add a self-loop for each node in the graph,
    which can be achieved by setting the diagonal of the adjacency matrix to be 1.

    See also
    --------
    `GraphConv <https://docs.dgl.ai/api/python/nn.pytorch.html#graphconv>`__
    """

    def __init__(
        self, in_feats, out_feats, norm="both", bias=True, activation=None
    ):
        super(DenseGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        with self.name_scope():
            self.weight = self.params.get(
                "weight",
                shape=(in_feats, out_feats),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
            )
            if bias:
                self.bias = self.params.get(
                    "bias", shape=(out_feats,), init=mx.init.Zero()
                )
            else:
                self.bias = None
            self._activation = activation

    def forward(self, adj, feat):
        r"""

        Description
        -----------
        Compute (Dense) Graph Convolution layer.

        Parameters
        ----------
        adj : mxnet.NDArray
            The adjacency matrix of the graph to apply Graph Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : mxnet.NDArray
            The input feature.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        adj = adj.astype(feat.dtype).as_in_context(feat.context)
        src_degrees = nd.clip(adj.sum(axis=0), a_min=1, a_max=float("inf"))
        dst_degrees = nd.clip(adj.sum(axis=1), a_min=1, a_max=float("inf"))
        feat_src = feat

        if self._norm == "both":
            norm_src = nd.power(src_degrees, -0.5)
            shp_src = norm_src.shape + (1,) * (feat.ndim - 1)
            norm_src = norm_src.reshape(shp_src).as_in_context(feat.context)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = nd.dot(feat_src, self.weight.data(feat_src.context))
            rst = nd.dot(adj, feat_src)
        else:
            # aggregate first then mult W
            rst = nd.dot(adj, feat_src)
            rst = nd.dot(rst, self.weight.data(feat_src.context))

        if self._norm != "none":
            if self._norm == "both":
                norm_dst = nd.power(dst_degrees, -0.5)
            else:  # right
                norm_dst = 1.0 / dst_degrees
            shp_dst = norm_dst.shape + (1,) * (feat.ndim - 1)
            norm_dst = norm_dst.reshape(shp_dst).as_in_context(feat.context)
            rst = rst * norm_dst

        if self.bias is not None:
            rst = rst + self.bias.data(feat.context)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
