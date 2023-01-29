"""MXNet Module for DenseGraphSAGE"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from ....utils import check_eq_shape


class DenseSAGEConv(nn.Block):
    """GraphSAGE layer from `Inductive Representation Learning on Large Graphs
    <https://arxiv.org/abs/1706.02216>`__

    We recommend to use this module when appying GraphSAGE on dense graphs.

    Note that we only support gcn aggregator in DenseSAGEConv.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    feat_drop : float, optional
        Dropout rate on features. Default: 0.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    See also
    --------
    `SAGEConv <https://docs.dgl.ai/api/python/nn.pytorch.html#sageconv>`__
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        feat_drop=0.0,
        bias=True,
        norm=None,
        activation=None,
    ):
        super(DenseSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        with self.name_scope():
            self.feat_drop = nn.Dropout(feat_drop)
            self.activation = activation
            self.fc = nn.Dense(
                out_feats,
                in_units=in_feats,
                use_bias=bias,
                weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
            )

    def forward(self, adj, feat):
        r"""

        Description
        -----------
        Compute (Dense) Graph SAGE layer.

        Parameters
        ----------
        adj : mxnet.NDArray
            The adjacency matrix of the graph to apply SAGE Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : mxnet.NDArray or a pair of mxnet.NDArray
            If a mxnet.NDArray is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of mxnet.NDArray is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        check_eq_shape(feat)
        if isinstance(feat, tuple):
            feat_src = self.feat_drop(feat[0])
            feat_dst = self.feat_drop(feat[1])
        else:
            feat_src = feat_dst = self.feat_drop(feat)
        adj = adj.astype(feat_src.dtype).as_in_context(feat_src.context)
        in_degrees = adj.sum(axis=1, keepdims=True)
        h_neigh = (nd.dot(adj, feat_src) + feat_dst) / (in_degrees + 1)
        rst = self.fc(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)

        return rst
