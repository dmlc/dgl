"""MXNet Module for DenseGraphConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn
from ....utils import expand_as_pair


class DenseGraphConv(nn.Block):
    """Graph Convolutional Network layer where the graph structure
    is given by an adjacency matrix.
    We recommend user to use this module when applying graph convolution on
    dense graphs.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : bool
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    See also
    --------
    GraphConv
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(DenseGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None
            self._activation = activation

    def forward(self, adj, feat):
        r"""Compute (Dense) Graph Convolution layer.

        Parameters
        ----------
        adj : mxnet.NDArray
            The adjacency matrix of the graph to apply Graph Convolution on, when
            applied to a unidirectional bipartite graph, ``adj`` should be of shape
            should be of shape :math:`(N_{out}, N_{in})`; when applied to a homo
            graph, ``adj`` should be of shape :math:`(N, N)`. In both cases,
            a row represents a destination node while a column represents a source
            node.
        feat : mxnet.NDArray or a pair of mxnet.NDArray
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        feat_src, feat_dst = expand_as_pair(feat)
        adj = adj.astype(feat.dtype).as_in_context(feat_src.context)

        if self._norm:
            src_degrees = adj.sum(axis=0)
            dst_degrees = adj.sum(axis=1)
            norm_src = nd.power(src_degrees, -0.5)
            norm_dst = nd.power(dst_degrees, -0.5)
            shp_src = norm_src.shape + (1,) * (feat_src.ndim - 1)
            shp_dst = norm_dst.shape + (1,) * (feat_dst.ndim - 1)
            norm_src = norm_src.reshape(shp_src).as_in_context(feat_src.context)
            norm_dst = norm_dst.reshape(shp_dst).as_in_context(feat_dst.context)
            feat_src = feat_src * norm_src

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat_src = nd.dot(feat_src, self.weight.data(feat_src.context))
            rst = nd.dot(adj, feat_src)
        else:
            # aggregate first then mult W
            rst = nd.dot(adj, feat_src)
            rst = nd.dot(rst, self.weight.data(feat_src.context))

        if self._norm:
            rst = rst * norm_dst

        if self.bias is not None:
            rst = rst + self.bias.data(feat.context)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst
