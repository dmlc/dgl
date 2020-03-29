"""MXNet module for TAGConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import gluon

from .... import function as fn


class TAGConv(gluon.Block):
    r"""Apply Topology Adaptive Graph Convolutional Network

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^K \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}\mathbf{X} \mathbf{\Theta}_{k},

    where :math:`\mathbf{A}` denotes the adjacency matrix and
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k: int, optional
        Number of hops :math: `k`. (default: 2)
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    lin : mxnet.gluon.parameter.Parameter
        The learnable weight tensor.
    bias : mxnet.gluon.parameter.Parameter
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None):
        super(TAGConv, self).__init__()
        self.out_feats = out_feats
        self.k = k
        self.bias = bias
        self.activation = activation
        self.in_feats = in_feats

        self.lin = self.params.get(
            'weight', shape=(self.in_feats * (self.k + 1), self.out_feats),
            init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        if self.bias:
            self.h_bias = self.params.get('bias', shape=(out_feats,),
                                          init=mx.init.Zero())

    def forward(self, graph, feat):
        r"""Compute graph convolution

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        assert graph.is_homograph(), 'Graph is not homogeneous'
        graph = graph.local_var()

        degs = graph.in_degrees().astype('float32')
        norm = mx.nd.power(mx.nd.clip(degs, a_min=1, a_max=float("inf")), -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = norm.reshape(shp).as_in_context(feat.context)

        rst = feat
        for _ in range(self.k):
            rst = rst * norm
            graph.ndata['h'] = rst

            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = rst * norm
            feat = mx.nd.concat(feat, rst, dim=-1)

        rst = mx.nd.dot(feat, self.lin.data(feat.context))
        if self.bias is not None:
            rst = rst + self.h_bias.data(rst.context)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst
