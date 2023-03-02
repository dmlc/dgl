"""MXNet module for TAGConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import gluon

from .... import function as fn


class TAGConv(gluon.Block):
    r"""Topology Adaptive Graph Convolutional layer from `Topology
    Adaptive Graph Convolutional Networks <https://arxiv.org/pdf/1710.10370.pdf>`__.

    .. math::
        H^{K} = {\sum}_{k=0}^K (D^{-1/2} A D^{-1/2})^{k} X {\Theta}_{k},

    where :math:`A` denotes the adjacency matrix,
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix,
    :math:`{\Theta}_{k}` denotes the linear weights to sum the results of different hops together.

    Parameters
    ----------
    in_feats : int
        Input feature size. i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Output feature size.  i.e, the number of dimensions of :math:`H^{K}`.
    k: int, optional
        Number of hops :math:`K`. Default: ``2``.
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    lin : torch.Module
        The learnable linear module.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> from dgl.nn import TAGConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = TAGConv(10, 2, k=2)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, feat)
    >>> res
    [[-0.86147034  0.10089529]
    [-0.86147034  0.10089529]
    [-0.86147034  0.10089529]
    [-0.9707841   0.0360311 ]
    [-0.6716844   0.02247889]
    [ 0.32964635 -0.7669234 ]]
    <NDArray 6x2 @cpu(0)>
    """

    def __init__(self, in_feats, out_feats, k=2, bias=True, activation=None):
        super(TAGConv, self).__init__()
        self.out_feats = out_feats
        self.k = k
        self.bias = bias
        self.activation = activation
        self.in_feats = in_feats

        self.lin = self.params.get(
            "weight",
            shape=(self.in_feats * (self.k + 1), self.out_feats),
            init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
        )
        if self.bias:
            self.h_bias = self.params.get(
                "bias", shape=(out_feats,), init=mx.init.Zero()
            )

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute topology adaptive graph convolution.

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
        with graph.local_scope():
            assert graph.is_homogeneous, "Graph is not homogeneous"

            degs = graph.in_degrees().astype("float32")
            norm = mx.nd.power(
                mx.nd.clip(degs, a_min=1, a_max=float("inf")), -0.5
            )
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = norm.reshape(shp).as_in_context(feat.context)

            rst = feat
            for _ in range(self.k):
                rst = rst * norm
                graph.ndata["h"] = rst

                graph.update_all(
                    fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
                )
                rst = graph.ndata["h"]
                rst = rst * norm
                feat = mx.nd.concat(feat, rst, dim=-1)

            rst = mx.nd.dot(feat, self.lin.data(feat.context))
            if self.bias is not None:
                rst = rst + self.h_bias.data(rst.context)

            if self.activation is not None:
                rst = self.activation(rst)

            return rst
