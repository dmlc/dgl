"""MXNet Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

from .... import broadcast_nodes, function as fn
from ....base import dgl_warning


class ChebConv(nn.Block):
    r"""Chebyshev Spectral Graph Convolution layer from `Convolutional Neural Networks on Graphs
    with Fast Localized Spectral Filtering <https://arxiv.org/pdf/1606.09375.pdf>`__

    .. math::
        h_i^{l+1} &= \sum_{k=0}^{K-1} W^{k, l}z_i^{k, l}

        Z^{0, l} &= H^{l}

        Z^{1, l} &= \tilde{L} \cdot H^{l}

        Z^{k, l} &= 2 \cdot \tilde{L} \cdot Z^{k-1, l} - Z^{k-2, l}

        \tilde{L} &= 2\left(I - \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}\right)/\lambda_{max} - I

    where :math:`\tilde{A}` is :math:`A` + :math:`I`, :math:`W` is learnable weight.


    Parameters
    ----------
    in_feats: int
        Dimension of input features; i.e, the number of dimensions of :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k : int
        Chebyshev filter size :math:`K`.
    activation : function, optional
        Activation function. Default ``ReLu``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from dgl.nn import ChebConv
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = ChebConv(10, 2, 2)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, feat)
    >>> res
    [[ 0.832592   -0.738757  ]
    [ 0.832592   -0.738757  ]
    [ 0.832592   -0.738757  ]
    [ 0.43377423 -1.0455742 ]
    [ 1.1145986  -0.5218046 ]
    [ 1.7954229   0.00196505]]
    <NDArray 6x2 @cpu(0)>
    """

    def __init__(self, in_feats, out_feats, k, bias=True):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        with self.name_scope():
            self.fc = nn.Sequential()
            for _ in range(k):
                self.fc.add(
                    nn.Dense(
                        out_feats,
                        use_bias=False,
                        weight_initializer=mx.init.Xavier(
                            magnitude=math.sqrt(2.0)
                        ),
                        in_units=in_feats,
                    )
                )
            if bias:
                self.bias = self.params.get(
                    "bias", shape=(out_feats,), init=mx.init.Zero()
                )
            else:
                self.bias = None

    def forward(self, graph, feat, lambda_max=None):
        r"""

        Description
        -----------
        Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.

            If None, this method would set the default value to 2.
            One can use :func:`dgl.laplacian_lambda_max` to compute this value.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        with graph.local_scope():
            degs = graph.in_degrees().astype("float32")
            norm = mx.nd.power(
                mx.nd.clip(degs, a_min=1, a_max=float("inf")), -0.5
            )
            norm = norm.expand_dims(-1).as_in_context(feat.context)

            if lambda_max is None:
                dgl_warning(
                    "lambda_max is not provided, using default value of 2.  "
                    "Please use dgl.laplacian_lambda_max to compute the eigenvalues."
                )
                lambda_max = [2] * graph.batch_size

            if isinstance(lambda_max, list):
                lambda_max = nd.array(lambda_max).as_in_context(feat.context)
            if lambda_max.ndim == 1:
                lambda_max = lambda_max.expand_dims(-1)
            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            # T0(X)
            Tx_0 = feat
            rst = self.fc[0](Tx_0)
            # T1(X)
            if self._k > 1:
                graph.ndata["h"] = Tx_0 * norm
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                h = graph.ndata.pop("h") * norm
                # Λ = 2 * (I - D ^ -1/2 A D ^ -1/2) / lambda_max - I
                #   = - 2(D ^ -1/2 A D ^ -1/2) / lambda_max + (2 / lambda_max - 1) I
                Tx_1 = -2.0 * h / lambda_max + Tx_0 * (2.0 / lambda_max - 1)
                rst = rst + self.fc[1](Tx_1)
            # Ti(x), i = 2...k
            for i in range(2, self._k):
                graph.ndata["h"] = Tx_1 * norm
                graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
                h = graph.ndata.pop("h") * norm
                # Tx_k = 2 * Λ * Tx_(k-1) - Tx_(k-2)
                #      = - 4(D ^ -1/2 A D ^ -1/2) / lambda_max Tx_(k-1) +
                #        (4 / lambda_max - 2) Tx_(k-1) -
                #        Tx_(k-2)
                Tx_2 = (
                    -4.0 * h / lambda_max + Tx_1 * (4.0 / lambda_max - 2) - Tx_0
                )
                rst = rst + self.fc[i](Tx_2)
                Tx_1, Tx_0 = Tx_2, Tx_1
            # add bias
            if self.bias is not None:
                rst = rst + self.bias.data(feat.context)
            return rst
