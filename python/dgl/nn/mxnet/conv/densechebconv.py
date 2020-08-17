"""MXNet Module for DenseChebConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn


class DenseChebConv(nn.Block):
    r"""

    Description
    -----------
    Chebyshev Spectral Graph Convolution layer from paper `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__.

    We recommend to use this module when applying ChebConv on dense graphs.

    Parameters
    ----------
    in_feats: int
        Dimension of input features :math:`h_i^{(l)}`.
    out_feats: int
        Dimension of output features :math:`h_i^{(l+1)}`.
    k : int
        Chebyshev filter size.
    activation : function, optional
        Activation function, default is ReLu.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    See also
    --------
    `ChebConv <https://docs.dgl.ai/api/python/nn.pytorch.html#chebconv>`__
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 bias=True):
        super(DenseChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        with self.name_scope():
            self.fc = nn.Sequential()
            for _ in range(k):
                self.fc.add(
                    nn.Dense(out_feats, in_units=in_feats, use_bias=False,
                             weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)))
                )
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

    def forward(self, adj, feat, lambda_max=None):
        r"""

        Description
        -----------
        Compute (Dense) Chebyshev Spectral Graph Convolution layer.

        Parameters
        ----------
        adj : mxnet.NDArray
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(N, N)`, where a row represents the destination
            and a column represents the source.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : float or None, optional
            A float value indicates the largest eigenvalue of given graph.
            Default: None.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        A = adj.astype(feat.dtype).as_in_context(feat.context)
        num_nodes = A.shape[0]

        in_degree = 1. / nd.clip(A.sum(axis=1), 1, float('inf')).sqrt()
        D_invsqrt = nd.diag(in_degree)
        I = nd.eye(num_nodes, ctx=A.context)
        L = I - nd.dot(D_invsqrt, nd.dot(A, D_invsqrt))

        if lambda_max is None:
            # NOTE(zihao): this only works for directed graph.
            lambda_max = (nd.linalg.syevd(L)[1]).max()

        L_hat = 2 * L / lambda_max - I
        Z = [nd.eye(num_nodes, ctx=A.context)]
        Zh = self.fc[0](feat)
        for i in range(1, self._k):
            if i == 1:
                Z.append(L_hat)
            else:
                Z.append(2 * nd.dot(L_hat, Z[-1]) - Z[-2])
            Zh = Zh + nd.dot(Z[i], self.fc[i](feat))

        if self.bias is not None:
            Zh = Zh + self.bias.data(feat.context)
        return Zh
