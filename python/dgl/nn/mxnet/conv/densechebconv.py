"""MXNet Module for DenseChebConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
import math
from mxnet import nd
from mxnet.gluon import nn


class DenseChebConv(nn.Block):
    r"""Chebyshev Spectral Graph Convolution layer from paper `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__.

    We recommend to use this module when inducing ChebConv operations on dense
    graphs / k-hop graphs.

    Parameters
    ----------
    in_feats: int
        Number of input features.
    out_feats: int
        Number of output features.
    k : int
        Chebyshev filter size.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    See also
    --------
    ChebConv
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
                             weight_initializer=mx.init.Xavier(math.sqrt(2.0)))
                )
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

    def forward(self, adj, feat, lambda_max=None):
        r"""Compute (Dense) Chebyshev Spectral Graph Convolution layer.

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
        pass