"""Tensorflow Module for DenseChebConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class DenseChebConv(layers.Layer):
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
    `ChebConv <https://docs.dgl.ai/api/python/nn.tensorflow.html#chebconv>`__
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

        # keras initializer assume last two dims as fan_in and fan_out
        xinit = tf.keras.initializers.glorot_normal()
        self.W = tf.Variable(initial_value=xinit(
            shape=(k, in_feats, out_feats), dtype='float32'), trainable=True)

        if bias:
            zeroinit = tf.keras.initializers.zeros()
            self.bias = tf.Variable(initial_value=zeroinit(
                shape=(out_feats), dtype='float32'), trainable=True)
        else:
            self.bias = None

    def call(self, adj, feat, lambda_max=None):
        r"""

        Description
        -----------
        Compute (Dense) Chebyshev Spectral Graph Convolution layer.

        Parameters
        ----------
        adj : tf.Tensor
            The adjacency matrix of the graph to apply Graph Convolution on,
            should be of shape :math:`(N, N)`, where a row represents the destination
            and a column represents the source.
        feat : tf.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : float or None, optional
            A float value indicates the largest eigenvalue of given graph.
            Default: None.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        A = adj
        num_nodes = A.shape[0]
        in_degree = 1 / tf.sqrt(tf.clip_by_value(tf.reduce_sum(A, 1),
                                                 clip_value_min=1,
                                                 clip_value_max=np.inf))
        D_invsqrt = tf.linalg.diag(in_degree)
        I = tf.eye(num_nodes)
        L = I - D_invsqrt @ A @ D_invsqrt

        if lambda_max is None:
            lambda_ = tf.linalg.eig(L)[0][:, 0]
            lambda_max = tf.reduce_max(lambda_)

        L_hat = 2 * L / lambda_max - I
        Z = [tf.eye(num_nodes)]
        for i in range(1, self._k):
            if i == 1:
                Z.append(L_hat)
            else:
                Z.append(2 * L_hat @ Z[-1] - Z[-2])

        Zs = tf.stack(Z, 0)  # (k, n, n)

        Zh = (Zs @ tf.expand_dims(feat, axis=0) @ self.W)
        Zh = tf.reduce_sum(Zh, 0)

        if self.bias is not None:
            Zh = Zh + self.bias
        return Zh
