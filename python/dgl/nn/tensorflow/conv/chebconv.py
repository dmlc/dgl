"""Tensorflow Module for Chebyshev Spectral Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from ....base import dgl_warning
from .... import laplacian_lambda_max, broadcast_nodes, function as fn


class ChebConv(layers.Layer):
    r"""

    Description
    -----------
    Chebyshev Spectral Graph Convolution layer from paper `Convolutional
    Neural Networks on Graphs with Fast Localized Spectral Filtering
    <https://arxiv.org/pdf/1606.09375.pdf>`__.

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
    >>> import tensorflow as tf
    >>> from dgl.nn import ChebConv
    >>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = tf.ones(6, 10)
    >>> conv = ChebConv(10, 2, 2)
    >>> res = conv(g, feat)
    >>> res
    tensor([[ 0.6163, -0.1809],
            [ 0.6163, -0.1809],
            [ 0.6163, -0.1809],
            [ 0.9698, -1.5053],
            [ 0.3664,  0.7556],
            [-0.2370,  3.0164]])
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 activation=tf.nn.relu,
                 bias=True):
        super(ChebConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = layers.Dense(out_feats, use_bias=bias)

    def call(self, graph, feat, lambda_max=None):
        r"""

        Description
        -----------
        Compute ChebNet layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        lambda_max : list or tensor or None, optional.
            A list(tensor) with length :math:`B`, stores the largest eigenvalue
            of the normalized laplacian of each individual graph in ``graph``,
            where :math:`B` is the batch size of the input graph. Default: None.
            If None, this method would compute the list by calling
            ``dgl.laplacian_lambda_max``.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation Feat * D^-1/2 A D^-1/2 """
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            return graph.ndata.pop('h') * D_invsqrt

        with graph.local_scope():
            in_degrees = tf.clip_by_value(tf.cast(graph.in_degrees(), tf.float32),
                                          clip_value_min=1,
                                          clip_value_max=np.inf)
            D_invsqrt = tf.expand_dims(tf.pow(in_degrees, -0.5), axis=-1)
            if lambda_max is None:
                try:
                    lambda_max = laplacian_lambda_max(graph)
                except BaseException:
                    # if the largest eigenvalue is not found
                    dgl_warning(
                        "Largest eigonvalue not found, using default value 2 for lambda_max",
                        RuntimeWarning)
                    lambda_max = tf.constant(2, dtype=tf.float32)

            if isinstance(lambda_max, list):
                lambda_max = tf.constant(lambda_max, dtype=tf.float32)
            if lambda_max.ndim == 1:
                lambda_max = tf.expand_dims(
                    lambda_max, axis=-1)  # (B,) to (B, 1)

            # broadcast from (B, 1) to (N, 1)
            lambda_max = broadcast_nodes(graph, lambda_max)
            re_norm = 2. / lambda_max

            # X_0 is the raw feature, Xt refers to the concatenation of X_0, X_1, ... X_t
            Xt = X_0 = feat

            # X_1(f)
            if self._k > 1:
                h = unnLaplacian(X_0, D_invsqrt, graph)
                X_1 = - re_norm * h + X_0 * (re_norm - 1)
                # Concatenate Xt and X_1
                Xt = tf.concat((Xt, X_1), 1)

            # Xi(x), i = 2...k
            for _ in range(2, self._k):
                h = unnLaplacian(X_1, D_invsqrt, graph)
                X_i = - 2 * re_norm * h + X_1 * 2 * (re_norm - 1) - X_0
                # Concatenate Xt and X_i
                Xt = tf.concat((Xt, X_i), 1)
                X_1, X_0 = X_i, X_1

            # linear projection
            h = self.linear(Xt)

            # activation
            if self.activation:
                h = self.activation(h)

        return h
