"""tf Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, W0613
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from .... import function as fn


class SGConv(layers.Layer):
    r"""Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    .. math::
        H^{l+1} = (\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2})^K H^{l} \Theta^{l}

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
    cached : bool
        If True, the module would cache

        .. math::
            (\hat{D}^{-\frac{1}{2}}\hat{A}\hat{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None):
        super(SGConv, self).__init__()
        self.fc = layers.Dense(out_feats, use_bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm

    def call(self, graph, feat):
        r"""Compute Simplifying Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.

        Notes
        -----
        If ``cache`` is se to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = tf.clip_by_value(tf.cast(
                graph.in_degrees(), tf.float32), clip_value_min=1, clip_value_max=np.inf)
            norm = tf.pow(degs, -0.5)
            norm = tf.expand_dims(norm, 1)
            # compute (D^-1 A^k D)^k X
            for _ in range(self._k):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm

            if self.norm is not None:
                feat = self.norm(feat)

            # cache feature
            if self._cached:
                self._cached_h = feat
        return self.fc(feat)
