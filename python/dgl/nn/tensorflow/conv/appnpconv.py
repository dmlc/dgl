"""TF Module for APPNPConv"""
# pylint: disable= no-member, arguments-differ, invalid-name
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn


class APPNPConv(layers.Layer):
    r"""Approximate Personalized Propagation of Neural Predictions
    layer from `Predict then Propagate: Graph Neural Networks
    meet Personalized PageRank <https://arxiv.org/pdf/1810.05997.pdf>`__

    .. math::
        H^{0} & = X

        H^{t+1} & = (1-\alpha)\left(\hat{D}^{-1/2}
        \hat{A} \hat{D}^{-1/2} H^{t}\right) + \alpha H^{0}

    Parameters
    ----------
    k : int
        Number of iterations :math:`K`.
    alpha : float
        The teleport probability :math:`\alpha`.
    edge_drop : float, optional
        Dropout rate on edges that controls the
        messages received by each node. Default: ``0``.
    """

    def __init__(self, k, alpha, edge_drop=0.0):
        super(APPNPConv, self).__init__()
        self._k = k
        self._alpha = alpha
        self.edge_drop = layers.Dropout(edge_drop)

    def call(self, graph, feat):
        r"""Compute APPNP layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor
            The input feature of shape :math:`(N, *)` :math:`N` is the
            number of nodes, and :math:`*` could be of any shape.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, *)` where :math:`*`
            should be the same as input shape.
        """
        with graph.local_scope():
            degs = tf.clip_by_value(
                tf.cast(graph.in_degrees(), tf.float32),
                clip_value_min=1,
                clip_value_max=np.inf,
            )
            norm = tf.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = tf.reshape(norm, shp)
            feat_0 = feat
            for _ in range(self._k):
                # normalization by src node
                feat = feat * norm
                graph.ndata["h"] = feat
                graph.edata["w"] = self.edge_drop(tf.ones(graph.num_edges(), 1))
                graph.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
                feat = graph.ndata.pop("h")
                # normalization by dst node
                feat = feat * norm
                feat = (1 - self._alpha) * feat + self._alpha * feat_0
            return feat
