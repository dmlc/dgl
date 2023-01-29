"""Tensorflow Module for Graph Isomorphism Network layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from ....utils import expand_as_pair


class GINConv(layers.Layer):
    r"""Graph Isomorphism Network layer from `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__

    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)

    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dgl.nn import GINConv
    >>>
    >>> with tf.device("CPU:0"):
    >>>     g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>>     feat = tf.ones((6, 10))
    >>>     lin = tf.keras.layers.Dense(10)
    >>>     conv = GINConv(lin, 'max')
    >>>     res = conv(g, feat)
    >>>     res
    <tf.Tensor: shape=(6, 10), dtype=float32, numpy=
    array([[-0.1090256 ,  1.9050574 , -0.30704725, -1.995831  , -0.36399186,
            1.10414   ,  2.4885745 , -0.35387516,  1.3568261 ,  1.7267858 ],
        [-0.1090256 ,  1.9050574 , -0.30704725, -1.995831  , -0.36399186,
            1.10414   ,  2.4885745 , -0.35387516,  1.3568261 ,  1.7267858 ],
        [-0.1090256 ,  1.9050574 , -0.30704725, -1.995831  , -0.36399186,
            1.10414   ,  2.4885745 , -0.35387516,  1.3568261 ,  1.7267858 ],
        [-0.1090256 ,  1.9050574 , -0.30704725, -1.995831  , -0.36399186,
            1.10414   ,  2.4885745 , -0.35387516,  1.3568261 ,  1.7267858 ],
        [-0.1090256 ,  1.9050574 , -0.30704725, -1.995831  , -0.36399186,
            1.10414   ,  2.4885745 , -0.35387516,  1.3568261 ,  1.7267858 ],
        [-0.0545128 ,  0.9525287 , -0.15352362, -0.9979155 , -0.18199593,
            0.55207   ,  1.2442873 , -0.17693758,  0.67841303,  0.8633929 ]],
        dtype=float32)>
    """

    def __init__(
        self, apply_func, aggregator_type, init_eps=0, learn_eps=False
    ):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if aggregator_type == "sum":
            self._reducer = fn.sum
        elif aggregator_type == "max":
            self._reducer = fn.max
        elif aggregator_type == "mean":
            self._reducer = fn.mean
        else:
            raise KeyError(
                "Aggregator type {} not recognized.".format(aggregator_type)
            )
        # to specify whether eps is trainable or not.
        self.eps = tf.Variable(
            initial_value=[init_eps], dtype=tf.float32, trainable=learn_eps
        )

    def call(self, graph, feat):
        r"""Compute Graph Isomorphism Network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : tf.Tensor or pair of tf.Tensor
            If a tf.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tf.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.

        Returns
        -------
        tf.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata["h"] = feat_src
            graph.update_all(fn.copy_u("h", "m"), self._reducer("m", "neigh"))
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst
