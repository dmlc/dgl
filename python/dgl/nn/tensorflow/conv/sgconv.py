"""tf Module for Simplifying Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, W0613
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from .... import function as fn
from ....base import DGLError


class SGConv(layers.Layer):
    r"""

    Description
    -----------
    Simplifying Graph Convolution layer from paper `Simplifying Graph
    Convolutional Networks <https://arxiv.org/pdf/1902.07153.pdf>`__.

    .. math::
        H^{K} = (\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2})^K X \Theta

    where :math:`\tilde{A}` is :math:`A` + :math:`I`.
    Thus the graph input is expected to have self-loop edges added.

    Parameters
    ----------
    in_feats : int
        Number of input features; i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Number of output features; i.e, the number of dimensions of :math:`H^{K}`.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
    cached : bool
        If True, the module would cache

        .. math::
            (\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})^K X\Theta

        at the first forward call. This parameter should only be set to
        ``True`` in Transductive Learning setting.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.  Default: ``False``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dgl.nn import SGConv
    >>>
    >>> with tf.device("CPU:0"):
    >>>     g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>>     g = dgl.add_self_loop(g)
    >>>     feat = tf.ones((6, 10))
    >>>     conv = SGConv(10, 2, k=2, cached=True)
    >>>     res = conv(g, feat)
    >>>     res
    <tf.Tensor: shape=(6, 2), dtype=float32, numpy=
    array([[0.61023676, 0.5246612 ],
        [0.61023676, 0.5246612 ],
        [0.61023676, 0.5246612 ],
        [0.8697353 , 0.7477695 ],
        [0.60570633, 0.520766  ],
        [0.6102368 , 0.52466124]], dtype=float32)>
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True,
                 norm=None,
                 allow_zero_in_degree=False):
        super(SGConv, self).__init__()
        self.fc = layers.Dense(out_feats, use_bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def call(self, graph, feat):
        r"""

        Description
        -----------
        Compute Simplifying Graph Convolution layer.

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

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        If ``cache`` is set to True, ``feat`` and ``graph`` should not change during
        training, or you will get wrong results.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if  tf.math.count_nonzero(graph.in_degrees() == 0) > 0:
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

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
