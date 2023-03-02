"""Tensorflow modules for graph convolutions(GCN)."""
import numpy as np

# pylint: disable= no-member, arguments-differ, invalid-name
import tensorflow as tf
from tensorflow.keras import layers

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair

# pylint: disable=W0235


class GraphConv(layers.Layer):
    r"""Graph convolution from `Semi-Supervised Classification with Graph Convolutional Networks
    <https://arxiv.org/abs/1609.02907>`__

    Mathematically it is defined as follows:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the set of neighbors of node :math:`i`,
    :math:`c_{ij}` is the product of the square root of node degrees
    (i.e.,  :math:`c_{ij} = \sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`),
    and :math:`\sigma` is an activation function.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    norm : str, optional
        How to apply the normalizer.  Can be one of the following values:

        * ``right``, to divide the aggregated messages by each node's in-degrees,
          which is equivalent to averaging the received messages.

        * ``none``, where no normalization is applied.

        * ``both`` (default), where the messages are scaled with :math:`1/c_{ji}` above, equivalent
          to symmetric normalization.

        * ``left``, to divide the messages sent out from each node by its out-degrees,
          equivalent to random walk normalization.
    weight : bool, optional
        If True, apply a linear layer. Otherwise, aggregating the messages
        without a weight matrix.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.

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

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import tensorflow as tf
    >>> from dgl.nn import GraphConv

    >>> # Case 1: Homogeneous graph
    >>> with tf.device("CPU:0"):
    ...     g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    ...     g = dgl.add_self_loop(g)
    ...     feat = tf.ones((6, 10))
    ...     conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
    ...     res = conv(g, feat)
    >>> print(res)
    <tf.Tensor: shape=(6, 2), dtype=float32, numpy=
    array([[ 0.6208475 , -0.4896223 ],
        [ 0.68356586, -0.5390842 ],
        [ 0.6208475 , -0.4896223 ],
        [ 0.7859846 , -0.61985517],
        [ 0.8251371 , -0.65073216],
        [ 0.48335412, -0.38119012]], dtype=float32)>
    >>> # allow_zero_in_degree example
    >>> with tf.device("CPU:0"):
    ...     g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    ...     conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
    ...     res = conv(g, feat)
    >>> print(res)
        <tf.Tensor: shape=(6, 2), dtype=float32, numpy=
        array([[ 0.6208475 , -0.4896223 ],
            [ 0.68356586, -0.5390842 ],
            [ 0.6208475 , -0.4896223 ],
            [ 0.7859846 , -0.61985517],
            [ 0.8251371 , -0.65073216],
            [ 0., 0.]], dtype=float32)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> with tf.device("CPU:0"):
    ...     g = dgl.heterograph({('_N', '_E', '_N'):(u, v)})
    ...     u_fea = tf.convert_to_tensor(np.random.rand(2, 5))
    ...     v_fea = tf.convert_to_tensor(np.random.rand(4, 5))
    ...     conv = GraphConv(5, 2, norm='both', weight=True, bias=True)
    ...     res = conv(g, (u_fea, v_fea))
    >>> res
    <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
    array([[ 1.3607183, -0.1636453],
        [ 1.6665325, -0.2004239],
        [ 2.1405895, -0.2574358],
        [ 1.3607183, -0.1636453]], dtype=float32)>
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        norm="both",
        weight=True,
        bias=True,
        activation=None,
        allow_zero_in_degree=False,
    ):
        super(GraphConv, self).__init__()
        if norm not in ("none", "both", "right", "left"):
            raise DGLError(
                'Invalid norm value. Must be either "none", "both", "right" or "left".'
                ' But got "{}".'.format(norm)
            )
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self._allow_zero_in_degree = allow_zero_in_degree

        if weight:
            xinit = tf.keras.initializers.glorot_uniform()
            self.weight = tf.Variable(
                initial_value=xinit(
                    shape=(in_feats, out_feats), dtype="float32"
                ),
                trainable=True,
            )
        else:
            self.weight = None

        if bias:
            zeroinit = tf.keras.initializers.zeros()
            self.bias = tf.Variable(
                initial_value=zeroinit(shape=(out_feats), dtype="float32"),
                trainable=True,
            )
        else:
            self.bias = None

        self._activation = activation

    def set_allow_zero_in_degree(self, set_value):
        r"""Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def call(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, which is the case for bipartite graph, the pair
            must contain two tensors of shape :math:`(N_{in}, D_{in_{src}})` and
            :math:`(N_{out}, D_{in_{dst}})`.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if tf.math.count_nonzero(graph.in_degrees() == 0) > 0:
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm in ["both", "left"]:
                degs = tf.clip_by_value(
                    tf.cast(graph.out_degrees(), tf.float32),
                    clip_value_min=1,
                    clip_value_max=np.inf,
                )
                if self._norm == "both":
                    norm = tf.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.ndim - 1)
                norm = tf.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = tf.matmul(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(
                    fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
                )
                rst = graph.dstdata["h"]
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(
                    fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
                )
                rst = graph.dstdata["h"]
                if weight is not None:
                    rst = tf.matmul(rst, weight)

            if self._norm in ["both", "right"]:
                degs = tf.clip_by_value(
                    tf.cast(graph.in_degrees(), tf.float32),
                    clip_value_min=1,
                    clip_value_max=np.inf,
                )
                if self._norm == "both":
                    norm = tf.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.ndim - 1)
                norm = tf.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}"
        summary += ", normalization={_norm}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"
        return summary.format(**self.__dict__)
