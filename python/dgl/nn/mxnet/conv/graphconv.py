"""MXNet modules for graph convolutions(GCN)"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import mxnet as mx
from mxnet import gluon

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair


class GraphConv(gluon.Block):
    r"""Graph convolutional layer from `Semi-Supervised Classification with Graph Convolutional
    Networks <https://arxiv.org/abs/1609.02907>`__

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
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> import numpy as np
    >>> from dgl.nn import GraphConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, feat)
    >>> print(res)
    [[1.0209361  0.22472616]
    [1.1240715  0.24742813]
    [1.0209361  0.22472616]
    [1.2924911  0.28450024]
    [1.3568745  0.29867214]
    [0.7948386  0.17495811]]
    <NDArray 6x2 @cpu(0)>

    >>> # allow_zero_in_degree example
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> conv = GraphConv(10, 2, norm='both', weight=True, bias=True, allow_zero_in_degree=True)
    >>> res = conv(g, feat)
    >>> print(res)
    [[1.0209361  0.22472616]
    [1.1240715  0.24742813]
    [1.0209361  0.22472616]
    [1.2924911  0.28450024]
    [1.3568745  0.29867214]
    [0.  0.]]
    <NDArray 6x2 @cpu(0)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.heterograph({('_N', '_E', '_N'):(u, v)})
    >>> u_fea = mx.nd.random.randn(2, 5)
    >>> v_fea = mx.nd.random.randn(4, 5)
    >>> conv = GraphConv(5, 2, norm='both', weight=True, bias=True)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, (u_fea, v_fea))
    >>> res
    [[ 0.26967263  0.308129  ]
    [ 0.05143356 -0.11355402]
    [ 0.22705637  0.1375853 ]
    [ 0.26967263  0.308129  ]]
    <NDArray 4x2 @cpu(0)>
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

        with self.name_scope():
            if weight:
                self.weight = self.params.get(
                    "weight",
                    shape=(in_feats, out_feats),
                    init=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                )
            else:
                self.weight = None

            if bias:
                self.bias = self.params.get(
                    "bias", shape=(out_feats,), init=mx.init.Zero()
                )
            else:
                self.bias = None

        self._activation = activation

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

    def forward(self, graph, feat, weight=None):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray or pair of mxnet.NDArray
            If a single tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of tensors are given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

            Note that in the special case of graph convolutional networks, if a pair of
            tensors is given, the latter element will not participate in computation.
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        mxnet.NDArray
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
                if graph.in_degrees().min() == 0:
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
                degs = (
                    graph.out_degrees()
                    .as_in_context(feat_dst.context)
                    .astype("float32")
                )
                degs = mx.nd.clip(degs, a_min=1, a_max=float("inf"))
                if self._norm == "both":
                    norm = mx.nd.power(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_src.ndim - 1)
                norm = norm.reshape(shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError(
                        "External weight is provided while at the same time the"
                        " module has defined its own weight parameter. Please"
                        " create the module with flag weight=False."
                    )
            else:
                weight = self.weight.data(feat_src.context)

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = mx.nd.dot(feat_src, weight)
                graph.srcdata["h"] = feat_src
                graph.update_all(
                    fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
                )
                rst = graph.dstdata.pop("h")
            else:
                # aggregate first then mult W
                graph.srcdata["h"] = feat_src
                graph.update_all(
                    fn.copy_u(u="h", out="m"), fn.sum(msg="m", out="h")
                )
                rst = graph.dstdata.pop("h")
                if weight is not None:
                    rst = mx.nd.dot(rst, weight)

            if self._norm in ["both", "right"]:
                degs = (
                    graph.in_degrees()
                    .as_in_context(feat_dst.context)
                    .astype("float32")
                )
                degs = mx.nd.clip(degs, a_min=1, a_max=float("inf"))
                if self._norm == "both":
                    norm = mx.nd.power(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.ndim - 1)
                norm = norm.reshape(shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias.data(rst.context)

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def __repr__(self):
        summary = "GraphConv("
        summary += "in={:d}, out={:d}, normalization={}, activation={}".format(
            self._in_feats, self._out_feats, self._norm, self._activation
        )
        summary += ")"
        return summary
