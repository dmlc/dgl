"""MXNet modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ....base import DGLError
from ...functional import edge_softmax
from ....utils import expand_as_pair

#pylint: enable=W0235
class GATConv(nn.Block):
    r"""

    Description
    -----------
    Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} &= \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} &= \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        GATConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Defaults: ``False``.

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
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> from dgl.nn import GATConv
    >>>
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = mx.nd.ones((6, 10))
    >>> gatconv = GATConv(10, 2, num_heads=3)
    >>> gatconv.initialize(ctx=mx.cpu(0))
    >>> res = gatconv(g, feat)
    >>> res
    [[[ 0.32368395 -0.10501936]
    [ 1.0839728   0.92690575]
    [-0.54581136 -0.84279203]]
    [[ 0.32368395 -0.10501936]
    [ 1.0839728   0.92690575]
    [-0.54581136 -0.84279203]]
    [[ 0.32368395 -0.10501936]
    [ 1.0839728   0.92690575]
    [-0.54581136 -0.84279203]]
    [[ 0.32368395 -0.10501937]
    [ 1.0839728   0.9269058 ]
    [-0.5458114  -0.8427921 ]]
    [[ 0.32368395 -0.10501936]
    [ 1.0839728   0.92690575]
    [-0.54581136 -0.84279203]]
    [[ 0.32368395 -0.10501936]
    [ 1.0839728   0.92690575]
    [-0.54581136 -0.84279203]]]
    <NDArray 6x3x2 @cpu(0)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = mx.nd.random.randn(2, 5)
    >>> v_feat = mx.nd.random.randn(4, 10)
    >>> gatconv = GATConv((5,10), 2, 3)
    >>> gatconv.initialize(ctx=mx.cpu(0))
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    [[[-1.01624     1.8138596 ]
    [ 1.2322129  -0.8410206 ]
    [-1.9325689   1.3824553 ]]
    [[ 0.9915016  -1.6564168 ]
    [-0.32610354  0.42505783]
    [ 1.5278397  -0.92114615]]
    [[-0.32592064  0.62067866]
    [ 0.6162219  -0.3405491 ]
    [-1.356375    0.9988818 ]]
    [[-1.01624     1.8138596 ]
    [ 1.2322129  -0.8410206 ]
    [-1.9325689   1.3824553 ]]]
    <NDArray 4x3x2 @cpu(0)>
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        with self.name_scope():
            if isinstance(in_feats, tuple):
                self.fc_src = nn.Dense(out_feats * num_heads, use_bias=False,
                                       weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                       in_units=self._in_src_feats)
                self.fc_dst = nn.Dense(out_feats * num_heads, use_bias=False,
                                       weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                       in_units=self._in_dst_feats)
            else:
                self.fc = nn.Dense(out_feats * num_heads, use_bias=False,
                                   weight_initializer=mx.init.Xavier(magnitude=math.sqrt(2.0)),
                                   in_units=in_feats)
            self.attn_l = self.params.get('attn_l',
                                          shape=(1, num_heads, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            self.attn_r = self.params.get('attn_r',
                                          shape=(1, num_heads, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            self.feat_drop = nn.Dropout(feat_drop)
            self.attn_drop = nn.Dropout(attn_drop)
            self.leaky_relu = nn.LeakyReLU(negative_slope)
            if residual:
                if in_feats != out_feats:
                    self.res_fc = nn.Dense(out_feats * num_heads, use_bias=False,
                                           weight_initializer=mx.init.Xavier(
                                               magnitude=math.sqrt(2.0)),
                                           in_units=in_feats)
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = None
            self.activation = activation

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

    def forward(self, graph, feat, get_attention=False):
        r"""

        Description
        -----------
        Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray or pair of mxnet.NDArray
            If a mxnet.NDArray is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of mxnet.NDArray is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        get_attention : bool, optional
            Whether to return the attention values. Default to False.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        mxnet.NDArray, optional
            The attention values of shape :math:`(E, H, 1)`, where :math:`E` is the number of
            edges. This is returned only when :attr:`get_attention` is ``True``.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if graph.in_degrees().min() == 0:
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).reshape(
                    -1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).reshape(
                    -1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).reshape(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l.data(feat_src.context)).sum(axis=-1).expand_dims(-1)
            er = (feat_dst * self.attn_r.data(feat_src.context)).sum(axis=-1).expand_dims(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).reshape(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst
