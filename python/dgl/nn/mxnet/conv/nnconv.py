"""MXNet Module for NNConv layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ....utils import expand_as_pair


class NNConv(nn.Block):
    r"""

    Description
    -----------
    Graph Convolution layer introduced in `Neural Message Passing
    for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`__.

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    where :math:`e_{ij}` is the edge feature, :math:`f_\Theta` is a function
    with learnable parameters.

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
        NN can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    edge_func : callable activation function/layer
        Maps each edge feature to a vector of shape
        ``(in_feats * out_feats)`` as weight to compute
        messages.
        Also is the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``mean`` or ``max``).
    residual : bool, optional
        If True, use residual connection. Default: ``False``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from mxnet import gluon
    >>> from dgl.nn import NNConv
    >>>
    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = mx.nd.ones((6, 10))
    >>> lin = gluon.nn.Dense(20)
    >>> lin.initialize(ctx=mx.cpu(0))
    >>> def edge_func(efeat):
    >>>      return lin(efeat)
    >>> efeat = mx.nd.ones((12, 5))
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> res = conv(g, feat, efeat)
    >>> res
    [[0.39946803 0.32098457]
    [0.39946803 0.32098457]
    [0.39946803 0.32098457]
    [0.39946803 0.32098457]
    [0.39946803 0.32098457]
    [0.39946803 0.32098457]]
    <NDArray 6x2 @cpu(0)>

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = mx.nd.random.randn(2, 10)
    >>> v_feat = mx.nd.random.randn(4, 10)
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> efeat = mx.nd.ones((5, 5))
    >>> res = conv(g, (u_feat, v_feat), efeat)
    >>> res
    [[ 0.24425688  0.3238042 ]
    [-0.11651017 -0.01738572]
    [ 0.06387337  0.15320925]
    [ 0.24425688  0.3238042 ]]
    <NDArray 4x2 @cpu(0)>
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_func,
                 aggregator_type,
                 residual=False,
                 bias=True):
        super(NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type

        with self.name_scope():
            self.edge_nn = edge_func
            if residual:
                if self._in_dst_feats != out_feats:
                    self.res_fc = nn.Dense(
                        out_feats, in_units=self._in_dst_feats,
                        use_bias=False, weight_initializer=mx.init.Xavier())
                else:
                    self.res_fc = Identity()
            else:
                self.res_fc = None

            if bias:
                self.bias = self.params.get('bias',
                                            shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

    def forward(self, graph, feat, efeat):
        r"""Compute MPNN Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray or pair of mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        efeat : mxnet.NDArray
            The edge feature of shape :math:`(N, *)`, should fit the input
            shape requirement of ``edge_nn``.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.expand_dims(-1)
            # (n, d_in, d_out)
            graph.edata['w'] = self.edge_nn(efeat).reshape(-1, self._in_src_feats, self._out_feats)
            # (n, d_in, d_out)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))
            rst = graph.dstdata.pop('neigh').sum(axis=1) # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias.data(feat_dst.context)
            return rst
