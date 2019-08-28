"""MXNet modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
import numpy as np

from . import utils
from ... import function as fn

__all__ = ['GraphConv', 'TAGConv', 'RelGraphConv']

class GraphConv(gluon.Block):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : mxnet.gluon.parameter.Parameter
        The learnable weight tensor.
    bias : mxnet.gluon.parameter.Parameter
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(GraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_feats, out_feats),
                                          init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if bias:
                self.bias = self.params.get('bias', shape=(out_feats,),
                                            init=mx.init.Zero())
            else:
                self.bias = None

        self._activation = activation

    def forward(self, graph, feat):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature

        Returns
        -------
        mxnet.NDArray
            The output feature
        """
        graph = graph.local_var()
        if self._norm:
            degs = graph.in_degrees().astype('float32')
            norm = mx.nd.power(mx.nd.clip(degs, a_min=1, a_max=float("inf")), -0.5)
            shp = norm.shape + (1,) * (feat.ndim - 1)
            norm = norm.reshape(shp).as_in_context(feat.context)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = mx.nd.dot(feat, self.weight.data(feat.context))
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata.pop('h')
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata.pop('h')
            rst = mx.nd.dot(rst, self.weight.data(feat.context))

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias.data(rst.context)

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def __repr__(self):
        summary = 'GraphConv('
        summary += 'in={:d}, out={:d}, normalization={}, activation={}'.format(
            self._in_feats, self._out_feats,
            self._norm, self._activation)
        summary += '\n)'
        return summary

class TAGConv(gluon.Block):
    r"""Apply Topology Adaptive Graph Convolutional Network

    .. math::
        \mathbf{X}^{\prime} = \sum_{k=0}^K \mathbf{D}^{-1/2} \mathbf{A}
        \mathbf{D}^{-1/2}\mathbf{X} \mathbf{\Theta}_{k},

    where :math:`\mathbf{A}` denotes the adjacency matrix and
    :math:`D_{ii} = \sum_{j=0} A_{ij}` its diagonal degree matrix.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k: int, optional
        Number of hops :math: `k`. (default: 2)
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    lin : mxnet.gluon.parameter.Parameter
        The learnable weight tensor.
    bias : mxnet.gluon.parameter.Parameter
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None):
        super(TAGConv, self).__init__()
        self.out_feats = out_feats
        self.k = k
        self.bias = bias
        self.activation = activation
        self.in_feats = in_feats

        self.lin = self.params.get(
            'weight', shape=(self.in_feats * (self.k + 1), self.out_feats),
            init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
        if self.bias:
            self.h_bias = self.params.get('bias', shape=(out_feats,),
                                          init=mx.init.Zero())

    def forward(self, graph, feat):
        r"""Compute graph convolution

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        graph = graph.local_var()

        degs = graph.in_degrees().astype('float32')
        norm = mx.nd.power(mx.nd.clip(degs, a_min=1, a_max=float("inf")), -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = norm.reshape(shp).as_in_context(feat.context)

        rst = feat
        for _ in range(self.k):
            rst = rst * norm
            graph.ndata['h'] = rst

            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = rst * norm
            feat = mx.nd.concat(feat, rst, dim=-1)

        rst = mx.nd.dot(feat, self.lin.data(feat.context))
        if self.bias is not None:
            rst = rst + self.h_bias.data(rst.context)

        if self.activation is not None:
            rst = self.activation(rst)

        return rst

class RelGraphConv(gluon.Block):
    r"""Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

      h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
      \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

      W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            self.weight = self.params.get(
                'weight', shape=(self.num_bases, self.in_feat, self.out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = self.params.get(
                    'w_comp', shape=(self.num_rels, self.num_bases),
                    init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = self.params.get(
                'weight',
                shape=(self.num_rels, self.num_bases * self.submat_in * self.submat_out),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = self.params.get('bias', shape=(out_feat,),
                                          init=mx.init.Zero())

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.params.get(
                'W_0', shape=(in_feat, out_feat),
                init=mx.init.Xavier(magnitude=math.sqrt(2.0)))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        ctx = edges.src['h'].context
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.data(ctx).reshape(
                self.num_bases, self.in_feat * self.out_feat)
            weight = nd.dot(self.w_comp.data(ctx), weight).reshape(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight.data(ctx)

        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        ctx = edges.src['h'].context
        if edges.src['h'].dtype in (np.int32, np.int64) and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        weight = self.weight.data(ctx)[edges.data['type'], :].reshape(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].reshape(-1, 1, self.submat_in)
        msg = nd.batch_dot(node, weight).reshape(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        """Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : mx.ndarray.NDArray
            Input node features. Could be either
              - (|V|, D) dense tensor
              - (|V|,) int64 vector, representing the categorical values of each
                node. We then treat the input feature as an one-hot encoding feature.
        etypes : mx.ndarray.NDArray
            Edge type tensor. Shape: (|E|,)
        norm : mx.ndarray.NDArray
            Optional edge normalizer tensor. Shape: (|E|, 1)

        Returns
        -------
        mx.ndarray.NDArray
            New node features.
        """
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight.data(x.context))

        # message passing
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'))

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias.data(x.context)
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)

        return node_repr
