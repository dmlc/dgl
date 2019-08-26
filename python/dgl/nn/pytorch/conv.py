"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from . import utils
from ... import function as fn
from ...batched_graph import broadcast_nodes
from ...transform import laplacian_lambda_max
from .softmax import edge_softmax

__all__ = ['GraphConv', 'GATConv', 'TAGConv', 'RelGraphConv', 'SAGEConv',
           'SGConv', 'APPNPConv', 'GINConv', 'GatedGraphConv', 'GMMConv',
           'AGNNConv', 'NNConv', 'DenseGCNConv', 'DenseSAGEConv']

class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.

    Args:
        args: any argument (unused)
        kwargs: any keyword argument (unused)

    Examples::

        >>> m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 20])
    """
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class GraphConv(nn.Module):
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
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
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

        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, feat, graph):
        r"""Compute graph convolution.

        Notes
        -----
            * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
              dimensions, :math:`N` is the number of nodes.
            * Output shaApply graph convolution over an input signal.

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
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    pe: :math:`(N, *, \text{out_feats})` where all but the last dimension are
              the same shape as the input.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()
        if self._norm:
            norm = th.pow(graph.in_degrees().float(), -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
        else:
            # aggregate first then mult W
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = th.matmul(rst, self.weight)

        if self._norm:
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
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)


class GATConv(nn.Module):
    r"""Apply graph attention over an input signal.

    Parameters
    ----------
    in_feats : int
    out_feats : int
    num_heads : int
    feat_drop : float
    attn_drop : float
    alpha : float
    residual : bool, optional
    activation : callable activation function/layer or None, optional.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop,
                 attn_drop,
                 alpha,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop else Identity()
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else Identity()
        self.leaky_relu = nn.LeakyReLU(alpha)
        self._residual = residual
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        self._reset_parameters()

        self.activation = activation

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if self._residual and self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, feat, graph):
        r"""Compute graph attention

        TODO(zihao): docstring
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})

        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))
        rst = graph.ndata['ft']

        # residual
        if self._residual:
            resval = self.res_fc(h).view(feat.shape[0], -1, self._out_feats)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst

    def extra_repr(self):
        pass


class TAGConv(nn.Module):
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
        Number of hops :math: `k`. (default: 3)
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    lin : torch.Module
        The learnable linear module.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=2,
                 bias=True,
                 activation=None):
        super(TAGConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._k = k
        self._activation = activation
        self.lin = nn.Linear(in_feats * (self._k + 1), out_feats, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def forward(self, feat, graph):
        r"""Compute graph convolution

        Parameters
        ----------
        feat : torch.Tensor
            The input feature
        graph : DGLGraph
            The graph.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        norm = th.pow(graph.in_degrees().float(), -0.5)
        shp = norm.shape + (1,) * (feat.dim() - 1)
        norm = th.reshape(norm, shp).to(feat.device)

        #D-1/2 A D -1/2 X
        fstack = [feat]
        for _ in range(self._k):

            rst = fstack[-1] * norm
            graph.ndata['h'] = rst

            graph.update_all(fn.copy_src(src='h', out='m'),
                             fn.sum(msg='m', out='h'))
            rst = graph.ndata['h']
            rst = rst * norm
            fstack.append(rst)

        rst = self.lin(th.cat(fstack, dim=-1))

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

class RelGraphConv(nn.Module):
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
            self.weight = nn.Parameter(th.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(th.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(th.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = th.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight

        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == th.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = th.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        """Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
              - (|V|, D) dense tensor
              - (|V|,) int64 vector, representing the categorical values of each
                node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: (|E|,)
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: (|E|, 1)

        Returns
        -------
        torch.Tensor
            New node features.
        """
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight)
        # message passing
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr


class SAGEConv(nn.Module):
    r""" GraphSAGE layer from paper "".

    Parameters
    ----------
    in_feats : int
    out_feats : int
    feat_drop : float
    aggregator_type : str
    bias : bool
    norm : callable activation function/layer or None, optional
    activation : callable activation function/layer or None, optional
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        # note(zihao): lstm reducer with default schedule (degree bucketing)
        # is slow, we could accelerate this with degree padding in the future.
        input = nodes.mailbox['m'] # (B, L, D)
        batch_size = input.shape[0]
        h = (input.new_zeros((1, batch_size, self._in_feats)),
             input.new_zeros((1, batch_size, self._in_feats)))
        _, (rst, _) = self.lstm(input, h)
        return {'neigh': rst.squeeze(0)}

    def forward(self, feat, graph):
        r"""Compute the output of a GraphSAGE layer.

        TODO(zihao): docstring
        """
        graph = graph.local_var()
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().float()
            degs = degs.to(feat.device)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = F.relu(self.fc_pool(feat))
            graph.update_all(fn.copy_src('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.ndata['neigh']
        else: # lstm:
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'), self._lstm_reducer)
            h_neigh = graph.ndata['neigh']
        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)
        return rst

    def extra_repr(self):
        pass


class GatedGraphConv(nn.Module):
    """Gated Graph Convolution layer from paper ""

    Parameters
    ----------
    in_feats : int
    out_feats : int
    n_steps : int
    n_etyps : int
    aggregator_type : str
    bias : bool
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_etypes,
                 aggregator_type,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._aggre_type = aggregator_type
        self.weight = nn.Embedding(n_etypes, out_feats * out_feats)
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.gru.reset_parameters()
        # TODO(zihao): initialize weight

    def forward(self, feat, etypes, graph):
        graph = graph.local_var()
        zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
        feat = th.cat([feat, zero_pad], -1)
        # NOTE(zihao): there is still room to optimize, we may do kernel fusion
        # for such operations in the future.
        graph.edata['w'] = self.weight(etypes).view(-1, self._out_feats, self._out_feats) # (E, D, D)
        for i in range(self._n_steps):
            graph.ndata['h'] = feat.unsqueeze(-1) # (N, D, 1)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'),
                             fn.sum('m', 'a'))
            a = graph.ndata.pop('a').sum(dim=1) # (N, D)
            feat = self.gru(a, feat)
        return feat


class GMMConv(nn.Module):
    """The Gaussian Mixture Model Convolution layer from "Geometric Deep
     Learning on Graphs and Manifolds using Mixture Model CNNs”

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    dim : int
        Dimension of pseudo-coordinte.
    n_kernels : int
        Number of kernels :math:`K`.
    aggregator_type : str
        Aggregator type (``sum``, ``mean``, ``max``).
    residual : bool
        If True, use residual connection inside this layer.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 dim,
                 n_kernels,
                 aggregator_type,
                 residual=True,
                 bias=True):
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._dim = dim
        self._n_kernels = n_kernels
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'mean':
            self._reducer == fn.mean
        elif aggregator_type == 'max':
            self._reducer == fn.max
        else:
            raise KeyError("Aggregator type {} not recognized.".format(aggregator_type))
        self.mu = nn.Parameter(th.Tensor(n_kernels, dim))
        self.inv_sigma = nn.Parameter(th.Tensor(n_kernels, dim))
        self.fc = nn.Linear(in_feats, n_kernels * out_feats, bias=False)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)

    def reset_parameters(self):
        # TODO(zihao) pay attention to the initialization of mu and sigma
        pass

    def forward(self, feat, pseudo, graph):
        graph = graph.local_var()
        graph.ndata['h'] = self.fc(feat).view(-1, self._n_kernels, self._out_feats)
        E = graph.number_of_edges()
        # compute gaussian weight
        gaussian = -0.5 * ((pseudo.view(E, 1, self._dim) - self.mu.view(1, self._n_kernels, self._dim)) ** 2)
        gaussian = gaussian * (self.inv_sigma.view(1, self._n_kernels, self._dim) ** 2)
        gaussian = th.exp(gaussian.sum(dim=-1, keepdims=True)) # (E, K, 1)
        graph.edata['w'] = gaussian
        graph.update_all(fn.u_mul_e('h', 'w', 'm'), self._reducer('m', 'h'))
        rst = graph.ndata['h'].sum(1)
        # residual connection
        if self.res_fc is not None:
            rst = rst + self.res_fc(feat)
        # bias
        if self.bias is not None:
            rst = rst + self.bias
        return rst


class GINConv(nn.Module):
    """Graph Isomorphism Network layer from paper "How Powerful are Graph Neural Networks?"

    Parameters
    ----------
    nn : torch.nn.Module
    aggregator_type : str
    init_eps : float
    learn_eps : bool
    """
    def __init__(self,
                 nn,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.nn = nn
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = th.nn.Parameter(th.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', th.FloatTensor([init_eps]))

    def forward(self, feat, graph):
        graph = graph.local_var()
        graph.ndata['h'] = feat
        graph.update_all(fn.copy_u('h', 'm'), self._reducer('m', 'neigh'))
        rst = self.nn((1 + self.eps) * feat + graph.ndata['neigh'])
        return rst


class ChebConv(nn.Module):
    """Chebyshev Spectral Ggraph Convolution layer.

    Parameters
    ----------
    in_feats: int
        Number of input features.
    out_feats: int
        Number of output features.
    k : int
        Chebyshev filter size.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k,
                 bias=True):
        super(ChebConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.ModuleList([
            nn.Linear(in_feats, out_feats, bias=False) for _ in range(k)
        ])
        self._k = k
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, feat, graph, lambda_max=None):
        """
        graph : DGLGraph or BatchedDGLGraph
        """
        with graph.local_scope():
            norm = th.pow(graph.in_degrees().float(), -0.5).unsqueeze(-1).to(feat.device)
            if lambda_max is None:
                lambda_max = laplacian_lambda_max(graph)
            if isinstance(lambda_max, list):
                lambda_max = th.Tensor(lambda_max).to(feat.device)
            if lambda_max.dim() < 1:
                lambda_max = lambda_max.unsqueeze(-1) # (B,) to (B, 1)
            # 2 / lambda_max, and broadcast from (B, 1) to (N, 1)
            laplacian_norm = 2. / broadcast_nodes(graph, lambda_max)
            # T0(X)
            Tx_0 = feat
            rst = self.fc[0](Tx_0)
            # T1(X)
            if self._k > 1:
                graph.ndata['h'] = Tx_0 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                #Λ = 2 * L / lambda_max - I
                Tx_1 = (graph.ndata.pop('h') * norm) * laplacian_norm - Tx_0
                rst = rst + self.fc[1](Tx_1)
            # Ti(x), i = 2...k
            for i in range(2, self._k):
                graph.ndata['h'] = Tx_1 * norm
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
                # Λ = 2 * L / lambda_max - I, Tx_k = 2 * Λ * Tx_(k-1) - Tx_(k-2)
                Tx_2 = 2 * ((graph.ndata.pop('h') * norm) * laplacian_norm - Tx_1) - Tx_0
                rst = rst + self.fc[i](Tx_2)
                Tx_1, Tx_0 = Tx_2, Tx_1
            # add bias
            if self.bias:
                rst = rst + self.bias
            return rst


class SGConv(nn.Module):
    """Simplifying Grpah Convolution layer.

    Parameters
    ----------
    in_feats : int
        Number of input features.
    out_feats : int
        Number of output features.
    k : int
        Number of hops :math:`K`. Defaults:``1``.
    cached : bool
        TODO(zihao)
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 k=1,
                 cached=False,
                 bias=True):
        super(SGConv, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        # TODO(zihao): add normalization

    def forward(self, feat, graph):
        graph = graph.local_var()
        if self._cached_h is not None:
            feat = self._cached_h
        else:
            # compute normalization
            degs = graph.in_degrees().float()
            norm = th.pow(degs, -0.5)
            norm[th.isinf(norm)] = 0
            norm = norm.to(feat.device).unsqueeze(1)
            # compute (D^-1 A D) X
            for _ in range(self._k):
                feat = feat * norm
                graph.ndata['h'] = feat
                graph.update_all(fn.copy_u('h', 'm'),
                                 fn.sum('m', 'h'))
                feat = graph.ndata.pop('h')
                feat = feat * norm
            # cache feature
            if self._cached:
                self._cached_h = feat
        return self.fc(feat)


class NNConv(nn.Module):
    """Graph Convolution layer introduced in "Neural Message Passing for Quantum Chemistry".

    Parameters
    ----------
    in_feats : int
    out_feats : int
    edge_nn : torch.nn.Module
    aggregator_type : str
    residual : bool
    bias : bool, optional
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_nn,
                 aggregator_type,
                 residual,
                 bias=True):
        super(NNConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.edge_nn = edge_nn
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO(zihao): initialize root and bias
        pass

    def forward(self, feat, efeat, graph):
        graph = graph.local_var()
        graph.ndata['h'] = feat.unsqueeze(-1) # (n, d_in, 1)
        graph.edata['w'] = self.edge_nn(efeat).view(-1, self._in_feats, self._out_feats) # (n, d_in, d_out)
        graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh')) # (n, d_in, d_out)
        rst = graph.ndata.pop('neigh').sum(dim=1) # (n, d_out)
        # residual connection
        if self.res_fc is not None:
            rst = rst + self.res_fc(feat)
        # bias
        if self.bias is not None:
            rst = rst + self.bias
        return rst


class APPNPConv(nn.Module):
    """Approximate Personalized Propagation of Neural Predictions layer from
    paper "Predict then Propagate: Graph Neural Networks meet Personalized PageRank".

    Parameters
    ----------
    in_feats : int
    out_feats : int
    alpha : float
    k : int
    activation : callable activation function/layer or None, optional
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 alpha,
                 k,
                 activation=None):
        # TODO(zihao): add edge dropout.
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._alpha = alpha
        self._k = k
        self._activation = activation

    def forward(self, feat, graph):
        graph = graph.local_var()
        norm = th.pow(graph.in_degrees().float(), -0.5).unsqueeze(-1).to(feat.device)
        feat_0 = feat
        for _ in range(self._k):
            # normalization by src
            feat = feat * norm
            graph.ndata['h'] = feat
            graph.update_all(fn.copy_src('h', 'm'),
                             fn.sum('m', 'h'))
            feat = graph.ndata.pop('h')
            # normalization by dst
            feat = feat * norm
            feat = (1 - self._alpha) * feat + self._alpha * feat_0
        return feat


class AGNNConv(nn.Module):
    """Attention-based Graph Neural Network layer from paper "Attention-based
     Graph Neural Network for Semi-Supervised Learning"

    Parameters
    ----------
    init_beta : float, optional
    learn_beta : bool, optional
    """
    def __init__(self,
                 init_beta=1,
                 learn_beta=True):
        super(AGNNConv, self).__init__()
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor(init_beta))
        else:
            self.register_buffer('beta', th.Tensor(init_beta))

    def forward(self, feat, graph):
        graph = graph.local_var()
        graph.ndata['norm_h'] = F.normalize(feat, p=2, dim=-1)
        # compute cosine distance
        graph.apply_edges(fn.u_mul_v('norm_h', 'norm_h', 'cos'))
        cos = graph.edata.pop('cos').sum(-1)
        e = self.beta * cos
        graph.edata['p'] = edge_softmax(graph, e)
        graph.update_all(fn.u_mul_e('h', 'p', 'm'), fn.sum('m', 'h'))
        return graph.ndata.pop('h')


class DenseGCNConv(nn.Module):
    """Graph Convolutional Network layer where the graph structure
    is given by an adjacency matrix.
    We recommend user to use this module when inducing graph convolution
    on dense graphs / k-hop graphs.

    Parameters
    ----------
    in_feats : int
    out_feats : int
    norm : bool
    bias : bool
    activation : callable activation function/layer or None, optional
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm=True,
                 bias=True,
                 activation=None):
        super(DenseGCNConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self._activation = activation

    def forward(self, feat, adj):
        adj = adj.float().to(feat.device)
        if self._norm:
            in_degrees = adj.sum(dim=1)
            norm = th.pow(in_degrees, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp).to(feat.device)
            feat = feat * norm

        if self._in_feats > self._out_feats:
            # mult W first to reduce the feature size for aggregation.
            feat = th.matmul(feat, self.weight)
            rst = adj @ feat
        else:
            # aggregate first then mult W
            rst = adj @ feat
            rst = th.matmul(rst, self.weight)

        if self._norm:
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


class DenseSAGEConv(nn.Module):
    """GraphSAGE layer where the graph structure is given by an
    adjacency matrix.
    We recommend user to use this module when inducing GraphSAGE
    operations on dense graphs / k-hop graphs.

    Note that we only support mean aggregator in DenseSAGEConv.

    Parameters
    ----------
    in_feats : int
    out_feats : int
    feat_drop : float
    norm : bool
    bias : bool
    activation : callable activation function/layer or None, optional
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop,
                 bias=True,
                 norm=None,
                 activation=None):
        super(DenseSAGEConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, feat, adj):
        adj = adj.float().to(feat.device)
        feat = self.feat_drop(feat)
        h_self = feat
        if self._aggre_type == 'mean':
            in_degrees = adj.sum(dim=1).unsqueeze(-1)
            h_neigh = (adj @ feat) / in_degrees.clamp(min=1)
        rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self._norm is not None:
            rst = self._norm(rst)
