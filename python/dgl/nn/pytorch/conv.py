"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from ... import function as fn
from .softmax import edge_softmax

__all__ = ['GraphConv', 'GraphAttention', 'GraphSAGE']

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
            * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
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
    

class GraphAttention(nn.Module):
    r"""Apply graph attention over an input signal.

    TODO(zihao): docstring
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
        super(GraphAttention, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop else lambda x : x
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x : x
        self.leaky_relu = nn.LeakyReLU(alpha)
        self._residual = residual
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = None
        self._reset_parameters()

        self.activation = activation

    def _reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)
        if self._residual and self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=1.414)

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
            if self.res_fc is not None:
                resval = self.res_fc(h).view(-1, self._num_heads, self._out_feats)
            else:
                resval = h.unsqueeze(1)
            rst = rst + resval

        # activation
        if self.activation:
            rst = self.activation(rst)

        return rst

    def extra_repr(self):
        pass


class GraphSAGE(nn.Module):
    r"""Apply GraphSAGE layer over an input signal.

    TODO(zihao): docstring
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 feat_drop,
                 aggregator_type,
                 bias=False,
                 norm=None,
                 activation=None):
        super(GraphSAGE, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self._norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/pool/lstm/gcn
        if aggregator_type == 'pool':
            self.fc_pool = nn.Linear(in_feats, in_feats)
        else: # lstm
            self.lstm = nn.LSTM(in_feats, in_feats, batch_first=True)

        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(in_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        else: # lstm
            pass

        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def _lstm_reducer(self, nodes):
        # TODO(zihao): this implementation is slow.
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
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.sum('m', 'neigh')
            )
            # divide in_degrees
            degs = graph.in_degrees().float()
            degs = degs.to(feat.device)
            eps = 1e-8
            h_neigh = (graph.ndata['neigh'] + eps) / (degs.unsqueeze(-1) + eps)
        elif self._aggre_type == 'gcn':
            graph.ndata['h'] = feat
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.sum('m', 'neigh')
            )
            # divide in_degrees
            degs = graph.in_degrees().float()
            degs = degs.to(feat.device)
            h_neigh = (graph.ndata['neigh'] + graph.ndata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'pool':
            graph.ndata['h'] = F.relu(self.fc_pool(feat))
            graph.update_all(
                fn.copy_src('h', 'm'),
                fn.max('m', 'neigh')
            )
            h_neigh = graph.ndata['neigh']
        else: # lstm:
            graph.ndata['h'] = feat
            graph.update_all(
                fn.copy_src('h', 'm'),
                self._lstm_reducer,
            )
            h_neigh = graph.ndata['neigh']

        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

        if self.activation is not None:
            rst = self.activation(rst)

        if self._norm is not None:
            rst = self._norm(rst)

        return rst

    def extra_repr(self):
        pass
