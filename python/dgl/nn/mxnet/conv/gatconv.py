"""MXNet modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ..softmax import edge_softmax

#pylint: enable=W0235
class GATConv(nn.Block):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        with self.name_scope():
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

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

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
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).reshape(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l.data(feat.context)).sum(axis=-1).expand_dims(-1)
        er = (feat * self.attn_r.data(feat.context)).sum(axis=-1).expand_dims(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).reshape(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
