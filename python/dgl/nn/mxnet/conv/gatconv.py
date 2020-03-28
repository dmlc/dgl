"""MXNet modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ..softmax import edge_softmax
from ....utils import expand_as_pair

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
    in_feats : int or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
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
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._in_feats = in_feats
        self._out_feats = out_feats
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

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            If a mxnet.NDArray is given, the input feature of shape :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of
            nodes.
            If a pair of mxnet.NDArray is given, the pair must contain two tensors of
            shape :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).reshape(
                -1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).reshape(
                -1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).reshape(
                -1, self._num_heads, self._out_feats)
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
        return rst
