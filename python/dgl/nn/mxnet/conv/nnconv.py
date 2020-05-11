"""MXNet Module for NNConv layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity

from .... import function as fn
from ....utils import expand_as_pair


class NNConv(nn.Block):
    r"""Graph Convolution layer introduced in `Neural Message Passing
    for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`__.

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    Parameters
    ----------
    in_feats : int or pair of ints
        Input feature size.

        If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size.
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
            feat_src, feat_dst = expand_as_pair(feat)

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
