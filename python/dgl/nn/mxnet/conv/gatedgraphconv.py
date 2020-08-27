"""MXNet Module for Gated Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

from .... import function as fn

class GatedGraphConv(nn.Block):
    r"""

    Description
    -----------
    Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} &= [ x_i \| \mathbf{0} ]

        a_{i}^{t} &= \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} &= \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`x_i`.
    out_feats : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(t+1)}`.
    n_steps : int
        Number of recurrent steps; i.e, the :math:`t` in the above formula.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
        Can only be set to True in MXNet.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import mxnet as mx
    >>> from dgl.nn import GatedGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = mx.nd.ones((6, 10))
    >>> conv = GatedGraphConv(10, 10, 2, 3)
    >>> conv.initialize(ctx=mx.cpu(0))
    >>> etype = mx.nd.array([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    [[0.24378185 0.17402579 0.2644723  0.2740628  0.14041871 0.32523093
    0.2703067  0.18234392 0.32777587 0.30957845]
    [0.17872348 0.28878236 0.2509409  0.20139427 0.3355541  0.22643831
    0.2690711  0.22341749 0.27995753 0.21575949]
    [0.23911178 0.16696918 0.26120248 0.27397877 0.13745922 0.3223175
    0.27561218 0.18071817 0.3251124  0.30608907]
    [0.25242943 0.3098581  0.25249368 0.27968448 0.24624602 0.12270881
    0.335147   0.31550157 0.19065917 0.21087633]
    [0.17503153 0.29523152 0.2474858  0.20848347 0.3526433  0.23443702
    0.24741334 0.21986549 0.28935105 0.21859099]
    [0.2159364  0.26942077 0.23083271 0.28329757 0.24758333 0.24230732
    0.23958017 0.23430146 0.26431587 0.27001363]]
    <NDArray 6x10 @cpu(0)>
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 n_steps,
                 n_etypes,
                 bias=True):
        super(GatedGraphConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._n_steps = n_steps
        self._n_etypes = n_etypes
        if not bias:
            raise KeyError('MXNet do not support disabling bias in GRUCell.')
        with self.name_scope():
            self.linears = nn.Sequential()
            for _ in range(n_etypes):
                self.linears.add(
                    nn.Dense(out_feats,
                             weight_initializer=mx.init.Xavier(),
                             in_units=out_feats)
                )
            self.gru = gluon.rnn.GRUCell(out_feats, input_size=out_feats)

    def forward(self, graph, feat, etypes):
        """Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : mxnet.NDArray
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        mxnet.NDArray
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            zero_pad = nd.zeros((feat.shape[0], self._out_feats - feat.shape[1]),
                                ctx=feat.context)
            feat = nd.concat(feat, zero_pad, dim=-1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = feat
                for i in range(self._n_etypes):
                    eids = (etypes.asnumpy() == i).nonzero()[0]
                    eids = nd.from_numpy(eids, zero_copy=True).as_in_context(
                        feat.context).astype(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a')
                feat = self.gru(a, [feat])[0]
            return feat
