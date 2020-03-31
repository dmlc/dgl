"""MXNet Module for Gated Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

from .... import function as fn

class GatedGraphConv(nn.Block):
    r"""Gated Graph Convolution layer from paper `Gated Graph Sequence
    Neural Networks <https://arxiv.org/pdf/1511.05493.pdf>`__.

    .. math::
        h_{i}^{0} & = [ x_i \| \mathbf{0} ]

        a_{i}^{t} & = \sum_{j\in\mathcal{N}(i)} W_{e_{ij}} h_{j}^{t}

        h_{i}^{t+1} & = \mathrm{GRU}(a_{i}^{t}, h_{i}^{t})

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    n_steps : int
        Number of recurrent steps.
    n_etypes : int
        Number of edge types.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
        Can only be set to True in MXNet.
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
        assert graph.is_homograph(), \
            "not a homograph; convert it with to_homo and pass in the edge type as argument"
        graph = graph.local_var()
        zero_pad = nd.zeros((feat.shape[0], self._out_feats - feat.shape[1]), ctx=feat.context)
        feat = nd.concat(feat, zero_pad, dim=-1)

        for _ in range(self._n_steps):
            graph.ndata['h'] = feat
            for i in range(self._n_etypes):
                eids = (etypes.asnumpy() == i).nonzero()[0]
                eids = nd.from_numpy(eids, zero_copy=True)
                if len(eids) > 0:
                    graph.apply_edges(
                        lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                        eids
                    )
            graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
            a = graph.ndata.pop('a')
            feat = self.gru(a, [feat])[0]
        return feat
