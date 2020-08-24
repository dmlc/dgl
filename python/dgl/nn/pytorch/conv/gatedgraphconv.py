"""Torch Module for Gated Graph Convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name, cell-var-from-loop
import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn


class GatedGraphConv(nn.Module):
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

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GatedGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = GatedGraphConv(10, 10, 2, 3)
    >>> etype = th.tensor([0,1,2,0,1,2])
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.4652,  0.4458,  0.5169,  0.4126,  0.4847,  0.2303,  0.2757,  0.7721,
            0.0523,  0.0857],
            [ 0.0832,  0.1388, -0.5643,  0.7053, -0.2524, -0.3847,  0.7587,  0.8245,
            0.9315,  0.4063],
            [ 0.6340,  0.4096,  0.7692,  0.2125,  0.2106,  0.4542, -0.0580,  0.3364,
            -0.1376,  0.4948],
            [ 0.5551,  0.7946,  0.6220,  0.8058,  0.5711,  0.3063, -0.5454,  0.2272,
            -0.6931, -0.1607],
            [ 0.2644,  0.2469, -0.6143,  0.6008, -0.1516, -0.3781,  0.5878,  0.7993,
            0.9241,  0.1835],
            [ 0.6393,  0.3447,  0.3893,  0.4279,  0.3342,  0.3809,  0.0406,  0.5030,
            0.1342,  0.0425]], grad_fn=<AddBackward0>)
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
        self.linears = nn.ModuleList(
            [nn.Linear(out_feats, out_feats) for _ in range(n_etypes)]
        )
        self.gru = nn.GRUCell(out_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        Note
        ----
        The model parameters are initialized using Glorot uniform initialization
        and the bias is initialized to be zero.
        """
        gain = init.calculate_gain('relu')
        self.gru.reset_parameters()
        for linear in self.linears:
            init.xavier_normal_(linear.weight, gain=gain)
            init.zeros_(linear.bias)

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

    def forward(self, graph, feat, etypes):
        """

        Description
        -----------
        Compute Gated Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        etypes : torch.LongTensor
            The edge type tensor of shape :math:`(E,)` where :math:`E` is
            the number of edges of the graph.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            assert graph.is_homogeneous, \
                "not a homogeneous graph; convert it with to_homogeneous " \
                "and pass in the edge type as argument"
            assert etypes.min() >= 0 and etypes.max() < self._n_etypes, \
                "edge type indices out of range [0, {})".format(self._n_etypes)
            zero_pad = feat.new_zeros((feat.shape[0], self._out_feats - feat.shape[1]))
            feat = th.cat([feat, zero_pad], -1)

            for _ in range(self._n_steps):
                graph.ndata['h'] = feat
                for i in range(self._n_etypes):
                    eids = (etypes == i).nonzero().view(-1).type(graph.idtype)
                    if len(eids) > 0:
                        graph.apply_edges(
                            lambda edges: {'W_e*h': self.linears[i](edges.src['h'])},
                            eids
                        )
                graph.update_all(fn.copy_e('W_e*h', 'm'), fn.sum('m', 'a'))
                a = graph.ndata.pop('a') # (N, D)
                feat = self.gru(a, feat)
            return feat
