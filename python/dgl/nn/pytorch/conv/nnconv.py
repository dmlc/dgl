"""Torch Module for NNConv layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn
from ..utils import Identity
from ....utils import expand_as_pair


class NNConv(nn.Module):
    r"""Graph Convolution layer introduced in `Neural Message Passing
    for Quantum Chemistry <https://arxiv.org/pdf/1704.01212.pdf>`__.

    .. math::
        h_{i}^{l+1} = h_{i}^{l} + \mathrm{aggregate}\left(\left\{
        f_\Theta (e_{ij}) \cdot h_j^{l}, j\in \mathcal{N}(i) \right\}\right)

    where :math:`e_{ij}` is the edge feature, :math:`f_\Theta` is a function with learnable parameters.

    Notes
    -----
    Zero in degree nodes could lead to invalid output. A common practice
    to avoid this is to add a self-loop for each node in the graph if it's homogeneous,
    which can be achieved by:

    >>> g = ... # some homogeneous graph
    >>> dgl.add_self_loop(g)

    For Unidirectional bipartite graph, we need to filter out the destination nodes with zero in-degree when use in downstream.

    Parameters
    ----------
    in_feats : int
        Input feature size.

        GATConv can be applied on homogeneous graph and unidirectional `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`. If the layer is to be applied on a unidirectional bipartite graph, ``in_feats``
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

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import NNConv

    Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(5, 20)
    >>> def edge_func(efeat):
    ...     return lin(efeat)
    >>> efeat = th.ones(6, 5)
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> res = conv(g, feat, efeat)
    >>> res
    tensor([[0.5983, 1.7884],
            [0.5983, 1.7884],
            [0.5983, 1.7884],
            [0.5983, 1.7884],
            [0.5983, 1.7884],
            [0.0000, 0.0000]], grad_fn=<AddBackward0>)

    Case 2: Unidirectional bipartite graph
    >>> u = [0, 0, 1]
    >>> v = [2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = th.tensor(np.random.rand(2, 10).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> conv = NNConv(10, 2, edge_func, 'mean')
    >>> efeat = th.ones(3, 5)
    >>> res = conv(g, (u_feat, v_feat), efeat)
    >>> res
    tensor([[0.0000, 0.0000],
            [0.0000, 0.0000],
            [0.4474, 0.6424],
            [0.4315, 0.9775]], grad_fn=<AddBackward0>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_func,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.edge_func = edge_func
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
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
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
        """Reinitialize learnable parameters."""
        gain = init.calculate_gain('relu')
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, efeat):
        r"""Compute MPNN Graph Convolution layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`N`
            is the number of nodes of the graph and :math:`D_{in}` is the
            input feature size.
        efeat : torch.Tensor
            The edge feature of shape :math:`(N, *)`, should fit the input
            shape requirement of ``edge_func``.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is the output feature size.
        """
        with graph.local_scope():
            if self._add_self_loop:
                graph = transform.add_self_loop(graph)

            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)
            # (n, d_in, d_out)
            graph.edata['w'] = self.edge_func(efeat).view(-1, self._in_src_feats, self._out_feats)
            # (n, d_in, d_out)
            graph.update_all(fn.u_mul_e('h', 'w', 'm'), self.reducer('m', 'neigh'))
            rst = graph.dstdata['neigh'].sum(dim=1) # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst
