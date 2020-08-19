"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
from torch import nn

from .... import function as fn
from ....ops import edge_softmax
from ....base import DGLError
from ....utils import expand_as_pair


class DotGatConv(nn.Module):
    r"""

    Description
    -----------
    Apply dot product version of self attention in GCN.

        .. math::
            h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i, j} h_j^{(l)}

        where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and node :math:`j`:

        .. math::
            \alpha_{i, j} &= \mathrm{softmax_i}(e_{ij}^{l})

            e_{ij}^{l} &= ({W_i^{(l)} h_i^{(l)}})^T \cdot {W_j^{(l)} h_j^{(l)}}

        where :math:`W_i` and :math:`W_j` transform node :math:`i`'s and node :math:`j`'s
        features into the same dimension, so that when compute note features' similarity,
        it can use dot-product.

    Parameters
    ----------
    in_feats : int, or pair of ints
        Input feature size; i.e, the number of dimensions of :math:`h_i^{(l)}`.
        DotGatConv can be applied on homogeneous graph and unidirectional
        `bipartite graph <https://docs.dgl.ai/generated/dgl.bipartite.html?highlight=bipartite>`__.
        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
    out_feats : int
        Output feature size; i.e, the number of dimensions of :math:`h_i^{(l+1)}`.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zere-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zere-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import DotGatConv

    >>> # Case 1: Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10)
    >>> gatconv = DotGatConv(10, 2)
    >>> res = gatconv(g, feat)
    >>> res
    tensor([[-0.6958, -0.8752],
            [-0.6958, -0.8752],
            [-0.6958, -0.8752],
            [-0.6958, -0.8752],
            [-0.6958, -0.8752],
            [-0.6958, -0.8752]], grad_fn=<CopyReduceBackward>)

    >>> # Case 2: Unidirectional bipartite graph
    >>> u = [0, 1, 0, 0, 1]
    >>> v = [0, 1, 2, 3, 2]
    >>> g = dgl.bipartite((u, v))
    >>> u_feat = th.tensor(np.random.rand(2, 5).astype(np.float32))
    >>> v_feat = th.tensor(np.random.rand(4, 10).astype(np.float32))
    >>> gatconv = DotGatConv((5,10), 2)
    >>> res = gatconv(g, (u_feat, v_feat))
    >>> res
    tensor([[ 0.4718,  0.0864],
            [ 0.7099, -0.0335],
            [ 0.5869,  0.0284],
            [ 0.4718,  0.0864]], grad_fn=<CopyReduceBackward>)
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 allow_zero_in_degree=False):
        super(DotGatConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(self._in_src_feats, self._out_feats, bias=False)
            self.fc_dst = nn.Linear(self._in_dst_feats, self._out_feats, bias=False)
        else:
            self.fc = nn.Linear(self._in_src_feats, self._out_feats, bias=False)

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Apply dot product version of self attention in GCN.

        Parameters
        ----------
        graph: DGLGraph or bi_partities graph
            The graph
        feat: torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}` is size
            of output feature.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """

        graph = graph.local_var()

        if not self._allow_zero_in_degree:
            if (graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        # check if feat is a tuple
        if isinstance(feat, tuple):
            h_src = feat[0]
            h_dst = feat[1]
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            h_src = feat
            feat_src = feat_dst = self.fc(h_src)
            if graph.is_block:
                feat_dst = feat_src[:graph.number_of_dst_nodes()]

        # Assign features to nodes
        graph.srcdata.update({'ft': feat_src})
        graph.dstdata.update({'ft': feat_dst})

        # Step 1. dot product
        graph.apply_edges(fn.u_dot_v('ft', 'ft', 'a'))

        # Step 2. edge softmax to compute attention scores
        graph.edata['sa'] = edge_softmax(graph, graph.edata['a'])

        # Step 3. Broadcast softmax value to each edge, and aggregate dst node
        graph.update_all(fn.u_mul_e('ft', 'sa', 'attn'), fn.sum('attn', 'agg_u'))

        # output results to the destination nodes
        rst = graph.dstdata['agg_u']

        return rst
