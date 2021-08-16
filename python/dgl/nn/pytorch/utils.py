"""Utilities for pytorch NN package"""
#pylint: disable=no-member, invalid-name

import torch as th
from torch import nn
from ... import DGLGraph
from ...base import dgl_warning

def matmul_maybe_select(A, B):
    """Perform Matrix multiplication C = A * B but A could be an integer id vector.

    If A is an integer vector, we treat it as multiplying a one-hot encoded tensor.
    In this case, the expensive dense matrix multiply can be replaced by a much
    cheaper index lookup.

    For example,
    ::

        A = [2, 0, 1],
        B = [[0.1, 0.2],
             [0.3, 0.4],
             [0.5, 0.6]]

    then matmul_maybe_select(A, B) is equivalent to
    ::

        [[0, 0, 1],     [[0.1, 0.2],
         [1, 0, 0],  *   [0.3, 0.4],
         [0, 1, 0]]      [0.5, 0.6]]

    In all other cases, perform a normal matmul.

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor

    Returns
    -------
    C : torch.Tensor
        result tensor
    """
    if A.dtype == th.int64 and len(A.shape) == 1:
        return B.index_select(0, A)
    else:
        return th.matmul(A, B)

def bmm_maybe_select(A, B, index):
    """Slice submatrices of A by the given index and perform bmm.

    B is a 3D tensor of shape (N, D1, D2), which can be viewed as a stack of
    N matrices of shape (D1, D2). The input index is an integer vector of length M.
    A could be either:
    (1) a dense tensor of shape (M, D1),
    (2) an integer vector of length M.
    The result C is a 2D matrix of shape (M, D2)

    For case (1), C is computed by bmm:
    ::

        C[i, :] = matmul(A[i, :], B[index[i], :, :])

    For case (2), C is computed by index select:
    ::

        C[i, :] = B[index[i], A[i], :]

    Parameters
    ----------
    A : torch.Tensor
        lhs tensor
    B : torch.Tensor
        rhs tensor
    index : torch.Tensor
        index tensor

    Returns
    -------
    C : torch.Tensor
        return tensor
    """
    if A.dtype == th.int64 and len(A.shape) == 1:
        # following is a faster version of B[index, A, :]
        B = B.view(-1, B.shape[2])
        flatidx = index * B.shape[1] + A
        return B.index_select(0, flatidx)
    else:
        BB = B.index_select(0, index)
        return th.bmm(A.unsqueeze(1), BB).squeeze()

# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x

class Sequential(nn.Sequential):
    r"""

    Description
    -----------
    A squential container for stacking graph neural network modules.

    DGL supports two modes: sequentially apply GNN modules on 1) the same graph or
    2) a list of given graphs. In the second case, the number of graphs equals the
    number of modules inside this container.

    Parameters
    ----------
    *args :
        Sub-modules of torch.nn.Module that will be added to the container in
        the order by which they are passed in the constructor.

    Examples
    --------
    The following example uses PyTorch backend.

    Mode 1: sequentially apply GNN modules on the same graph

    >>> import torch
    >>> import dgl
    >>> import torch.nn as nn
    >>> import dgl.function as fn
    >>> from dgl.nn.pytorch import Sequential
    >>> class ExampleLayer(nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>     def forward(self, graph, n_feat, e_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             graph.apply_edges(fn.u_add_v('h', 'h', 'e'))
    >>>             e_feat += graph.edata['e']
    >>>             return n_feat, e_feat
    >>>
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1, 2, 0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> net = Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    >>> n_feat = torch.rand(3, 4)
    >>> e_feat = torch.rand(9, 4)
    >>> net(g, n_feat, e_feat)
    (tensor([[39.8597, 45.4542, 25.1877, 30.8086],
             [40.7095, 45.3985, 25.4590, 30.0134],
             [40.7894, 45.2556, 25.5221, 30.4220]]),
     tensor([[80.3772, 89.7752, 50.7762, 60.5520],
             [80.5671, 89.3736, 50.6558, 60.6418],
             [80.4620, 89.5142, 50.3643, 60.3126],
             [80.4817, 89.8549, 50.9430, 59.9108],
             [80.2284, 89.6954, 50.0448, 60.1139],
             [79.7846, 89.6882, 50.5097, 60.6213],
             [80.2654, 90.2330, 50.2787, 60.6937],
             [80.3468, 90.0341, 50.2062, 60.2659],
             [80.0556, 90.2789, 50.2882, 60.5845]]))

    Mode 2: sequentially apply GNN modules on different graphs

    >>> import torch
    >>> import dgl
    >>> import torch.nn as nn
    >>> import dgl.function as fn
    >>> import networkx as nx
    >>> from dgl.nn.pytorch import Sequential
    >>> class ExampleLayer(nn.Module):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>     def forward(self, graph, n_feat):
    >>>         with graph.local_scope():
    >>>             graph.ndata['h'] = n_feat
    >>>             graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
    >>>             n_feat += graph.ndata['h']
    >>>             return n_feat.view(graph.number_of_nodes() // 2, 2, -1).sum(1)
    >>>
    >>> g1 = dgl.DGLGraph(nx.erdos_renyi_graph(32, 0.05))
    >>> g2 = dgl.DGLGraph(nx.erdos_renyi_graph(16, 0.2))
    >>> g3 = dgl.DGLGraph(nx.erdos_renyi_graph(8, 0.8))
    >>> net = Sequential(ExampleLayer(), ExampleLayer(), ExampleLayer())
    >>> n_feat = torch.rand(32, 4)
    >>> net([g1, g2, g3], n_feat)
    tensor([[209.6221, 225.5312, 193.8920, 220.1002],
            [250.0169, 271.9156, 240.2467, 267.7766],
            [220.4007, 239.7365, 213.8648, 234.9637],
            [196.4630, 207.6319, 184.2927, 208.7465]])
    """

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)

    def forward(self, graph, *feats):
        r"""

        Sequentially apply modules to the input.

        Parameters
        ----------
        graph : DGLGraph or list of DGLGraphs
            The graph(s) to apply modules on.

        *feats :
            Input features.
            The output of the :math:`i`-th module should match the input
            of the :math:`(i+1)`-th module in the sequential.
        """
        if isinstance(graph, list):
            for graph_i, module in zip(graph, self):
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph_i, *feats)
        elif isinstance(graph, DGLGraph):
            for module in self:
                if not isinstance(feats, tuple):
                    feats = (feats,)
                feats = module(graph, *feats)
        else:
            raise TypeError('The first argument of forward must be a DGLGraph'
                            ' or a list of DGLGraph s')
        return feats

class WeightBasis(nn.Module):
    r"""Basis decomposition module.

    Basis decomposition is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

        W_o = \sum_{b=1}^B a_{ob} V_b

    Each weight output :math:`W_o` is essentially a linear combination of basis
    transformations :math:`V_b` with coefficients :math:`a_{ob}`.

    If is useful as a form of regularization on a large parameter matrix. Thus,
    the number of weight outputs is usually larger than the number of bases.

    Parameters
    ----------
    shape : tuple[int]
        Shape of the basis parameter.
    num_bases : int
        Number of bases.
    num_outputs : int
        Number of outputs.
    """
    def __init__(self,
                 shape,
                 num_bases,
                 num_outputs):
        super(WeightBasis, self).__init__()
        self.shape = shape
        self.num_bases = num_bases
        self.num_outputs = num_outputs

        if num_outputs <= num_bases:
            dgl_warning('The number of weight outputs should be larger than the number'
                        ' of bases.')

        self.weight = nn.Parameter(th.Tensor(self.num_bases, *shape))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        # linear combination coefficients
        self.w_comp = nn.Parameter(th.Tensor(self.num_outputs, self.num_bases))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def forward(self):
        r"""Forward computation

        Returns
        -------
        weight : torch.Tensor
            Composed weight tensor of shape ``(num_outputs,) + shape``
        """
        # generate all weights from bases
        weight = th.matmul(self.w_comp, self.weight.view(self.num_bases, -1))
        return weight.view(self.num_outputs, *self.shape)


def _column_norm(g, edge_features: torch.Tensor, eps: float) -> torch.Tensor:
    with g.local_scope():
        g.edata["_e"] = edge_features
        g.update_all(fn.copy_edge("_e", 'm'), fn.sum('m', '_in_weight'))
        degs = g.dstdata['_in_weight'] + eps
        norm = 1.0 / degs
        g.dstdata['_dst_in_w'] = norm
        g.apply_edges(lambda e: {"_norm_e": e.dst['_dst_in_w'] * e.data["_e"]})
        return g.edata["_norm_e"]


def _row_norm(g, edge_features: torch.Tensor, eps: float) -> torch.Tensor:
    reversed_g = reverse(g)
    return _column_norm(reversed_g, edge_features, eps)


def _doubly_stochastic_norm_adj(g, edge_features, eps: float) -> torch.Tensor:
    _, p = edge_features
    num_nodes = g.num_nodes()
    with g.local_scope():
        g.edata["_row_norm_e"] = _row_norm(g, edge_features, eps)
        g.edata["_col_norm_e"] = _column_norm(g, g.edata["_row_norm_e"], eps)

        # adjacency matrix for features e_tilde
        u, v = g.all_edges(order='eid')
        row_norm_adj = torch.zeros((num_nodes, num_nodes, p))
        row_norm_adj[u, v] = g.edata['_row_norm_e']

        # transposed adjacency matrix for features e_dot
        column_norm_adj = torch.zeros((num_nodes, num_nodes, p))
        # transpose by switching node order
        column_norm_adj[v, u] = g.edata['_col_norm_e']

    # put matrices in format expected by torch.bmm
    ds_adj_matrix = torch.bmm(
        row_norm_adj.permute(2, 0, 1), column_norm_adj.permute(2, 0, 1)
    ).permute(1, 2, 0)
    return ds_adj_matrix


class TensorEdgeWeightNorm(nn.Module):
    """
    Description
    -----------
    Parameters
    ----------
    norm: str, optional
        f

    eps: float, optional
        f

    Example
    -------
    """

    def __init__(self, norm='both', eps=0.):
        super(TensorEdgeWeightNorm, self).__init__()
        self._norm = norm
        self._eps = eps

    def forward(self, graph, edge_feature):
        with graph.local_scope():
            doubly_stochastic_adj = _doubly_stochastic_norm_adj(g, edge_feature, self._eps)
            # TODO: get graph from adj matrix
