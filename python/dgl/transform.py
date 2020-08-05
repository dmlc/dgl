"""Module for graph transformation utilities."""

from collections.abc import Iterable, Mapping
from collections import defaultdict
import time
import numpy as np
from scipy import sparse

from ._ffi.function import _init_api
from .base import EID, NID, dgl_warning, DGLError, is_internal_column
from . import convert
from .heterograph import DGLHeteroGraph, DGLBlock
from . import ndarray as nd
from . import backend as F
from . import utils, batch
from .partition import metis_partition_assignment as hetero_metis_partition_assignment
from .partition import partition_graph_with_halo as hetero_partition_graph_with_halo
from .partition import metis_partition as hetero_metis_partition

# TO BE DEPRECATED
from ._deprecate.graph import DGLGraph as DGLGraphStale
from .graph_index import _get_halo_subgraph_inner_node

__all__ = [
    'line_graph',
    'khop_adj',
    'khop_graph',
    'reverse',
    'to_bidirected',
    'to_bidirected_stale',
    'add_reverse_edges',
    'laplacian_lambda_max',
    'knn_graph',
    'segmented_knn_graph',
    'add_edges',
    'add_nodes',
    'remove_edges',
    'remove_nodes',
    'add_self_loop',
    'remove_self_loop',
    'metapath_reachable_graph',
    'compact_graphs',
    'to_block',
    'to_simple',
    'to_simple_graph',
    'in_subgraph',
    'out_subgraph',
    'as_immutable_graph',
    'as_heterograph']


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)

#pylint: disable=invalid-name
def knn_graph(x, k):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix. The predecessors of each point are its k-nearest
    neighbors.

    If a 3D tensor is given instead, then each row would be transformed into
    a separate graph.  The graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.

        If 2D, each row of ``x`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    dist = pairwise_squared_distance(x)
    k_indices = F.argtopk(dist, k, 2, descending=False)
    dst = F.copy_to(k_indices, F.cpu())

    src = F.zeros_like(dst) + F.reshape(F.arange(0, n_points), (1, -1, 1))

    per_sample_offset = F.reshape(F.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix(
        (F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))),
        shape=(n_samples * n_points, n_samples * n_points))

    return convert.graph(adj)

#pylint: disable=invalid-name
def segmented_knn_graph(x, k, segs):
    """Transforms the given point set to a directed graph, whose coordinates
    are given as a matrix.  The predecessors of each point are its k-nearest
    neighbors.

    The matrices are concatenated along the first axis, and are segmented by
    ``segs``.  Each block would be transformed into a separate graph.  The
    graphs will be unioned.

    Parameters
    ----------
    x : Tensor
        The input tensor.
    k : int
        The number of neighbors
    segs : iterable of int
        Number of points of each point set.
        Must sum up to the number of rows in ``x``.

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``x``.
    """
    n_total_points, _ = F.shape(x)
    offset = np.insert(np.cumsum(segs), 0, 0)

    h_list = F.split(x, segs, 0)
    dst = [
        F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False) +
        int(offset[i])
        for i, h_g in enumerate(h_list)]
    dst = F.cat(dst, 0)
    src = F.arange(0, n_total_points).unsqueeze(1).expand(n_total_points, k)

    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    g = convert.graph(adj)
    return g

def to_bidirected(g, readonly=None, copy_ndata=False):
    r""" Convert the graph to a bidirected one.

    The function generates a new graph with no edge features.
    If g has an edge for i->j but no edge for j->i, then the
    returned graph will have both i->j and j->i.

    For a heterograph with multiple edge types, we can treat edges corresponding
    to each type as a separate graph and convert the graph to a bidirected one
    for each of them.

    Since **to_bidirected is not well defined for unidirectional bipartite graphs**,
    an error will be raised if an edge type of the input heterograph is for a
    unidirectional bipartite graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Deprecated. There will be no difference between readonly and non-readonly
    copy_ndata: bool, optional
        If True, the node features of the bidirected graph are copied from the
        original graph. If False, the bidirected graph will not have any node features.
        (Default: False)

    Notes
    -----
    Please make sure g is a single graph.

    Returns
    -------
    dgl.DGLGraph
        The bidirected graph

    Examples
    --------
    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    The graph already have i->j and j->i

    >>> g = dgl.graph((th.tensor([0, 1, 2, 0]), th.tensor([1, 2, 0, 2])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    **Heterographs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2]), th.tensor([1, 1, 2, 0])),
    >>>     ('user', 'follows', 'user'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> })
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges(etype='wins')
    (tensor([0, 0, 1, 1, 2, 2]), tensor([1, 2, 0, 2, 0, 1]))
    >>> bg1.edges(etype='follows')
    (tensor([1, 1, 2]), tensor([1, 2, 1]))
    """
    if readonly is not None:
        dgl_warning("Parameter readonly is deprecated" \
            "There will be no difference between readonly and non-readonly DGLGraph")

    for c_etype in g.canonical_etypes:
        if c_etype[0] != c_etype[2]:
            assert False, "to_bidirected is not well defined for " \
                "unidirectional bipartite graphs" \
                ", but {} is unidirectional bipartite".format(c_etype)

    assert g.is_multigraph is False, "to_bidirected only support simple graph"

    g = add_reverse_edges(g, copy_ndata=copy_ndata, copy_edata=False)
    g = to_simple(g, return_counts=None, copy_ndata=copy_ndata, copy_edata=False)
    return g

def add_reverse_edges(g, readonly=None, copy_ndata=True,
                      copy_edata=False, ignore_bipartite=False):
    r"""Add reverse edges to a graph

    For a graph with edges :math:`(i_1, j_1), \cdots, (i_n, j_n)`, this
    function creates a new graph with edges
    :math:`(i_1, j_1), \cdots, (i_n, j_n), (j_1, i_1), \cdots, (j_n, i_n)`.

    For a heterograph with multiple edge types, we can treat edges corresponding
    to each type as a separate graph and add reverse edges for each of them.

    Since **add_reverse_edges is not well defined for unidirectional bipartite graphs**,
    an error will be raised if an edge type of the input heterograph is for a
    unidirectional bipartite graph. We can simply skip the edge types corresponding
    to unidirectional bipartite graphs by specifying ``ignore_bipartite=True``.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Deprecated. There will be no difference between readonly and non-readonly
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the features of the reversed edges will be identical to
        the original ones."
        If False, the new graph will not have any edge
        features.
        (Default: False)
    ignore_bipartite: bool, optional
        If True, unidirectional bipartite graphs are ignored and
        no error is raised. If False, an error  will be raised if
        an edge type of the input heterograph is for a unidirectional
        bipartite graph.

    Returns
    -------
    dgl.DGLGraph
        The graph with reversed edges added.

    Notes
    -----
    If ``copy_ndata`` is ``True``, same tensors are used as
    the node features of the original graph and the new graph.
    As a result, users should avoid performing in-place operations
    on the node features of the new graph to avoid feature corruption.
    On the contrary, edge features are concatenated,
    and they are not shared due to concatenation.
    For concrete examples, refer to the ``Examples`` section below.


    Examples
    --------
    **Homographs**

    >>> g = dgl.graph(th.tensor([0, 0]), th.tensor([0, 1]))
    >>> bg1 = dgl.add_reverse_edges(g)
    >>> bg1.edges()
    (tensor([0, 0, 0, 1]), tensor([0, 1, 0, 0]))

    To remove duplicate edges, see :func:to_simple

    **Heterographs with Multiple Edge Types**

    g = dgl.heterograph({
    >>>     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    >>>     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1])),
    >>>     ('user', 'follows', 'user'): (th.tensor([1, 2, 1), th.tensor([0, 0, 0]))
    >>> })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['wins'].data['h'] = th.tensor([0, 1, 2, 3, 4])

    The add_reverse_edges operation is applied to the subgraph
    corresponding to ('user', 'wins', 'user') and the
    subgraph corresponding to ('user', 'follows', 'user).
    The unidirectional bipartite subgraph ('user', 'plays', 'game')
    is ignored. Both the node features and edge features
    are shared.

    >>> bg = dgl.add_reverse_edges(g, copy_ndata=True,
                               copy_edata=True, ignore_bipartite=True)
    >>> bg.edges(('user', 'wins', 'user'))
    (tensor([0, 2, 0, 2, 2, 1, 1, 2, 1, 0]), tensor([1, 1, 2, 1, 0, 0, 2, 0, 2, 2]))
    >>> bg.edges(('user', 'follows', 'user'))
    (tensor([1, 2, 1, 0, 0, 0]), tensor([0, 0, 0, 1, 2, 1]))
    >>> bg.edges(('user', 'plays', 'game'))
    (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> bg.nodes['game'].data['hv']
    tensor([0, 0, 0])
    >>> bg.edges[('user', 'wins', 'user')].data['h']
    th.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    """
    if readonly is not None:
        dgl_warning("Parameter readonly is deprecated" \
            "There will be no difference between readonly and non-readonly DGLGraph")

    canonical_etypes = g.canonical_etypes
    # fast path
    if ignore_bipartite is False:
        subgs = {}
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                assert False, "add_reverse_edges is not well defined for " \
                    "unidirectional bipartite graphs" \
                    ", but {} is unidirectional bipartite".format(c_etype)

            u, v = g.edges(form='uv', order='eid', etype=c_etype)
            subgs[c_etype] = (F.cat([u, v], dim=0), F.cat([v, u], dim=0))

        new_g = convert.heterograph(subgs)
    else:
        subgs = {}
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                u, v = g.edges(form='uv', order='eid', etype=c_etype)
                subgs[c_etype] = (u, v)
            else:
                u, v = g.edges(form='uv', order='eid', etype=c_etype)
                subgs[c_etype] = (F.cat([u, v], dim=0), F.cat([v, u], dim=0))

        new_g = convert.heterograph(subgs)

    # handle features
    if copy_ndata:
        # for each ntype
        for ntype in g.ntypes:
            # for each data field
            for k in g.nodes[ntype].data:
                new_g.nodes[ntype].data[k] = g.nodes[ntype].data[k]

    if copy_edata:
        # for each etype
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                # for each data field
                for k in g.edges[c_etype].data:
                    new_g.edges[c_etype].data[k] = g.edges[c_etype].data[k]
            else:
                for k in g.edges[c_etype].data:
                    new_g.edges[c_etype].data[k] = \
                        F.cat([g.edges[c_etype].data[k], g.edges[c_etype].data[k]], dim=0)
    return new_g

def line_graph(g, backtracking=True, shared=False):
    """Return the line graph of this graph.

    The graph should be an directed homogeneous graph. Aother type of graphs
    are not supported right now.

    All node features and edge features are not copied to the output

    Parameters
    ----------
    g : DGLGraph
        Input graph.
    backtracking : bool, optional
        Whether the pair of (v, u) (u, v) edges are treated as linked. Default True.
    shared : bool, optional
        Whether to copy the edge features of the original graph as the node features
        of the result line graph.

    Returns
    -------
    G : DGLHeteroGraph
        The line graph of this graph.

    Examples:
    A = [[0, 0, 1],
            [1, 0, 1],
            [1, 1, 0]]
    >>> g = dgl.graph(([0, 1, 1, 2, 2],[2, 0, 2, 0, 1]), 'user', 'follows')
    >>> lg = g.line_graph()
    >>> lg
    ... Graph(num_nodes=5, num_edges=8,
    ... ndata_schemes={}
    ... edata_schemes={})
    >>> lg.edges()
    ... (tensor([0, 0, 1, 2, 2, 3, 4, 4]), tensor([3, 4, 0, 3, 4, 0, 1, 2]))
    >>>
    >>> lg = g.line_graph(backtracking=False)
    >>> lg
    ... Graph(num_nodes=5, num_edges=4,
    ... ndata_schemes={}
    ... edata_schemes={})
    >>> lg.edges()
    ... (tensor([0, 1, 2, 4]), tensor([4, 0, 3, 1]))

    """
    assert g.is_homogeneous(), \
        'line_heterograph only support directed homogeneous graph right now'
    lg = DGLHeteroGraph(_CAPI_DGLHeteroLineGraph(g._graph, backtracking))
    if shared:
        # copy edge features
        lg.ndata.update(g.edata)
    return lg

DGLHeteroGraph.line_graph = line_graph

def khop_adj(g, k):
    """Return the matrix of :math:`A^k` where :math:`A` is the adjacency matrix of :math:`g`,
    where a row represents the destination and a column represents the source.

    Parameters
    ----------
    g : dgl.DGLGraph
        The input graph.
    k : int
        The :math:`k` in :math:`A^k`.

    Returns
    -------
    tensor
        The returned tensor, dtype is ``np.float32``.

    Examples
    --------

    >>> import dgl
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0])
    >>> dgl.khop_adj(g, 1)
    tensor([[1., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0.],
            [0., 0., 0., 1., 1.]])
    >>> dgl.khop_adj(g, 3)
    tensor([[1., 0., 1., 3., 3.],
            [3., 1., 0., 1., 3.],
            [3., 3., 1., 0., 1.],
            [1., 3., 3., 1., 0.],
            [0., 1., 3., 3., 1.]])
    """
    adj_k = g.adj(scipy_fmt=g.formats()['created'][0]) ** k
    return F.tensor(adj_k.todense().astype(np.float32))

def khop_graph(g, k):
    """Return the graph that includes all :math:`k`-hop neighbors of the given graph as edges.
    The adjacency matrix of the returned graph is :math:`A^k`
    (where :math:`A` is the adjacency matrix of :math:`g`).

    Parameters
    ----------
    g : dgl.DGLGraph
        The input graph.
    k : int
        The :math:`k` in `k`-hop graph.

    Returns
    -------
    dgl.DGLGraph
        The returned ``DGLGraph``.

    Examples
    --------

    Below gives an easy example:

    >>> import dgl
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1], [1, 2])
    >>> g_2 = dgl.transform.khop_graph(g, 2)
    >>> print(g_2.edges())
    (tensor([0]), tensor([2]))

    A more complicated example:

    >>> import dgl
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0])
    >>> dgl.khop_graph(g, 1)
    DGLGraph(num_nodes=5, num_edges=10,
             ndata_schemes={}
             edata_schemes={})
    >>> dgl.khop_graph(g, 3)
    DGLGraph(num_nodes=5, num_edges=40,
             ndata_schemes={}
             edata_schemes={})
    """
    n = g.number_of_nodes()
    adj_k = g.adj(transpose=True, scipy_fmt=g.formats()['created'][0]) ** k
    adj_k = adj_k.tocoo()
    multiplicity = adj_k.data
    row = np.repeat(adj_k.row, multiplicity)
    col = np.repeat(adj_k.col, multiplicity)
    # TODO(zihao): we should support creating multi-graph from scipy sparse matrix
    # in the future.
    return convert.graph((row, col), num_nodes=n)

def reverse(g, copy_ndata=True, copy_edata=False, *, share_ndata=None, share_edata=None):
    r"""Return the reverse of a graph.

    The reverse (also called converse, transpose) of a graph with edges
    :math:`(i_1, j_1), (i_2, j_2), \cdots` is a new graph with edges
    :math:`(j_1, i_1), (j_2, i_2), \cdots`.

    For a heterograph with multiple edge types, we can treat edges corresponding
    to each type as a separate graph and compute the reverse for each of them.
    If the original edge type is (A, B, C), its reverse will have edge type (C, B, A).

    Given a :class:`dgl.DGLGraph` object, we return another :class:`dgl.DGLGraph`
    object representing its reverse.

    Parameters
    ----------
    g : dgl.DGLGraph
        The input graph.
    copy_ndata: bool, optional
        If True, the node features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any edge features.
        (Default: False)

    Return
    ------
    dgl.DGLGraph
        The reversed graph.

    Notes
    -----
    If ``copy_ndata`` or ``copy_edata`` is ``True``, same tensors will be used for
    the features of the original graph and the reversed graph to save memory cost.
    As a result, users
    should avoid performing in-place operations on the features of the reversed
    graph, which will corrupt the features of the original graph as well. For
    concrete examples, refer to the ``Examples`` section below.

    Examples
    --------
    **Homographs or Heterographs with A Single Edge Type**

    Create a graph to reverse.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.]])

    Reverse the graph.

    >>> rg = dgl.reverse(g, copy_edata=True)
    >>> rg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])

    The i-th edge in the reversed graph corresponds to the i-th edge in the
    original graph. When ``copy_edata`` is ``True``, they have the same features.

    >>> rg.edges()
    (tensor([1, 2, 0]), tensor([0, 1, 2]))
    >>> rg.edata['h']
    tensor([[3.],
            [4.],
            [5.]])

    **In-place operations on features of one graph will be reflected on features of
    its reverse, which is dangerous. Out-place operations will not be reflected.**

    >>> rg.ndata['h'] += 1
    >>> g.ndata['h']
    tensor([[1.],
            [2.],
            [3.]])
    >>> g.ndata['h'] += 1
    >>> rg.ndata['h']
    tensor([[2.],
            [3.],
            [4.]])
    >>> rg.ndata['h2'] = th.ones(3, 1)
    >>> 'h2' in g.ndata
    False

    **Heterographs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'follows', 'user'): (th.tensor([0, 2]), th.tensor([1, 2])),
    >>>     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The reverse of the graph above can be obtained by combining the reverse of the
    subgraph corresponding to ('user', 'follows', 'user') and the subgraph corresponding
    to ('user', 'plays', 'game'). The reverse for a graph with relation (h, r, t) will
    have relation (t, r, h).

    >>> rg = dgl.reverse(g, copy_ndata=True)
    >>> rg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'follows', 'user'): 2, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> rg.edges(etype='follows')
    (tensor([1, 2]), tensor([0, 2]))
    >>> rg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> rg.nodes['game'].data['hv]
    tensor([[1.],
            [1.],
            [1.]])
    >>> rg.edges['plays'].data
    {}
    """
    if share_ndata is not None:
        dgl_warning('share_ndata argument has been renamed to copy_ndata.')
        copy_ndata = share_ndata
    if share_edata is not None:
        dgl_warning('share_edata argument has been renamed to copy_edata.')
        copy_edata = share_edata
    if g.is_block:
        raise DGLError('Reversing a block graph is not allowed.')
    # TODO(0.5 release, xiangsx) need to handle BLOCK
    # currently reversing a block results in undefined behavior
    gidx = g._graph.reverse()
    new_g = DGLHeteroGraph(gidx, g.ntypes, g.etypes)

    # handle ndata
    if copy_ndata:
        # for each ntype
        for ntype in g.ntypes:
            # for each data field
            for k in g.nodes[ntype].data:
                new_g.nodes[ntype].data[k] = g.nodes[ntype].data[k]

    # handle edata
    if copy_edata:
        # for each etype
        for etype in g.etypes:
            # for each data field
            for k in g.edges[etype].data:
                new_g.edges[etype].data[k] = g.edges[etype].data[k]

    return new_g

DGLHeteroGraph.reverse = reverse

def to_simple_graph(g):
    """Convert the graph to a simple graph with no multi-edge.

    DEPRECATED: renamed to dgl.to_simple

    Parameters
    ----------
    g : DGLGraph
        The input graph.

    Returns
    -------
    DGLGraph
        A simple graph.
    """
    dgl_warning('dgl.to_simple_graph is renamed to dgl.to_simple in v0.5.')
    return to_simple(g)

def to_bidirected_stale(g, readonly=True):
    """Convert the graph to a bidirected graph.

    The function generates a new graph with no node/edge feature.
    If g has an edge for i->j but no edge for j->i, then the
    returned graph will have both i->j and j->i.

    If the input graph is a multigraph (there are multiple edges from node i to node j),
    the returned graph isn't well defined.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Whether the returned bidirected graph is readonly or not.

    Notes
    -----
    Please make sure g is a single graph, otherwise the return value is undefined.

    Returns
    -------
    DGLGraph

    Examples
    --------
    The following two examples use PyTorch backend, one for non-multi graph
    and one for multi-graph.

    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(2)
    >>> g.add_edges([0, 0], [0, 1])
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 0]), tensor([0, 0, 1]))
    """
    if readonly:
        newgidx = _CAPI_DGLToBidirectedImmutableGraph(g._graph)
    else:
        newgidx = _CAPI_DGLToBidirectedMutableGraph(g._graph)
    return DGLGraphStale(newgidx)

def laplacian_lambda_max(g):
    """Return the largest eigenvalue of the normalized symmetric laplacian of g.

    The eigenvalue of the normalized symmetric of any graph is less than or equal to 2,
    ref: https://en.wikipedia.org/wiki/Laplacian_matrix#Properties

    Parameters
    ----------
    g : DGLGraph
        The input graph, it should be an undirected graph.

    Returns
    -------
    list :
        Return a list, where the i-th item indicates the largest eigenvalue
        of i-th graph in g.

    Examples
    --------

    >>> import dgl
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 4, 0, 1, 2, 3])
    >>> dgl.laplacian_lambda_max(g)
    [1.809016994374948]
    """
    g_arr = batch.unbatch(g)
    rst = []
    for g_i in g_arr:
        n = g_i.number_of_nodes()
        adj = g_i.adj(scipy_fmt=g_i.formats()['created'][0]).astype(float)
        norm = sparse.diags(F.asnumpy(g_i.in_degrees()).clip(1) ** -0.5, dtype=float)
        laplacian = sparse.eye(n) - norm * adj * norm
        rst.append(sparse.linalg.eigs(laplacian, 1, which='LM',
                                      return_eigenvectors=False)[0].real)
    return rst

def metapath_reachable_graph(g, metapath):
    """Return a graph where the successors of any node ``u`` are nodes reachable from ``u`` by
    the given metapath.

    If the beginning node type ``s`` and ending node type ``t`` are the same, it will return
    a homogeneous graph with node type ``s = t``.  Otherwise, a unidirectional bipartite graph
    with source node type ``s`` and destination node type ``t`` is returned.

    In both cases, two nodes ``u`` and ``v`` will be connected with an edge ``(u, v)`` if
    there exists one path matching the metapath from ``u`` to ``v``.

    The result graph keeps the node set of type ``s`` and ``t`` in the original graph even if
    they might have no neighbor.

    The features of the source/destination node type in the original graph would be copied to
    the new graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    metapath : list[str or tuple of str]
        Metapath in the form of a list of edge types

    Returns
    -------
    DGLHeteroGraph
        A homogeneous or bipartite graph.
    """
    adj = 1
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=True)

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    if srctype == dsttype:
        assert adj.shape[0] == adj.shape[1]
        new_g = convert.graph(adj, ntype=srctype, idtype=g.idtype, device=g.device)
    else:
        new_g = convert.bipartite(adj, utype=srctype, vtype=dsttype,
                                  idtype=g.idtype, device=g.device)

    # copy srcnode features
    for key, value in g.nodes[srctype].data.items():
        new_g.nodes[srctype].data[key] = value
    # copy dstnode features
    if srctype != dsttype:
        for key, value in g.nodes[dsttype].data.items():
            new_g.nodes[dsttype].data[key] = value

    return new_g

def add_nodes(g, num, data=None, ntype=None):
    r"""Add new nodes of the same node type.
    A new graph with newly added nodes is returned.

    Parameters
    ----------
    num : int
        Number of nodes to add.
    data : dict, optional
        Feature data of the added nodes.
    ntype : str, optional
        The type of the new nodes. Can be omitted if there is
        only one node type in the graph.

    Return
    ------
    DGLHeteroGraph
        The graph with newly added nodes.

    Notes
    -----

    * If the key of ``data`` does not contain some existing feature fields,
    those features for the new nodes will be filled with zeros).
    * If the key of ``data`` contains new feature fields, those features for
    the old nodes will be filled zeros).

    Examples
    --------

    The following example uses PyTorch backend.
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_nodes()
    3
    >>> g = dgl.add_nodes(g, 2)
    >>> g.num_nodes()
    5

    If the graph has some node features and new nodes are added without
    features, their features will be created by initializers defined
    with :func:`set_n_initializer`.

    >>> g.ndata['h'] = torch.ones(5, 1)
    >>> g = dgl.add_nodes(g, 1)
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

    We can also assign features for the new nodes in adding new nodes.

    >>> g = dgl.add_nodes(g, 1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

    Since ``data`` contains new feature fields, the features for old nodes
    will be created by initializers defined with :func:`set_n_initializer`.

    >>> g.ndata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])

    **Heterogeneous Graphs with Multiple Node Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    >>>                                 torch.tensor([0, 0, 1, 1])),
    >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    >>>                                         torch.tensor([0, 1]))
    >>>     })
    >>> g = dgl.add_nodes(g, 2)
    DGLError: Node type name must be specified
    if there are more than one node types.
    >>> g.num_nodes('user')
    3
    >>> g = dgl.add_nodes(g, 2, ntype='user')
    >>> g.num_nodes('user')
    5

    See Also
    --------
    remove_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.add_nodes(num, data=data, ntype=ntype)
    return g

def add_edges(g, u, v, data=None, etype=None):
    r"""Add multiple new edges for the specified edge type.
    A new graph with newly added edges is returned.

    The i-th new edge will be from ``u[i]`` to ``v[i]``.

    Parameters
    ----------
    u : int, tensor, numpy.ndarray, list
        Source node IDs, ``u[i]`` gives the source node for the i-th new edge.
    v : int, tensor, numpy.ndarray, list
        Destination node IDs, ``v[i]`` gives the destination node for the i-th new edge.
    data : dict, optional
        Feature data of the added edges. The i-th row of the feature data
        corresponds to the i-th new edge.
    etype : str or tuple of str, optional
        The type of the new edges. Can be omitted if there is
        only one edge type in the graph.

    Return
    ------
    DGLHeteroGraph
        The graph with newly added edges.

    Notes
    -----
    * If end nodes of adding edges does not exists, add_nodes is invoked
    to add new nodes. The node features of the new nodes will be created
    by initializers defined with :func:`set_n_initializer` (default
    initializer fills zeros). In certain cases, it is recommanded to
    add_nodes first and then add_edges.
    * If the key of ``data`` does not contain some existing feature fields,
    those features for the new edges will be created by initializers
    defined with :func:`set_n_initializer` (default initializer fills zeros).
    * If the key of ``data`` contains new feature fields, those features for
    the old edges will be created by initializers defined with
    :func:`set_n_initializer` (default initializer fills zeros).

    Examples
    --------

    The following example uses PyTorch backend.
    >>> import dgl
    >>> import torch
    **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**
    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_edges()
    2
    >>> g = dgl.add_edges(g, torch.tensor([1, 3]), torch.tensor([0, 1]))
    >>> g.num_edges()
    4
    Since ``u`` or ``v`` contains a non-existing node ID, the nodes are
    added implicitly.
    >>> g.num_nodes()
    4

    If the graph has some edge features and new edges are added without
    features, their features will be created by initializers defined
    with :func:`set_n_initializer`.

    >>> g.edata['h'] = torch.ones(4, 1)
    >>> g = dgl.add_edges(g, torch.tensor([1]), torch.tensor([1]))
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.]])

    We can also assign features for the new edges in adding new edges.

    >>> g = dgl.add_edges(g, torch.tensor([0, 0]), torch.tensor([2, 2]),
    >>>                   {'h': torch.tensor([[1.], [2.]]), 'w': torch.ones(2, 1)})
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.], [1.], [2.]])
    Since ``data`` contains new feature fields, the features for old edges
    will be created by initializers defined with :func:`set_n_initializer`.
    >>> g.edata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.]])

    **Heterogeneous Graphs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    >>>                                 torch.tensor([0, 0, 1, 1])),
    >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    >>>                                         torch.tensor([0, 1]))
    >>>     })
    >>> g = dgl.add_edges(g, torch.tensor([3]), torch.tensor([3]))
    DGLError: Edge type name must be specified
    if there are more than one edge types.
    >>> g.number_of_edges('plays')
    4
    >>> g = dgl.add_edges(g, torch.tensor([3]), torch.tensor([3]), etype='plays')
    >>> g.number_of_edges('plays')
    5

    See Also
    --------
    add_nodes
    remove_nodes
    remove_edges
    """
    g = g.clone()
    g.add_edges(u, v, data=data, etype=etype)
    return g

def remove_edges(g, eids, etype=None):
    r"""Remove multiple edges with the specified edge type.
    A new graph with certain edges deleted is returned.

    Nodes will not be removed. After removing edges, the rest
    edges will be re-indexed using consecutive integers from 0,
    with their relative order preserved.
    The features for the removed edges will be removed accordingly.

    Parameters
    ----------
    eids : int, tensor, numpy.ndarray, list
        IDs for the edges to remove.
    etype : str or tuple of str, optional
        The type of the edges to remove. Can be omitted if there is
        only one edge type in the graph.

    Return
    ------
    DGLHeteroGraph
        The graph with edges deleted.

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=3, num_edges=1,
        ndata_schemes={}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edges('all')
    (tensor([2]), tensor([2]), tensor([0]))
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    >>>                                 torch.tensor([0, 0, 1, 1])),
    >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    >>>                                         torch.tensor([0, 1]))
    >>>     })
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]))
    DGLError: Edge type name must be specified
    if there are more than one edge types.
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]), 'plays')
    >>> g.edges('all', etype='plays')
    (tensor([0, 1]), tensor([0, 0]), tensor([0, 1]))
    See Also
    --------
    add_nodes
    add_edges
    remove_nodes
    """
    g = g.clone()
    g.remove_edges(eids, etype=etype)
    return g


def remove_nodes(g, nids, ntype=None):
    r"""Remove multiple nodes with the specified node type.
    A new graph with certain nodes deleted is returned.

    Edges that connect to the nodes will be removed as well. After removing
    nodes and edges, the rest nodes and edges will be re-indexed using
    consecutive integers from 0, with their relative order preserved.
    The features for the removed nodes/edges will be removed accordingly.

    The features for the removed nodes/edges will be removed accordingly.

    Parameters
    ----------
    nids : int, tensor, numpy.ndarray, list
        Nodes to remove.
    ntype : str, optional
        The type of the nodes to remove. Can be omitted if there is
        only one node type in the graph.

    Return
    ------
    DGLHeteroGraph
        The graph with nodes deleted.

    Examples
    --------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=1, num_edges=1,
        ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.ndata['hv']
    tensor([[2.]])
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs with Multiple Node Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    >>>                                 torch.tensor([0, 0, 1, 1])),
    >>>     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    >>>                                         torch.tensor([0, 1]))
    >>>     })
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]))
    DGLError: Node type name must be specified
    if there are more than one node types.
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]), ntype='game')
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    0
    >>> g.num_edges('plays')
    0

    See Also
    --------
    add_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.remove_nodes(nids, ntype=ntype)
    return g

def add_self_loop(g, etype=None):
    r""" Add self loop for each node in the graph.
    A new graph with self-loop is returned.

    Since **selfloop is not well defined for unidirectional
    bipartite graphs**, we simply skip the nodes corresponding
    to unidirectional bipartite graphs.

    Return
    ------
    DGLHeteroGraph
        The graph with self-loop.

    Notes
    -----
    * It is recommanded to ``remove_self_loop`` before invoking
    ``add_self_loop``.
    * Features for the new edges (self-loop edges) will be created
    by initializers defined with :func:`set_n_initializer`
    (default initializer fills zeros).

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([2, 1, 0])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.add_self_loop(g)
    >>> g
    Graph(num_nodes=3, num_edges=6,
            ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
            edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],
            [1.],
            [2.],
            [0.],
            [0.],
            [0.]])

    **Heterogeneous Graphs with Multiple Node Types**

    >>> g = dgl.heterograph({
            ('user', 'follows', 'user'): (torch.tensor([1, 2]),
                                        torch.tensor([0, 1])),
            ('user', 'plays', 'game'): (torch.tensor([0, 1]),
                                        torch.tensor([0, 1]))})
    >>> g = dgl.add_self_loop(g, etype='follows')
    >>> g
    Graph(num_nodes={'user': 3, 'game': 2},
        num_edges={('user', 'plays', 'game'): 2, ('user', 'follows', 'user'): 5},
        metagraph=[('user', 'user'), ('user', 'game')])
    """
    etype = g.to_canonical_etype(etype)
    if etype[0] != etype[2]:
        raise DGLError(
            'add_self_loop does not support unidirectional bipartite graphs: {}.' \
            'Please make sure the types of head node and tail node are identical.' \
            ''.format(etype))
    nodes = g.nodes(etype[0])
    new_g = add_edges(g, nodes, nodes, etype=etype)
    return new_g

DGLHeteroGraph.add_self_loop = add_self_loop

def remove_self_loop(g, etype=None):
    r""" Remove self loops for each node in the graph.
    A new graph with self-loop removed is returned.

    If there are multiple self loops for a certain node,
    all of them will be removed.

    Parameters
    ----------
    etype : str or tuple of str, optional
        The type of the edges to remove. Can be omitted if there is
        only one edge type in the graph.

    Examples
    ---------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

    >>> g = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([1, 0, 0, 2])),
                        idtype=idtype, device=F.ctx())
    >>> g.edata['he'] = torch.arange(4).float().reshape(-1, 1)
    >>> g = dgl.remove_self_loop(g)
    >>> g
    Graph(num_nodes=3, num_edges=2,
        edata_schemes={'he': Scheme(shape=(2,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],[3.]])

    **Heterogeneous Graphs with Multiple Node Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'follows', 'user'): (torch.tensor([0, 1, 1, 1, 2]),
    >>>                                 torch.tensor([0, 0, 1, 1, 1])),
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    >>>                                         torch.tensor([0, 1]))
    >>>     })
    >>> g = dgl.remove_self_loop(g)
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    2
    >>> g.num_edges('follows')
    2
    >>> g.num_edges('plays')
    2

    See Also
    --------
    add_self_loop
    """
    etype = g.to_canonical_etype(etype)
    if etype[0] != etype[2]:
        raise DGLError(
            'remove_self_loop does not support unidirectional bipartite graphs: {}.' \
            'Please make sure the types of head node and tail node are identical.' \
            ''.format(etype))
    u, v = g.edges(form='uv', order='eid', etype=etype)
    self_loop_eids = F.tensor(F.nonzero_1d(u == v), dtype=F.dtype(u))
    new_g = remove_edges(g, self_loop_eids, etype=etype)
    return new_g

DGLHeteroGraph.remove_self_loop = remove_self_loop

def reorder_nodes(g, new_node_ids):
    """ Generate a new graph with new node Ids.

    We assign each node in the input graph with a new node Id. This results in
    a new graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    new_node_ids : a tensor
        The new node Ids
    Returns
    -------
    DGLGraph
        The graph with new node Ids.
    """
    assert len(new_node_ids) == g.number_of_nodes(), \
            "The number of new node ids must match #nodes in the graph."
    new_node_ids = utils.toindex(new_node_ids)
    sorted_ids, idx = F.sort_1d(new_node_ids.tousertensor())
    assert F.asnumpy(sorted_ids[0]) == 0 \
            and F.asnumpy(sorted_ids[-1]) == g.number_of_nodes() - 1, \
            "The new node Ids are incorrect."
    new_gidx = _CAPI_DGLReorderGraph(g._graph, new_node_ids.todgltensor())
    new_g = DGLGraphStale(new_gidx)
    new_g.ndata['orig_id'] = idx
    return new_g

def partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle=False):
    '''Partition a graph.

    Based on the given node assignments for each partition, the function splits
    the input graph into subgraphs. A subgraph may contain HALO nodes which does
    not belong to the partition of a subgraph but are connected to the nodes
    in the partition within a fixed number of hops.

    If `reshuffle` is turned on, the function reshuffles node Ids and edge Ids
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous Id range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node Ids
    in the original input graph.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned
    node_part: 1D tensor
        Specify which partition a node is assigned to. The length of this tensor
        needs to be the same as the number of nodes of the graph. Each element
        indicates the partition Id of a node.
    extra_cached_hops: int
        The number of hops a HALO node can be accessed.
    reshuffle : bool
        Resuffle nodes so that nodes in the same partition are in the same Id range.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition Id and the value is the DGLGraph of the partition.
    '''
    if isinstance(g, DGLHeteroGraph):
        return hetero_partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle)
    assert len(node_part) == g.number_of_nodes()
    node_part = utils.toindex(node_part)
    if reshuffle:
        start = time.time()
        node_part = node_part.tousertensor()
        sorted_part, new2old_map = F.sort_1d(node_part)
        new_node_ids = np.zeros((g.number_of_nodes(),), dtype=np.int64)
        new_node_ids[F.asnumpy(new2old_map)] = np.arange(0, g.number_of_nodes())
        g = reorder_nodes(g, new_node_ids)
        node_part = utils.toindex(sorted_part)
        # We reassign edges in in-CSR. In this way, after partitioning, we can ensure
        # that all edges in a partition are in the contiguous Id space.
        orig_eids = _CAPI_DGLReassignEdges(g._graph, True)
        orig_eids = utils.toindex(orig_eids)
        orig_eids = orig_eids.tousertensor()
        orig_nids = g.ndata['orig_id']
        print('Reshuffle nodes and edges: {:.3f} seconds'.format(time.time() - start))

    start = time.time()
    subgs = _CAPI_DGLPartitionWithHalo(g._graph, node_part.todgltensor(), extra_cached_hops)
    # g is no longer needed. Free memory.
    g = None
    print('Split the graph: {:.3f} seconds'.format(time.time() - start))
    subg_dict = {}
    node_part = node_part.tousertensor()
    start = time.time()

    # This creaets a subgraph from subgraphs returned from the CAPI above.
    def create_subgraph(subg, induced_nodes, induced_edges):
        subg1 = DGLGraphStale(graph_data=subg.graph, readonly=True)
        subg1.ndata[NID] = induced_nodes.tousertensor()
        subg1.edata[EID] = induced_edges.tousertensor()
        return subg1

    for i, subg in enumerate(subgs):
        inner_node = _get_halo_subgraph_inner_node(subg)
        subg = create_subgraph(subg, subg.induced_nodes, subg.induced_edges)
        inner_node = F.zerocopy_from_dlpack(inner_node.to_dlpack())
        subg.ndata['inner_node'] = inner_node
        subg.ndata['part_id'] = F.gather_row(node_part, subg.ndata[NID])
        if reshuffle:
            subg.ndata['orig_id'] = F.gather_row(orig_nids, subg.ndata[NID])
            subg.edata['orig_id'] = F.gather_row(orig_eids, subg.edata[EID])

        if extra_cached_hops >= 1:
            inner_edge = F.zeros((subg.number_of_edges(),), F.int8, F.cpu())
            inner_nids = F.nonzero_1d(subg.ndata['inner_node'])
            # TODO(zhengda) we need to fix utils.toindex() to avoid the dtype cast below.
            inner_nids = F.astype(inner_nids, F.int64)
            inner_eids = subg.in_edges(inner_nids, form='eid')
            inner_edge = F.scatter_row(inner_edge, inner_eids,
                                       F.ones((len(inner_eids),), F.dtype(inner_edge), F.cpu()))
        else:
            inner_edge = F.ones((subg.number_of_edges(),), F.int8, F.cpu())
        subg.edata['inner_edge'] = inner_edge
        subg_dict[i] = subg
    print('Construct subgraphs: {:.3f} seconds'.format(time.time() - start))
    return subg_dict

def metis_partition_assignment(g, k, balance_ntypes=None, balance_edges=False):
    ''' This assigns nodes to different partitions with Metis partitioning algorithm.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * `balance_ntypes` balances the number of nodes of different types in each partition.
    * `balance_edges` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    After the partition assignment, we construct partitions.

    Parameters
    ----------
    g : DGLGraph
        The graph to be partitioned
    k : int
        The number of partitions.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.

    Returns
    -------
    a 1-D tensor
        A vector with each element that indicates the partition Id of a vertex.
    '''
    if isinstance(g, DGLHeteroGraph):
        return hetero_metis_partition_assignment(g, k, balance_ntypes, balance_edges)
    # METIS works only on symmetric graphs.
    # The METIS runs on the symmetric graph to generate the node assignment to partitions.
    start = time.time()
    sym_g = to_bidirected_stale(g, readonly=True)
    print('Convert a graph into a bidirected graph: {:.3f} seconds'.format(time.time() - start))
    vwgt = []
    # To balance the node types in each partition, we can take advantage of the vertex weights
    # in Metis. When vertex weights are provided, Metis will tries to generate partitions with
    # balanced vertex weights. A vertex can be assigned with multiple weights. The vertex weights
    # are stored in a vector of N * w elements, where N is the number of vertices and w
    # is the number of weights per vertex. Metis tries to balance the first weight, and then
    # the second weight, and so on.
    # When balancing node types, we use the first weight to indicate the first node type.
    # if a node belongs to the first node type, its weight is set to 1; otherwise, 0.
    # Similary, we set the second weight for the second node type and so on. The number
    # of weights is the same as the number of node types.
    start = time.time()
    if balance_ntypes is not None:
        assert len(balance_ntypes) == g.number_of_nodes(), \
                "The length of balance_ntypes should be equal to #nodes in the graph"
        balance_ntypes = F.tensor(balance_ntypes)
        uniq_ntypes = F.unique(balance_ntypes)
        for ntype in uniq_ntypes:
            vwgt.append(F.astype(balance_ntypes == ntype, F.int64))

    # When balancing edges in partitions, we use in-degree as one of the weights.
    if balance_edges:
        if balance_ntypes is None:
            vwgt.append(F.astype(g.in_degrees(), F.int64))
        else:
            for ntype in uniq_ntypes:
                nids = F.asnumpy(F.nonzero_1d(balance_ntypes == ntype))
                degs = np.zeros((g.number_of_nodes(),), np.int64)
                degs[nids] = F.asnumpy(g.in_degrees(nids))
                vwgt.append(F.zerocopy_from_numpy(degs))

    # The vertex weights have to be stored in a vector.
    if len(vwgt) > 0:
        vwgt = F.stack(vwgt, 1)
        shape = (np.prod(F.shape(vwgt),),)
        vwgt = F.reshape(vwgt, shape)
        vwgt = F.zerocopy_to_dgl_ndarray(vwgt)
        print('Construct multi-constraint weights: {:.3f} seconds'.format(time.time() - start))
    else:
        vwgt = F.zeros((0,), F.int64, F.cpu())
        vwgt = F.zerocopy_to_dgl_ndarray(vwgt)

    start = time.time()
    node_part = _CAPI_DGLMetisPartition(sym_g._graph, k, vwgt)
    print('Metis partitioning: {:.3f} seconds'.format(time.time() - start))
    if len(node_part) == 0:
        return None
    else:
        node_part = utils.toindex(node_part)
        return node_part.tousertensor()

def metis_partition(g, k, extra_cached_hops=0, reshuffle=False,
                    balance_ntypes=None, balance_edges=False):
    ''' This is to partition a graph with Metis partitioning.

    Metis assigns vertices to partitions. This API constructs subgraphs with the vertices assigned
    to the partitions and their incoming edges. A subgraph may contain HALO nodes which does
    not belong to the partition of a subgraph but are connected to the nodes
    in the partition within a fixed number of hops.

    When performing Metis partitioning, we can put some constraint on the partitioning.
    Current, it supports two constrants to balance the partitioning. By default, Metis
    always tries to balance the number of nodes in each partition.

    * `balance_ntypes` balances the number of nodes of different types in each partition.
    * `balance_edges` balances the number of edges in each partition.

    To balance the node types, a user needs to pass a vector of N elements to indicate
    the type of each node. N is the number of nodes in the input graph.

    If `reshuffle` is turned on, the function reshuffles node Ids and edge Ids
    of the input graph before partitioning. After reshuffling, all nodes and edges
    in a partition fall in a contiguous Id range in the input graph.
    The partitioend subgraphs have node data 'orig_id', which stores the node Ids
    in the original input graph.

    The partitioned subgraph is stored in DGLGraph. The DGLGraph has the `part_id`
    node data that indicates the partition a node belongs to. The subgraphs do not contain
    the node/edge data in the input graph.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned
    k: int
        The number of partitions.
    extra_cached_hops: int
        The number of hops a HALO node can be accessed.
    reshuffle : bool
        Resuffle nodes so that nodes in the same partition are in the same Id range.
    balance_ntypes : tensor
        Node type of each node
    balance_edges : bool
        Indicate whether to balance the edges.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition Id and the value is the DGLGraph of the partition.
    '''
    if isinstance(g, DGLHeteroGraph):
        return hetero_metis_partition(g, k, extra_cached_hops, reshuffle,
                                      balance_ntypes, balance_edges)
    node_part = metis_partition_assignment(g, k, balance_ntypes, balance_edges)
    if node_part is None:
        return None

    # Then we split the original graph into parts based on the METIS partitioning results.
    return partition_graph_with_halo(g, node_part, extra_cached_hops, reshuffle)

def compact_graphs(graphs, always_preserve=None):
    """Given a list of graphs with the same set of nodes, find and eliminate the common
    isolated nodes across all graphs.

    This function requires the graphs to have the same set of nodes (i.e. the node types
    must be the same, and the number of nodes of each node type must be the same).  The
    metagraph does not have to be the same.

    It finds all the nodes that have zero in-degree and zero out-degree in all the given
    graphs, and eliminates them from all the graphs.

    Useful for graph sampling where we have a giant graph but we only wish to perform
    message passing on a smaller graph with a (tiny) subset of nodes.

    The node and edge features are not preserved.

    Parameters
    ----------
    graphs : DGLHeteroGraph or list[DGLHeteroGraph]
        The graph, or list of graphs
    always_preserve : Tensor or dict[str, Tensor], optional
        If a dict of node types and node ID tensors is given, the nodes of given
        node types would not be removed, regardless of whether they are isolated.
        If a Tensor is given, assume that all the graphs have one (same) node type.

    Returns
    -------
    DGLHeteroGraph or list[DGLHeteroGraph]
        The compacted graph or list of compacted graphs.

        Each returned graph would have a feature ``dgl.NID`` containing the mapping
        of node IDs for each type from the compacted graph(s) to the original graph(s).
        Note that the mapping is the same for all the compacted graphs.

    Bugs
    ----
    This function currently requires that the same node type of all graphs should have
    the same node type ID, i.e. the node types are *ordered* the same.

    Examples
    --------
    The following code constructs a bipartite graph with 20 users and 10 games, but
    only user #1 and #3, as well as game #3 and #5, have connections:

    >>> g = dgl.bipartite([(1, 3), (3, 5)], 'user', 'plays', 'game', num_nodes=(20, 10))

    The following would compact the graph above to another bipartite graph with only
    two users and two games.

    >>> new_g, induced_nodes = dgl.compact_graphs(g)
    >>> induced_nodes
    {'user': tensor([1, 3]), 'game': tensor([3, 5])}

    The mapping tells us that only user #1 and #3 as well as game #3 and #5 are kept.
    Furthermore, the first user and second user in the compacted graph maps to
    user #1 and #3 in the original graph.  Games are similar.

    One can verify that the edge connections are kept the same in the compacted graph.

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))

    When compacting multiple graphs, nodes that do not have any connections in any
    of the given graphs are removed.  So if we compact ``g`` and the following ``g2``
    graphs together:

    >>> g2 = dgl.bipartite([(1, 6), (6, 8)], 'user', 'plays', 'game', num_nodes=(20, 10))
    >>> (new_g, new_g2), induced_nodes = dgl.compact_graphs([g, g2])
    >>> induced_nodes
    {'user': tensor([1, 3, 6]), 'game': tensor([3, 5, 6, 8])}

    Then one can see that user #1 from both graphs, users #3 from the first graph, as
    well as user #6 from the second graph, are kept.  Games are similar.

    Similarly, one can also verify the connections:

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> new_g2.edges(form='all', order='eid', etype='plays')
    (tensor([0, 2]), tensor([2, 3]), tensor([0, 1]))
    """
    return_single = False
    if not isinstance(graphs, Iterable):
        graphs = [graphs]
        return_single = True
    if len(graphs) == 0:
        return []
    if graphs[0].is_block:
        raise DGLError('Compacting a block graph is not allowed.')

    # Ensure the node types are ordered the same.
    # TODO(BarclayII): we ideally need to remove this constraint.
    ntypes = graphs[0].ntypes
    idtype = graphs[0].idtype
    device = graphs[0].device
    for g in graphs:
        assert ntypes == g.ntypes, \
            ("All graphs should have the same node types in the same order, got %s and %s" %
             ntypes, g.ntypes)
        assert idtype == g.idtype, "Expect graph data type to be {}, but got {}".format(
            idtype, g.idtype)
        assert device == g.device, "Expect graph device to be {}, but got {}".format(
            device, g.device)

    # Process the dictionary or tensor of "always preserve" nodes
    if always_preserve is None:
        always_preserve = {}
    elif not isinstance(always_preserve, Mapping):
        if len(ntypes) > 1:
            raise ValueError("Node type must be given if multiple node types exist.")
        always_preserve = {ntypes[0]: always_preserve}

    always_preserve = utils.prepare_tensor_dict(graphs[0], always_preserve, 'always_preserve')
    always_preserve_nd = []
    for ntype in ntypes:
        nodes = always_preserve.get(ntype, None)
        if nodes is None:
            nodes = F.copy_to(F.tensor([], idtype), device)
        always_preserve_nd.append(F.to_dgl_nd(nodes))

    # Compact and construct heterographs
    new_graph_indexes, induced_nodes = _CAPI_DGLCompactGraphs(
        [g._graph for g in graphs], always_preserve_nd)
    induced_nodes = [F.from_dgl_nd(nodes) for nodes in induced_nodes]

    new_graphs = [
        DGLHeteroGraph(new_graph_index, graph.ntypes, graph.etypes)
        for new_graph_index, graph in zip(new_graph_indexes, graphs)]
    for g in new_graphs:
        for i, ntype in enumerate(graphs[0].ntypes):
            g.nodes[ntype].data[NID] = induced_nodes[i]
    if return_single:
        new_graphs = new_graphs[0]

    return new_graphs

def to_block(g, dst_nodes=None, include_dst_in_src=True, copy_ndata=True, copy_edata=True):
    """Convert a graph into a bipartite-structured "block" for message passing.

    A block graph is uni-directional bipartite graph consisting of two sets of nodes
    SRC and DST. Each set can have many node types while all the edges are from SRC
    nodes to DST nodes.

    Specifically, for each relation graph of canonical edge type ``(utype, etype, vtype)``,
    node type ``utype`` belongs to SRC while ``vtype`` belongs to DST.
    Edges from node type ``utype`` to node type ``vtype`` are preserved. If
    ``utype == vtype``, the result graph will have two node types of the same name ``utype``,
    but one belongs to SRC while the other belongs to DST. This is because although
    they have the same name, their node ids are relabeled differently (see below). In
    both cases, the canonical edge type in the new graph is still
    ``(utype, etype, vtype)``, so there is no difference when referring to it.

    Moreover, the function also relabels node ids in each type to make the graph more compact.
    Specifically, the nodes of type ``vtype`` would contain the nodes that have at least one
    inbound edge of any type, while ``utype`` would contain all the DST nodes of type ``vtype``,
    as well as the nodes that have at least one outbound edge to any DST node.

    Since DST nodes are included in SRC nodes, a common requirement is to fetch
    the DST node features from the SRC nodes features. To avoid expensive sparse lookup,
    the function assures that the DST nodes in both SRC and DST sets have the same ids.
    As a result, given the node feature tensor ``X`` of type ``utype``,
    the following code finds the corresponding DST node features of type ``vtype``:

    .. code::

        X[:block.number_of_nodes('DST/vtype')]

    If the ``dst_nodes`` argument is given, the DST nodes would contain the given nodes.
    Otherwise, the DST nodes would be determined by DGL via the rules above.

    Parameters
    ----------
    graph : DGLHeteroGraph
        The graph.
    dst_nodes : Tensor or dict[str, Tensor], optional
        Optional DST nodes. If a tensor is given, the graph must have only one node type.
    include_dst_in_src : bool
        If False, do not include DST nodes in SRC nodes.
        (Default: True)
    copy_ndata : bool, optional
        If True, the source and destination node features of the block are copied from the
        original graph.
        If False, the block will not have any node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the block are copied from the origianl graph.
        If False, the simple graph will not have any edge features.
        (Default: True)

    Returns
    -------
    DGLHeteroGraph
        The new graph describing the block.

        The node IDs induced for each type in both sides would be stored in feature
        ``dgl.NID``.

        The edge IDs induced for each type would be stored in feature ``dgl.EID``.

    Notes
    -----
    This function is primarily for creating the structures for efficient
    computation of message passing.  See [TODO] for a detailed example.

    Examples
    --------
    Converting a homogeneous graph to a block as described above:
    >>> g = dgl.graph([(0, 1), (1, 2), (2, 3)])
    >>> block = dgl.to_block(g, torch.LongTensor([3, 2]))

    The right hand side nodes would be exactly the same as the ones given: [3, 2].
    >>> induced_dst = block.dstdata[dgl.NID]
    >>> induced_dst
    tensor([3, 2])

    The first few nodes of the left hand side nodes would also be exactly the same as
    the ones given.  The rest of the nodes are the ones necessary for message passing
    into nodes 3, 2.  This means that the node 1 would be included.
    >>> induced_src = block.srcdata[dgl.NID]
    >>> induced_src
    tensor([3, 2, 1])

    We can notice that the first two nodes are identical to the given nodes as well as
    the right hand side nodes.

    The induced edges can also be obtained by the following:
    >>> block.edata[dgl.EID]
    tensor([2, 1])

    This indicates that edge (2, 3) and (1, 2) are included in the result graph.  We can
    verify that the first edge in the block indeed maps to the edge (2, 3), and the
    second edge in the block indeed maps to the edge (1, 2):
    >>> src, dst = block.edges(order='eid')
    >>> induced_src[src], induced_dst[dst]
    (tensor([2, 1]), tensor([3, 2]))

    Converting a heterogeneous graph to a block is similar, except that when specifying
    the right hand side nodes, you have to give a dict:
    >>> g = dgl.bipartite([(0, 1), (1, 2), (2, 3)], utype='A', vtype='B')

    If you don't specify any node of type A on the right hand side, the node type ``A``
    in the block would have zero nodes on the DST side.
    >>> block = dgl.to_block(g, {'B': torch.LongTensor([3, 2])})
    >>> block.number_of_dst_nodes('A')
    0
    >>> block.number_of_dst_nodes('B')
    2
    >>> block.dstnodes['B'].data[dgl.NID]
    tensor([3, 2])

    The left hand side would contain all the nodes on the right hand side:
    >>> block.srcnodes['B'].data[dgl.NID]
    tensor([3, 2])

    As well as all the nodes that have connections to the nodes on the right hand side:
    >>> block.srcnodes['A'].data[dgl.NID]
    tensor([2, 1])
    """
    if dst_nodes is None:
        # Find all nodes that appeared as destinations
        dst_nodes = defaultdict(list)
        for etype in g.canonical_etypes:
            _, dst = g.edges(etype=etype)
            dst_nodes[etype[2]].append(dst)
        dst_nodes = {ntype: F.unique(F.cat(values, 0)) for ntype, values in dst_nodes.items()}
    elif not isinstance(dst_nodes, Mapping):
        # dst_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise ValueError(
                'Graph has more than one node type; please specify a dict for dst_nodes.')
        dst_nodes = {g.ntypes[0]: dst_nodes}
    dst_nodes = {
        ntype: utils.toindex(nodes, g._idtype_str).tousertensor()
        for ntype, nodes in dst_nodes.items()}

    # dst_nodes is now a dict
    dst_nodes_nd = []
    for ntype in g.ntypes:
        nodes = dst_nodes.get(ntype, None)
        if nodes is not None:
            dst_nodes_nd.append(F.zerocopy_to_dgl_ndarray(nodes))
        else:
            dst_nodes_nd.append(nd.NULL[g._idtype_str])

    new_graph_index, src_nodes_nd, induced_edges_nd = _CAPI_DGLToBlock(
        g._graph, dst_nodes_nd, include_dst_in_src)

    # The new graph duplicates the original node types to SRC and DST sets.
    new_ntypes = (g.ntypes, g.ntypes)
    new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
    assert new_graph.is_unibipartite  # sanity check

    src_node_id = {
        ntype: F.zerocopy_from_dgl_ndarray(src)
        for ntype, src in zip(g.ntypes, src_nodes_nd)}
    dst_node_id = {
        ntype: dst_nodes.get(ntype, F.tensor([], dtype=g.idtype))
        for ntype in g.ntypes}
    edge_id = {
        canonical_etype: F.zerocopy_from_dgl_ndarray(edges)
        for canonical_etype, edges in zip(g.canonical_etypes, induced_edges_nd)}

    if copy_ndata:
        for ntype in g.ntypes:
            src = src_node_id[ntype]
            dst = dst_node_id[ntype]
            for key, value in g.nodes[ntype].data.items():
                if is_internal_column(key):
                    continue
                ctx = F.context(value)
                new_graph.srcnodes[ntype].data[key] = F.gather_row(value, F.copy_to(src, ctx))
                new_graph.dstnodes[ntype].data[key] = F.gather_row(value, F.copy_to(dst, ctx))
    if copy_edata:
        for canonical_etype in g.canonical_etypes:
            eid = edge_id[canonical_etype]
            for key, value in g.edges[canonical_etype].data.items():
                if is_internal_column(key):
                    continue
                ctx = F.context(value)
                new_graph.edges[canonical_etype].data[key] = F.gather_row(
                    value, F.copy_to(eid, ctx))

    for i, ntype in enumerate(g.ntypes):
        new_graph.srcnodes[ntype].data[NID] = F.zerocopy_from_dgl_ndarray(src_nodes_nd[i])
        if ntype in dst_nodes:
            new_graph.dstnodes[ntype].data[NID] = dst_nodes[ntype]
        else:
            # For empty dst node sets, still create empty mapping arrays.
            new_graph.dstnodes[ntype].data[NID] = F.tensor([], dtype=g.idtype)

    for i, canonical_etype in enumerate(g.canonical_etypes):
        induced_edges = F.zerocopy_from_dgl_ndarray(induced_edges_nd[i])
        new_graph.edges[canonical_etype].data[EID] = induced_edges

    return new_graph

def in_subgraph(g, nodes):
    """Extract the subgraph containing only the in edges of the given nodes.

    The subgraph keeps the same type schema and the cardinality of the original one.
    Node/edge features are not preserved. The original IDs
    the extracted edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.

    Returns
    -------
    DGLHeteroGraph
        The subgraph.
    """
    if g.is_block:
        raise DGLError('Extracting subgraph of a block graph is not allowed.')
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.NULL[g._idtype_str])

    subgidx = _CAPI_DGLInSubgraph(g._graph, nodes_all_types)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i]
    return ret

def out_subgraph(g, nodes):
    """Extract the subgraph containing only the out edges of the given nodes.

    The subgraph keeps the same type schema and the cardinality of the original one.
    Node/edge features are not preserved. The original IDs
    the extracted edges are stored as the `dgl.EID` feature in the returned graph.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.

    Returns
    -------
    DGLHeteroGraph
        The subgraph.
    """
    if g.is_block:
        raise DGLError('Extracting subgraph of a block graph is not allowed.')
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes = utils.prepare_tensor_dict(g, nodes, 'nodes')
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(F.to_dgl_nd(nodes[ntype]))
        else:
            nodes_all_types.append(nd.NULL[g._idtype_str])

    subgidx = _CAPI_DGLOutSubgraph(g._graph, nodes_all_types)
    induced_edges = subgidx.induced_edges
    ret = DGLHeteroGraph(subgidx.graph, g.ntypes, g.etypes)
    for i, etype in enumerate(ret.canonical_etypes):
        ret.edges[etype].data[EID] = induced_edges[i]
    return ret

def to_simple(g, return_counts='count', writeback_mapping=False, copy_ndata=True, copy_edata=False):
    r"""Convert a graph to a simple graph without duplicate edges.

    For a heterograph with multiple edge types, we
    treat edges corresponding
    to each type as a separate graph and convert each
    of them to a simple graph.

    When writeback_mapping=True, an extra mapping is returned.
    For the edges in the original graph,
    a writeback mapping is a tensor recording their new
    ids in the simple graph. If the graph has
    only one edge type, a single tensor is returned.
    If the graph has multiple edge types, a dictionary
    of tensor is returned using canonical edge types
    as the key.

    Given a :class:`dgl.DGLGraph` object, we return
    another :class:`dgl.DGLGraph` object representing the
    simple graph corresponding to it.


    Parameters
    ----------
    g : DGLGraph
        The input graph.
    return_counts : str, optional
        If given, the count of each edge in the original graph
        will be stored as edge features under the name
        eturn_counts.
        (Default: "count")
    writeback_mapping: bool, optional
        If True, a write back mapping is returned for each edge
        type subgraph. If False, only the simple graph is returned.
        (Default: False)
    copy_ndata: bool, optional
        If True, the node features of the simple graph are copied
        from the original graph. If False, the simple
        graph will not have any node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the simple graph are copied
        from the original graph. If there exists duplicate edges between
        two nodes (u, v), the feature of the edge is randomly selected
        from one of the duplicate edges.
        If False, the simple graph will not have any edge features.
        (Default: False)

    Returns
    -------
    DGLGraph
        A simple graph.
    tensor or dict of tensor
        If writeback_mapping is True, the writeback
        mapping is returned. If the graph has only
        one edge type, a tensor is returned. If the
        graph has multiple edge types, a dictionary
        of tensor is return.

    If ``copy_ndata`` is ``True``, same tensors will be used for
    the features of the original graph and the to_simpled graph. As a result, users
    should avoid performing in-place operations on the features of the to_simpled
    graph, which will corrupt the features of the original graph as well. For
    concrete examples, refer to the ``Examples`` section below.

    Examples
    --------
    **Homographs or Heterographs with A Single Edge Type**

    Create a graph for demonstrating to_simple API.
    In the original graph, there are multiple edges between 1 and 2.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2, 1]), th.tensor([1, 2, 0, 2])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.], [6.]])

    Convert the graph to a simple graph. The return counts is
    stored in the edge feature 'cnt' and the writeback mapping
    is returned in a tensor.

    >>> sg, wm = dgl.to_simple(g, return_counts='cnt', writeback_mapping=True)
    >>> sg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])
    >>> u, v, eid = sg.edges(form='all')
    >>> u
    tensor([0, 1, 2])
    >>> v
    tensor([1, 2, 0])
    >>> eid
    tensor([0, 1, 2])
    >>> sg.edata['cnt']
    tensor([1, 2, 1])
    >>> wm
    tensor([0, 1, 2, 1])
    >>> 'h' in g.edata
    False

    **In-place operations on features of one graph will be reflected on features of
    the simple graph, which is dangerous. Out-place operations will not be reflected.**

    >>> sg.ndata['h'] += 1
    >>> g.ndata['h']
    tensor([[1.],
            [2.],
            [3.]])
    >>> g.ndata['h'] += 1
    >>> sg.ndata['h']
    tensor([[2.],
            [3.],
            [4.]])
    >>> sg.ndata['h2'] = th.ones(3, 1)
    >>> 'h2' in g.ndata
    False

    **Heterographs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    >>>     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    >>>     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The to_simple operation is applied to the subgraph
    corresponding to ('user', 'wins', 'user') and the
    subgraph corresponding to ('user', 'plays', 'game').
    The return counts is stored in the default edge feature
    'count'.

    >>> sg, wm = dgl.to_simple(g, copy_ndata=False, writeback_mapping=True)
    >>> sg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'wins', 'user'): 4, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> sg.edges(etype='wins')
    (tensor([0, 2, 0, 2]), tensor([1, 1, 2, 0]))
    >>> wm[('user', 'wins', 'user')]
    tensor([0, 1, 2, 1, 3])
    >>> sg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> wm[('user', 'plays', 'game')]
    tensor([0, 1, 2])
    >>> 'hv' in sg.nodes['game'].data
    False
    >>> 'he' in sg.edges['plays'].data
    False
    >>> sg.edata['count']
    {('user', 'wins', 'user'): tensor([1, 2, 1, 1])
     ('user', 'plays', 'game'): tensor([1, 1, 1])}
    """
    if g.is_block:
        raise DGLError('Cannot convert a block graph to a simple graph.')
    simple_graph_index, counts, edge_maps = _CAPI_DGLToSimpleHetero(g._graph)
    simple_graph = DGLHeteroGraph(simple_graph_index, g.ntypes, g.etypes)
    counts = [F.zerocopy_from_dgl_ndarray(count) for count in counts]
    edge_maps = [F.zerocopy_from_dgl_ndarray(edge_map) for edge_map in edge_maps]

    if return_counts is not None:
        for count, canonical_etype in zip(counts, g.canonical_etypes):
            simple_graph.edges[canonical_etype].data[return_counts] = count

    if copy_ndata:
        for ntype in g.ntypes:
            for key in g.nodes[ntype].data:
                simple_graph.nodes[ntype].data[key] = g.nodes[ntype].data[key]

    if copy_edata:
        for i, c_etype in enumerate(g.canonical_etypes):
            for key in g.edges[c_etype].data:
                feat_idx = F.asnumpy(edge_maps[i])
                _, indices = np.unique(feat_idx, return_index=True)
                simple_graph.edges[c_etype].data[key] = \
                    F.gather_row(g.edges[c_etype].data[key],
                                 F.copy_to(F.tensor(indices),
                                           F.context(g.edges[c_etype].data[key])))

    if writeback_mapping:
        # single edge type
        if len(edge_maps) == 1:
            return simple_graph, edge_maps[0]
        # multiple edge type
        else:
            wb_map = {}
            for edge_map, canonical_etype in zip(edge_maps, g.canonical_etypes):
                wb_map[canonical_etype] = edge_map
            return simple_graph, wb_map

    return simple_graph

DGLHeteroGraph.to_simple = to_simple

def as_heterograph(g, ntype='_U', etype='_E'):  # pylint: disable=unused-argument
    """Convert a DGLGraph to a DGLHeteroGraph with one node and edge type.

    DEPRECATED: DGLGraph and DGLHeteroGraph have been merged. This function will
                do nothing and can be removed safely in all cases.
    """
    dgl_warning('DEPRECATED: DGLGraph and DGLHeteroGraph have been merged in v0.5.\n'
                '\tdgl.as_heterograph will do nothing and can be removed safely in all cases.')
    return g

def as_immutable_graph(hg):
    """Convert a DGLHeteroGraph with one node and edge type into a DGLGraph.

    DEPRECATED: DGLGraph and DGLHeteroGraph have been merged. This function will
                do nothing and can be removed safely in all cases.
    """
    dgl_warning('DEPRECATED: DGLGraph and DGLHeteroGraph have been merged in v0.5.\n'
                '\tdgl.as_immutable_graph will do nothing and can be removed safely in all cases.')
    return hg

_init_api("dgl.transform")
