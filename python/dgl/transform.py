"""Module for graph transformation utilities."""

import numpy as np
from scipy import sparse
from ._ffi.function import _init_api
from .graph import DGLGraph
from .subgraph import DGLSubGraph
from . import backend as F
from .graph_index import from_coo
from .graph_index import _get_halo_subgraph_inner_node
from .graph_index import _get_halo_subgraph_inner_edge
from .batched_graph import BatchedDGLGraph, unbatch
from .convert import graph, bipartite
from . import utils


__all__ = ['line_graph', 'khop_adj', 'khop_graph', 'reverse', 'to_simple_graph', 'to_bidirected',
           'laplacian_lambda_max', 'knn_graph', 'segmented_knn_graph', 'add_self_loop',
           'remove_self_loop', 'metapath_reachable_graph']


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
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    g = DGLGraph(adj, readonly=True)
    return g

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
        offset[i]
        for i, h_g in enumerate(h_list)]
    dst = F.cat(dst, 0)
    src = F.arange(0, n_total_points).unsqueeze(1).expand(n_total_points, k)

    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    g = DGLGraph(adj, readonly=True)
    return g

def line_graph(g, backtracking=True, shared=False):
    """Return the line graph of this graph.

    Parameters
    ----------
    g : dgl.DGLGraph
        The input graph.
    backtracking : bool, optional
        Whether the returned line graph is backtracking.
    shared : bool, optional
        Whether the returned line graph shares representations with `self`.

    Returns
    -------
    DGLGraph
        The line graph of this graph.
    """
    graph_data = g._graph.line_graph(backtracking)
    node_frame = g._edge_frame if shared else None
    return DGLGraph(graph_data, node_frame)

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
    adj_k = g.adjacency_matrix_scipy(return_edge_ids=False) ** k
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
    adj_k = g.adjacency_matrix_scipy(return_edge_ids=False) ** k
    adj_k = adj_k.tocoo()
    multiplicity = adj_k.data
    row = np.repeat(adj_k.row, multiplicity)
    col = np.repeat(adj_k.col, multiplicity)
    # TODO(zihao): we should support creating multi-graph from scipy sparse matrix
    # in the future.
    return DGLGraph(from_coo(n, row, col, True, True))

def reverse(g, share_ndata=False, share_edata=False):
    """Return the reverse of a graph

    The reverse (also called converse, transpose) of a directed graph is another directed
    graph on the same nodes with edges reversed in terms of direction.

    Given a :class:`DGLGraph` object, we return another :class:`DGLGraph` object
    representing its reverse.

    Notes
    -----
    * This function does not support :class:`~dgl.BatchedDGLGraph` objects.
    * We do not dynamically update the topology of a graph once that of its reverse changes.
      This can be particularly problematic when the node/edge attrs are shared. For example,
      if the topology of both the original graph and its reverse get changed independently,
      you can get a mismatched node/edge feature.

    Parameters
    ----------
    g : dgl.DGLGraph
        The input graph.
    share_ndata: bool, optional
        If True, the original graph and the reversed graph share memory for node attributes.
        Otherwise the reversed graph will not be initialized with node attributes.
    share_edata: bool, optional
        If True, the original graph and the reversed graph share memory for edge attributes.
        Otherwise the reversed graph will not have edge attributes.

    Examples
    --------
    Create a graph to reverse.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(3)
    >>> g.add_edges([0, 1, 2], [1, 2, 0])
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.]])

    Reverse the graph and examine its structure.

    >>> rg = g.reverse(share_ndata=True, share_edata=True)
    >>> print(rg)
    DGLGraph with 3 nodes and 3 edges.
    Node data: {'h': Scheme(shape=(1,), dtype=torch.float32)}
    Edge data: {'h': Scheme(shape=(1,), dtype=torch.float32)}

    The edges are reversed now.

    >>> rg.has_edges_between([1, 2, 0], [0, 1, 2])
    tensor([1, 1, 1])

    Reversed edges have the same feature as the original ones.

    >>> g.edges[[0, 2], [1, 0]].data['h'] == rg.edges[[1, 0], [0, 2]].data['h']
    tensor([[1],
            [1]], dtype=torch.uint8)

    The node/edge features of the reversed graph share memory with the original
    graph, which is helpful for both forward computation and back propagation.

    >>> g.ndata['h'] = g.ndata['h'] + 1
    >>> rg.ndata['h']
    tensor([[1.],
            [2.],
            [3.]])
    """
    assert not isinstance(g, BatchedDGLGraph), \
        'reverse is not supported for a BatchedDGLGraph object'
    g_reversed = DGLGraph(multigraph=g.is_multigraph)
    g_reversed.add_nodes(g.number_of_nodes())
    g_edges = g.all_edges(order='eid')
    g_reversed.add_edges(g_edges[1], g_edges[0])
    if share_ndata:
        g_reversed._node_frame = g._node_frame
    if share_edata:
        g_reversed._edge_frame = g._edge_frame
    return g_reversed

def to_simple_graph(g):
    """Convert the graph to a simple graph with no multi-edge.

    The function generates a new *readonly* graph with no node/edge feature.

    Parameters
    ----------
    g : DGLGraph
        The input graph.

    Returns
    -------
    DGLGraph
        A simple graph.
    """
    gidx = _CAPI_DGLToSimpleGraph(g._graph)
    return DGLGraph(gidx, readonly=True)

def to_bidirected(g, readonly=True):
    """Convert the graph to a bidirected graph.

    The function generates a new graph with no node/edge feature.
    If g has m edges for i->j and n edges for j->i, then the
    returned graph will have max(m, n) edges for both i->j and j->i.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Whether the returned bidirected graph is readonly or not.

    Returns
    -------
    DGLGraph

    Examples
    --------
    The following two examples use PyTorch backend, one for non-multi graph
    and one for multi-graph.

    >>> # non-multi graph
    >>> g = dgl.DGLGraph()
    >>> g.add_nodes(2)
    >>> g.add_edges([0, 0], [0, 1])
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 0]), tensor([0, 0, 1]))

    >>> # multi-graph
    >>> g.add_edges([0, 1], [1, 0])
    >>> g.edges()
    (tensor([0, 0, 0, 1]), tensor([0, 1, 1, 0]))

    >>> bg2 = dgl.to_bidirected(g)
    >>> bg2.edges()
    (tensor([0, 1, 1, 0, 0]), tensor([0, 0, 0, 1, 1]))
    """
    if readonly:
        newgidx = _CAPI_DGLToBidirectedImmutableGraph(g._graph)
    else:
        newgidx = _CAPI_DGLToBidirectedMutableGraph(g._graph)
    return DGLGraph(newgidx)

def laplacian_lambda_max(g):
    """Return the largest eigenvalue of the normalized symmetric laplacian of g.

    The eigenvalue of the normalized symmetric of any graph is less than or equal to 2,
    ref: https://en.wikipedia.org/wiki/Laplacian_matrix#Properties

    Parameters
    ----------
    g : DGLGraph or BatchedDGLGraph
        The input graph, it should be an undirected graph.

    Returns
    -------
    list :
        * If the input g is a DGLGraph, the returned value would be
          a list with one element, indicating the largest eigenvalue of g.
        * If the input g is a BatchedDGLGraph, the returned value would
          be a list, where the i-th item indicates the largest eigenvalue
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
    if isinstance(g, BatchedDGLGraph):
        g_arr = unbatch(g)
    else:
        g_arr = [g]

    rst = []
    for g_i in g_arr:
        n = g_i.number_of_nodes()
        adj = g_i.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
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
        new_g = graph(adj, ntype=srctype)
    else:
        new_g = bipartite(adj, utype=srctype, vtype=dsttype)

    for key, value in g.nodes[srctype].data.items():
        new_g.nodes[srctype].data[key] = value
    if srctype != dsttype:
        for key, value in g.nodes[dsttype].data.items():
            new_g.nodes[dsttype].data[key] = value

    return new_g

def add_self_loop(g):
    """Return a new graph containing all the edges in the input graph plus self loops
    of every nodes.
    No duplicate self loop will be added for nodes already having self loops.
    Self-loop edges id are not preserved. All self-loop edges would be added at the end.

    Examples
    ---------

    >>> g = DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0, 1, 2], [1, 1, 2])
    >>> new_g = dgl.transform.add_self_loop(g) # Nodes 0, 3, 4 don't have self-loop
    >>> new_g.edges()
    (tensor([0, 0, 1, 2, 3, 4]), tensor([1, 0, 1, 2, 3, 4]))

    Parameters
    ------------
    g: DGLGraph

    Returns
    --------
    DGLGraph
    """
    new_g = DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    src, dst = g.all_edges(order="eid")
    src = F.zerocopy_to_numpy(src)
    dst = F.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)
    return new_g

def remove_self_loop(g):
    """Return a new graph with all self-loop edges removed

    Examples
    ---------

    >>> g = DGLGraph()
    >>> g.add_nodes(5)
    >>> g.add_edges([0, 1, 2], [1, 1, 2])
    >>> new_g = dgl.transform.remove_self_loop(g) # Nodes 1, 2 have self-loop
    >>> new_g.edges()
    (tensor([0]), tensor([1]))

    Parameters
    ------------
    g: DGLGraph

    Returns
    --------
    DGLGraph
    """
    new_g = DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    src, dst = g.all_edges(order="eid")
    src = F.zerocopy_to_numpy(src)
    dst = F.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    return new_g

def partition_graph_with_halo(g, node_part, num_hops):
    '''
    This is to partition a graph. Each partition contains HALO nodes
    so that we can generate NodeFlow in each partition correctly.

    Parameters
    ------------
    g: DGLGraph
        The graph to be partitioned

    node_part: 1D tensor
        Specify which partition a node is assigned to. The length of this tensor
        needs to be the same as the number of nodes of the graph. Each element
        indicates the partition Id of a node.

    num_hops: int
        The number of hops a HALO node can be accessed.

    Returns
    --------
    a dict of DGLGraphs
        The key is the partition Id and the value is the DGLGraph of the partition.
    '''
    assert len(node_part) == g.number_of_nodes()
    node_part = utils.toindex(node_part)
    subgs = _CAPI_DGLPartitionWithHalo(g._graph, node_part.todgltensor(), num_hops)
    subg_dict = {}
    for i, subg in enumerate(subgs):
        inner_node = _get_halo_subgraph_inner_node(subg)
        inner_edge = _get_halo_subgraph_inner_edge(subg)
        subg = DGLSubGraph(g, subg)
        inner_node = F.zerocopy_from_dlpack(inner_node.to_dlpack())
        subg.ndata['inner_node'] = inner_node
        inner_edge = F.zerocopy_from_dlpack(inner_edge.to_dlpack())
        subg.edata['inner_edge'] = inner_edge
        subg_dict[i] = subg
    return subg_dict

_init_api("dgl.transform")
