"""Module for graph transformation methods."""
from ._ffi.function import _init_api
from .graph import DGLGraph
from . import backend as F
from .batched_graph import BatchedDGLGraph
from scipy import sparse
import numpy as np

__all__ = ['line_graph', 'reverse', 'nearest_neighbor_graph', 'segmented_nearest_neighbor_graph',
           'to_simple_graph', 'to_bidirected']


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)

def nearest_neighbor_graph(h, k):
    """Transforms the given point coordinate matrix to a directed graph, where
    the predecessors of each point are its k-nearest neighbors.

    Parameters
    ----------
    h : Tensor
        The input tensor.

        If 2D, each row of ``h`` corresponds to a node.

        If 3D, a k-NN graph would be constructed for each row.  Then
        the graphs are unioned.
    k : int
        The number of neighbors

    Returns
    -------
    DGLGraph
        The graph.  The node IDs are in the same order as ``h``.
    """
    if F.ndim(h) == 2:
        h = F.unsqueeze(h, 0)
    n_samples, n_points, n_dims = F.shape(h)

    d = pairwise_squared_distance(h)
    k_indices = F.argtopk(d, k, 2, descending=False)
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

def segmented_nearest_neighbor_graph(h, k, segs):
    n_total_points, n_dims = F.shape(h)
    offset = np.insert(np.cumsum(segs), 0, 0)

    hs = F.split(h, segs, 0)
    dst = [
        F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False) +
        offset[i]
        for i, h_g in enumerate(hs)]
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
    g_edges = g.edges()
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

_init_api("dgl.transform")
