"""Module for graph transformation methods."""
from ._ffi.function import _init_api
from .graph import DGLGraph
from .batched_graph import BatchedDGLGraph

__all__ = ['line_graph', 'reverse', 'to_simple_graph', 'to_bidirected']


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
