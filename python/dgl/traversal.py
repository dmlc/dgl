"""Module for graph traversal methods."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from . import backend as F
from . import utils

__all__ = ['bfs_nodes_generator', 'bfs_edges_generator',
           'topological_nodes_generator',
           'dfs_edges_generator', 'dfs_labeled_edges_generator',]

def bfs_nodes_generator(graph, source, reversed=False):
    """Node frontiers generator using breadth-first search.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    source : list, tensor of nodes
        Source nodes.
    reversed : bool, default False
        If True, traverse following the in-edge direction.

    Returns
    -------
    list of node frontiers
        Each node frontier is a list or tensor of node ids.

    Examples
    --------
    Given a graph (directed, edges from small node id to large):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5

    >>> g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])
    >>> list(dgl.bfs_nodes_generator(g, 0))
    [tensor([0]), tensor([1]), tensor([2, 3]), tensor([4, 5])]
    """
    ghandle = graph._graph._handle
    source = utils.toindex(source).todgltensor()
    ret = _CAPI_DGLBFSNodes(ghandle, source, reversed)
    all_nodes = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tonumpy().tolist()
    node_frontiers = F.split(all_nodes, sections, dim=0)
    return node_frontiers

def bfs_edges_generator(graph, source, reversed=False):
    """Edges frontiers generator using breadth-first search.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    source : list, tensor of nodes
        Source nodes.
    reversed : bool, default False
        If True, traverse following the in-edge direction.

    Returns
    -------
    list of edge frontiers
        Each edge frontier is a list or tensor of edge ids.

    Examples
    --------
    Given a graph (directed, edges from small node id to large, sorted
    in lexicographical order of source-destination node id tuple):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5

    >>> g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])
    >>> list(dgl.bfs_edges_generator(g, 0))
    [tensor([0]), tensor([1, 2]), tensor([4, 5])]
    """
    ghandle = graph._graph._handle
    source = utils.toindex(source).todgltensor()
    ret = _CAPI_DGLBFSEdges(ghandle, source, reversed)
    all_edges = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tonumpy().tolist()
    edge_frontiers = F.split(all_edges, sections, dim=0)
    return edge_frontiers

def topological_nodes_generator(graph, reversed=False):
    """Node frontiers generator using topological traversal.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    reversed : bool, optional
        If True, traverse following the in-edge direction.

    Returns
    -------
    list of node frontiers
        Each node frontier is a list or tensor of node ids.

    Examples
    --------
    Given a graph (directed, edges from small node id to large):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5

    >>> g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])
    >>> list(dgl.topological_nodes_generator(g))
    [tensor([0]), tensor([1]), tensor([2]), tensor([3, 4]), tensor([5])]
    """
    ghandle = graph._graph._handle
    ret = _CAPI_DGLTopologicalNodes(ghandle, reversed)
    all_nodes = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tonumpy().tolist()
    return F.split(all_nodes, sections, dim=0)

def dfs_edges_generator(graph, source, reversed=False):
    """Edge frontiers generator using depth-first-search (DFS).

    Multiple source nodes can be specified to start the DFS traversal. One
    needs to make sure that each source node belongs to different connected
    component, so the frontiers can be easily merged. Otherwise, the behavior
    is undefined.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    source : list, tensor of nodes
        Source nodes.
    reversed : bool, optional
        If True, traverse following the in-edge direction.

    Returns
    -------
    list of edge frontiers
        Each edge frontier is a list or tensor of edge ids.

    Examples
    --------
    Given a graph (directed, edges from small node id to large):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5

    Edge addition order [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)]

    >>> g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])
    >>> list(dgl.dfs_edges_generator(g, 0))
    [tensor([0]), tensor([1]), tensor([3]), tensor([5]), tensor([4])]
    """
    ghandle = graph._graph._handle
    source = utils.toindex(source).todgltensor()
    ret = _CAPI_DGLDFSEdges(ghandle, source, reversed)
    all_edges = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tonumpy().tolist()
    return F.split(all_edges, sections, dim=0)

def dfs_labeled_edges_generator(
        graph,
        source,
        reversed=False,
        has_reverse_edge=False,
        has_nontree_edge=False,
        return_labels=True):
    """Produce edges in a depth-first-search (DFS) labeled by type.

    There are three labels: FORWARD(0), REVERSE(1), NONTREE(2)

    A FORWARD edge is one in which `u` has been visised but `v` has not. A
    REVERSE edge is one in which both `u` and `v` have been visited and the
    edge is in the DFS tree. A NONTREE edge is one in which both `u` and `v`
    have been visited but the edge is NOT in the DFS tree.

    See ``networkx``'s :func:`dfs_labeled_edges
    <networkx.algorithms.traversal.depth_first_search.dfs_labeled_edges>`
    for more details.

    Multiple source nodes can be specified to start the DFS traversal. One
    needs to make sure that each source node belongs to different connected
    component, so the frontiers can be easily merged. Otherwise, the behavior
    is undefined.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    source : list, tensor of nodes
        Source nodes.
    reversed : bool, optional
        If true, traverse following the in-edge direction.
    has_reverse_edge : bool, optional
        True to include reverse edges.
    has_nontree_edge : bool, optional
        True to include nontree edges.
    return_labels : bool, optional
        True to return the labels of each edge.

    Returns
    -------
    list of edge frontiers
        Each edge frontier is a list or tensor of edge ids.
    list of list of int
        Label of each edge, organized in the same order as the edge frontiers.

    Examples
    --------
    Given a graph (directed, edges from small node id to large):
    ::

              2 - 4
             / \\
        0 - 1 - 3 - 5

    Edge addition order [(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)]

    >>> g = dgl.DGLGraph([(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 5)])
    >>> list(dgl.dfs_labeled_edges_generator(g, 0, has_nontree_edge=True))
    (tensor([0]), tensor([1]), tensor([3]), tensor([5]), tensor([4]), tensor([2])),
    (tensor([0]), tensor([0]), tensor([0]), tensor([0]), tensor([0]), tensor([2]))
    """
    ghandle = graph._graph._handle
    source = utils.toindex(source).todgltensor()
    ret = _CAPI_DGLDFSLabeledEdges(
            ghandle,
            source,
            reversed,
            has_reverse_edge,
            has_nontree_edge,
            return_labels)
    all_edges = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    if return_labels:
        all_labels = utils.toindex(ret(1)).tousertensor()
        sections = utils.toindex(ret(2)).tonumpy().tolist()
        return (F.split(all_edges, sections, dim=0),
                F.split(all_labels, sections, dim=0))
    else:
        sections = utils.toindex(ret(1)).tonumpy().tolist()
        return F.split(all_edges, sections, dim=0)

_init_api("dgl.traversal")
