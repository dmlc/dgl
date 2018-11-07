"""Module for graph traversal methods."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from . import backend as F
from . import utils

__all__ = ['bfs_nodes_generator', 'topological_nodes_generator', 'dfs_edges_generator']

def bfs_nodes_generator(graph, source, reversed=False):
    """Node frontiers generator using breadth-first search.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    source : list, tensor of nodes
        Source nodes.
    reversed : bool, optional
        If true, traverse following the in-edge direction.

    Returns
    -------
    list of node frontiers
        Each node frontier is a list, tensor of nodes.
    """
    ghandle = graph._graph._handle
    source = utils.toindex(source).todgltensor()
    ret = _CAPI_DGLBFSNodes(ghandle, source, reversed)
    all_nodes = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tousertensor().tolist()
    return F.split(all_nodes, sections, dim=0)

def topological_nodes_generator(graph, reversed=False):
    """Node frontiers generator using topological traversal.

    Parameters
    ----------
    graph : DGLGraph
        The graph object.
    reversed : bool, optional
        If true, traverse following the in-edge direction.

    Returns
    -------
    list of node frontiers
        Each node frontier is a list, tensor of nodes.
    """
    ghandle = graph._graph._handle
    ret = _CAPI_DGLTopologicalNodes(ghandle, reversed)
    all_nodes = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tousertensor().tolist()
    return F.split(all_nodes, sections, dim=0)

def dfs_edges_generator(graph, src, reversed=False):
    """
    """
    ghandle = graph._graph._handle
    ret = _CAPI_DGLDFSEdges(ghandle, src, reversed)
    all_edges = utils.toindex(ret(0)).tousertensor()
    # TODO(minjie): how to support directly creating python list
    sections = utils.toindex(ret(1)).tousertensor().tolist()
    #return F.split(all_nodes, sections, dim=0)
    return all_edges, sections

def dfs_labeled_edges(self, src, out, reverse_edge, nontree_edge):
    """ Produce edges in a depth-first-search (DFS) labeled by type.

    Parameters
    ----------
    src : int, list or tensor
        Source nodes.
    out : bool
        Whether to following incoming or outgoing edges.
    reverse_edge : bool
        Whether to yield reverse edges.
    nontree_edge : bool
        Whether to yield nontree edges.

    Returns
    -------
    list of tuple of tensor
        A tuple in returned list consists of three tensors:
        * src: Source id's.
        * dst: Destination id's.
        * type: A tensor that takes value in `GraphIndex.FORWARD`, `GraphIndex.REVERSE` and `GraphIndex.NONTREE`.
        Propagation from source nodes to destination nodes in the same tuple can be parallelized.
    """
    src = utils.toindex(src).todgltensor()
    ret = _CAPI_DGLGraphDFSLabeledEdges(self._handle, src, out, reverse_edge, nontree_edge)
    src = utils.toindex(ret(0)).tousertensor()
    dst = utils.toindex(ret(1)).tousertensor()
    type = utils.toindex(ret(2)).tousertensor()
    size = F.asnumpy(utils.toindex(ret(3)).tousertensor()).tolist()
    return list(zip(F.unpack(src, size), F.unpack(dst, size), F.unpack(type, size)))

_init_api("dgl.traversal")
