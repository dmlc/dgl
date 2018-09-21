from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx

from ._ffi.base import c_array
from ._ffi.function import _init_api
from . import backend as F
from . import utils

GraphIndexHandle = ctypes.c_void_p

class GraphIndex(object):
    """Graph index object.

    Parameters
    ----------
    handle : GraphIndexHandle
        Handler
    """
    def __init__(self, handle):
        self._handle = handle
        self._cache = {}

    def __del__(self):
        """Free this graph index object."""
        _CAPI_DGLGraphFree(self._handle)

    def add_nodes(self, num):
        """Add nodes.
        
        Parameters
        ----------
        num : int
            Number of nodes to be added.
        """
        _CAPI_DGLGraphAddVertices(self._handle, num);
        self._cache.clear()

    def add_edge(self, u, v):
        """Add one edge.
        
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        _CAPI_DGLGraphAddEdge(self._handle, u, v);
        self._cache.clear()

    def add_edges(self, u, v):
        """Add many edges.
        
        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        _CAPI_DGLGraphAddEdges(self._handle, u_array, v_array)
        self._cache.clear()

    def clear(self):
        """Clear the graph."""
        _CAPI_DGLGraphClear(self._handle)
        self._cache.clear()

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return _CAPI_DGLGraphNumVertices(self._handle)

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return _CAPI_DGLGraphNumEdges(self._handle)

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists
        """
        return _CAPI_DGLGraphHasVertex(self._handle, vid)

    def has_nodes(self, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        vid : utils.Index
            The nodes

        Returns
        -------
        utils.Index
            0-1 array indicating existence
        """
        vid_array = vids.todgltensor()
        return utils.toindex(_CAPI_DGLGraphHasVertices(self._handle, vid_array))

    def has_edge(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists
        """
        return _CAPI_DGLGraphHasEdge(self._handle, u, v)

    def has_edges(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.

        Returns
        -------
        utils.Index
            0-1 array indicating existence
        """
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphHasEdges(self._handle, u_array, v_array))

    def predecessors(self, v, radius=1):
        """Return the predecessors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        utils.Index
            Array of predecessors
        """
        return utils.toindex(_CAPI_DGLGraphPredecessors(self._handle, v, radius))

    def successors(self, v, radius=1):
        """Return the successors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        utils.Index
            Array of successors
        """
        return utils.toindex(_CAPI_DGLGraphSuccessors(self._handle, v, radius))

    def edge_id(self, u, v):
        """Return the id of the edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        int
            The edge id.
        """
        return _CAPI_DGLGraphEdgeId(self._handle, u, v)

    def edge_ids(self, u, v):
        """Return the edge ids.

        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.

        Returns
        -------
        utils.Index
            Teh edge id array.
        """
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphEdgeIds(self._handle, u_array, v_array))

    def in_edges(self, v):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : utils.Index
            The node(s).
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        if len(v) == 1:
            edge_array = _CAPI_DGLGraphInEdges_1(self._handle, v[0])
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphInEdges_2(self._handle, v_array)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    def out_edges(self, v):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : utils.Index
            The node(s).
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        if len(v) == 1:
            edge_array = _CAPI_DGLGraphOutEdges_1(self._handle, v[0])
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphOutEdges_2(self._handle, v_array)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    def edges(self, sorted=False):
        """Return all the edges

        Parameters
        ----------
        sorted : bool
            True if the returned edges are sorted by their src and dst ids.
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        edge_array = _CAPI_DGLGraphEdges(self._handle, sorted)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    def in_degree(self, v):
        """Return the in degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return _CAPI_DGLGraphInDegree(self._handle, v)

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The in degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphInDegrees(self._handle, v_array))

    def out_degree(self, v):
        """Return the out degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return _CAPI_DGLGraphOutDegree(self._handle, v)

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The out degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphOutDegrees(self._handle, v_array))

    def adjacency_matrix(self):
        """Return the adjacency matrix representation of this graph.

        Returns
        -------
        utils.CtxCachedObject
            An object that returns tensor given context.
        """
        if not 'adj' in self._cache:
            src, dst, _ = self.edges(sorted=False)
            src = F.unsqueeze(src.tousertensor(), 0)
            dst = F.unsqueeze(dst.tousertensor(), 0)
            idx = F.pack([dst, src])
            n = self.number_of_nodes()
            dat = F.ones((self.number_of_edges(),))
            mat = F.sparse_tensor(idx, dat, [n, n])
            self._cache['adj'] = utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))
        return self._cache['adj']

    def to_networkx(self):
        """Convert to networkx graph.

        The edge id will be saved as the 'id' edge attribute.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        src, dst, eid = self.edges()
        ret = nx.DiGraph()
        for u, v, id in zip(src, dst, eid):
            ret.add_edge(u, v, id=id)
        return ret

    def from_networkx(self, nx_graph):
        """Convert from networkx graph.

        If 'id' edge attribute exists, the edge will be added follows
        the edge id order. Otherwise, order is undefined.
        
        Parameters
        ----------
        nx_graph : networkx.DiGraph
            The nx graph
        """
        self.clear()
        if not isinstance(nx_graph, nx.DiGraph):
            nx_graph = nx.DiGraph(nx_graph)
        num_nodes = nx_graph.number_of_nodes()
        self.add_nodes(num_nodes)
        has_edge_id = 'id' in next(iter(nx_graph.edges))
        if has_edge_id:
            num_edges = nx_graph.number_of_edges()
            src = np.zeros((num_edges,), dtype=np.int64)
            dst = np.zeros((num_edges,), dtype=np.int64)
            for e, attr in nx_graph.edges.items:
                u, v = e
                eid = attr['id']
                src[eid] = u
                dst[eid] = v
        else:
            src = []
            dst = []
            for u, v in nx_graph.edges:
                src.append(u)
                dst.append(v)
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        self.add_edges(src, dst)

    @staticmethod
    def merge(graphs):
        """Merge a list of graphs into one graph.

        The new graph will include all the nodes/edges in the given graphs.
        Nodes/Edges will be relabled by adding the cumsum of the previous graph sizes
        in the given sequence order. For example, giving input [g1, g2, g3], where
        they have 5, 6, 7 nodes respectively. Then node#2 of g2 will become node#7
        in the result graph. Edge ids are re-assigned similarly.

        Parameters
        ----------
        graphs : iterable of GraphIndex
            The input graphs

        Returns
        -------
        GraphIndex
            The merged graph
        """
        inputs = c_array(GraphIndexHandle, [gr._handle for gr in graphs])
        inputs = ctypes.cast(inputs, ctypes.c_void_p)
        handle = _CAPI_DGLGraphMerge(inputs, len(graphs))
        return GraphIndex(handle)

def create_graph_index(graph_data=None):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    """
    handle = _CAPI_DGLGraphCreate()
    gi = GraphIndex(handle)
    if graph_data is not None:
        gi.from_networkx(graph_data)
    return gi

_init_api("dgl.graph_index")
