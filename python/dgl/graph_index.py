from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp

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

    def node_subgraph(self, v):
        """Return the induced node subgraph.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        GraphIndex
            The subgraph index.
        utils.Index
            The induced edge ids. This is also a map from new edge id to parent edge id.
        """
        v_array = v.todgltensor()
        rst = _CAPI_DGLGraphVertexSubgraph(self._handle, v_array)
        gi = GraphIndex(rst(0))
        induced_edges = utils.toindex(rst(2))
        return gi, induced_edges

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

    def incidence_matrix(self, oriented=False, sorted=False):
        """Return the incidence matrix representation of this graph.
        
        Parameters
        ----------
        oriented : bool, optional (default=False)
          Whether the returned incidence matrix is oriented.
        sorted : bool, optional (default=False)
          If true, nodes in L(G) are sorted as pairs.
          If False, nodes in L(G) are ordered by their edge id's in G.

        Returns
        -------
        utils.CtxCachedObject
            An object that returns tensor given context.
        """
        key = ('oriented ' if oriented else '') + \
                ('sorted ' if sorted else '') + 'incidence matrix'
        if not key in self._cache:
            src, dst, _ = self.edges(sorted=sorted)
            src = src.tousertensor()
            dst = dst.tousertensor()
            m = self.number_of_edges()
            eid = F.arange(m, dtype=F.int64)
            row = F.pack([src, dst])
            col = F.pack([eid, eid])
            idx = F.stack([row, col])

            diagonal = (src == dst)
            if oriented:
                x = -F.ones((m,))
                y = F.ones((m,))
                x[diagonal] = 0
                y[diagonal] = 0
                dat = F.pack([x, y])
            else:
                x = F.ones((m,))
                x[diagonal] = 0
                dat = F.pack([x, x])
            n = self.number_of_nodes()
            mat = F.sparse_tensor(idx, dat, [n, m])
            self._cache[key] = utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

        return self._cache[key]

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

    def from_scipy_sparse_matrix(self, adj):
        """Convert from scipy sparse matrix.

        Parameters
        ----------
        adj : scipy sparse matrix
        """
        self.clear()
        self.add_nodes(adj.shape[0])
        adj_coo = adj.tocoo()
        src = utils.toindex(adj_coo.row)
        dst = utils.toindex(adj_coo.col)
        self.add_edges(src, dst)

    def line_graph(self, backtracking=True, sorted=False):
        """Return the line graph of this graph.

        Parameters
        ----------
        backtracking : bool, optional (default=False)
          Whether (i, j) ~ (j, i) in L(G).
          (i, j) ~ (j, i) is the behavior of networkx.line_graph.
        sorted : bool, optional (default=False)
          If true, nodes in L(G) are sorted as pairs.
          If False, nodes in L(G) are ordered by their edge id's in G.

        Returns
        -------
        GraphIndex
            The line graph of this graph.
        """
        handle = _CAPI_DGLGraphLineGraph(self._handle, backtracking)
        return GraphIndex(handle)
        
def disjoint_union(graphs):
    """Return a disjoint union of the input graphs.

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
        The disjoint union
    """
    inputs = c_array(GraphIndexHandle, [gr._handle for gr in graphs])
    inputs = ctypes.cast(inputs, ctypes.c_void_p)
    handle = _CAPI_DGLDisjointUnion(inputs, len(graphs))
    return GraphIndex(handle)

def disjoint_partition(graph, num_or_size_splits):
    """Partition the graph disjointly.
   
    This is a reverse operation of DisjointUnion. The graph will be partitioned
    into num graphs. This requires the given number of partitions to evenly
    divides the number of nodes in the graph. If the a size list is given,
    the sum of the given sizes is equal.

    Parameters
    ----------
    graph : GraphIndex
        The graph to be partitioned
    num_or_size_splits : int or utils.Index
        The partition number of size splits

    Returns
    -------
    list of GraphIndex
        The partitioned graphs
    """
    if isinstance(num_or_size_splits, utils.Index):
        rst = _CAPI_DGLDisjointPartitionBySizes(
                graph._handle,
                num_or_size_splits.todgltensor())
    else:
        rst = _CAPI_DGLDisjointPartitionByNum(
                graph._handle,
                int(num_or_size_splits))
    graphs = []
    for val in rst.asnumpy():
        handle = ctypes.cast(int(val), ctypes.c_void_p)
        graphs.append(GraphIndex(handle))
    return graphs

def create_graph_index(graph_data=None):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    """
    if isinstance(graph_data, GraphIndex):
        return graph_data
    handle = _CAPI_DGLGraphCreate()
    gi = GraphIndex(handle)
    if graph_data is not None:
        gi.from_networkx(graph_data)
    return gi

_init_api("dgl.graph_index")
