from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp
import mxnet as mx

from . import backend as F
from . import utils

class ImmutableGraphIndex(object):
    """Graph index object on immutable graphs.

    Parameters
    ----------
    handle : GraphIndexHandle
        Handler
    """
    def __init__(self, in_csr, out_csr):
        self._in_csr = in_csr
        self._out_csr = out_csr
        self._num_edges = None

    def add_nodes(self, num):
        """Add nodes.
        
        Parameters
        ----------
        num : int
            Number of nodes to be added.
        """
        raise Exception('Immutable graph doesn\'t support adding nodes')

    def add_edge(self, u, v):
        """Add one edge.
        
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        raise Exception('Immutable graph doesn\'t support adding an edge')

    def add_edges(self, u, v):
        """Add many edges.
        
        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        raise Exception('Immutable graph doesn\'t support adding edges')

    def clear(self):
        """Clear the graph."""
        raise Exception('Immutable graph doesn\'t support clearing up')

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return len(self._in_csr)

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        if self._num_edges is None:
            self._num_edges = mx.nd.sum(self._in_csr)
        return self._num_edges

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
        return vid < self.number_of_nodes()

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
        vid_array = vids.tousertensor()
        return utils.toindex(vid_array < self.number_of_nodes())

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
        raise Exception('has_edge isn\'t supported temporarily')

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
        raise Exception('has_edges isn\'t supported temporarily')

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
        if radius > 1:
            raise Exception('Immutable graph doesn\'t support predecessors with radius > 1 for now.')
        row = self._in_csr[v]
        return utils.toindex(row.indices)

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
        if radius > 1:
            raise Exception('Immutable graph doesn\'t support successors with radius > 1 for now.')
        col = self._out_csr[v]
        return utils.toindex(col.indices)

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
        raise Exception('edge_id isn\'t supported temporarily')

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
        raise Exception('edge_ids isn\'t supported temporarily')

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
        dst = v.tousertensor()
        rows = mx.nd.take(self._in_csr, dst)
        return utils.toindex(rows.indices), None, None

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
        src = v.tousertensor()
        rows = mx.nd.take(self._out_csr, src)
        return utils.toindex(rows.indices), None, None

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
        raise Exception('Immutable graph doesn\'t support getting all edges.')

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
        deg = self._get_in_degree()
        return deg[v]

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
        deg = self._get_in_degree()
        return deg[v_array]

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
        deg = self._get_out_degree()
        return deg[v]

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
        deg = self._get_out_degree()
        return deg[v_array]

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
        """
        in_csr = mx.nd.contrib.dgl_subgraph(self._in_csr, v.tousertensor(), return_mapping=False)
        return ImmutableGraphIndex(in_csr, None)

    def node_subgraphs(self, vs_arr):
        """Return the induced node subgraphs.

        Parameters
        ----------
        vs_arr : a vector of utils.Index
            The nodes.

        Returns
        -------
        a vector of GraphIndex
            The subgraph index.
        """
        vs_arr = [v.tousertensor() for v in vs_arr]
        in_csrs = mx.nd.contrib.dgl_subgraph(self._in_csr, *vs_arr, return_mapping=False)
        return [ImmutableGraphIndex(in_csr, None) for in_csr in in_csrs]

    def adjacency_matrix(self, edge_type='in'):
        """Return the adjacency matrix representation of this graph.
        For a directed graph, we can construct two adjacency matrices:
        one stores in-edges as non-zero entries, the other stores out-edges
        as non-zero entries.

        Parameters
        ----------
        edge_type : a string
            The edge type used for constructing an adjacency matrix.

        Returns
        -------
        utils.CtxCachedObject
            An object that returns tensor given context.
        """
        if edge_type == 'in':
            return self._in_csr
        else:
            return self._out_csr

    def incidence_matrix(self, oriented=False):
        """Return the incidence matrix representation of this graph.
        
        Parameters
        ----------
        oriented : bool, optional (default=False)
          Whether the returned incidence matrix is oriented.

        Returns
        -------
        utils.CtxCachedObject
            An object that returns tensor given context.
        """
        raise Exception('immutable graph doesn\'t support incidence_matrix for now.')

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
        assert isinstance(nx_graph, nx.DiGraph), "The input graph has to be a NetworkX DiGraph."
        # We store edge Ids as an edge attribute.
        out_mat = nx.convert_matrix.to_scipy_sparse_matrix(nx_graph, dtype=np.int32, format='csr').astype(np.int64)
        in_mat = nx.convert_matrix.to_scipy_sparse_matrix(nx_graph, dtype=np.int32, format='csc').astype(np.int64).transpose()
        self.__init__(mx.nd.sparse.csr_matrix(in_mat), mx.nd.sparse.csr_matrix(out_mat))

    def from_scipy_sparse_matrix(self, adj):
        """Convert from scipy sparse matrix.

        Parameters
        ----------
        adj : scipy sparse matrix
        """
        assert isinstance(adj, sp.csr_matrix) or isinstance(adj, sp.coo_matrix), \
                "The input matrix has to be a SciPy sparse matrix."
        self._out_csr = mx.nd.sparse.csr_matrix(adj.tocsr(), dtype=np.int64)
        csc = adj.tocsc()
        self._in_csr = mx.nd.sparse.csr_matrix(csc.transpose(), dtype=np.int64)

    def line_graph(self, backtracking=True):
        """Return the line graph of this graph.

        Parameters
        ----------
        backtracking : bool, optional (default=False)
          Whether (i, j) ~ (j, i) in L(G).
          (i, j) ~ (j, i) is the behavior of networkx.line_graph.

        Returns
        -------
        GraphIndex
            The line graph of this graph.
        """
        raise Exception('immutable graph doesn\'t support line_graph')

def create_immutable_graph_index(graph_data=None):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    """
    if isinstance(graph_data, ImmutableGraphIndex):
        return graph_data
    if graph_data is None:
        return ImmutableGraphIndex(None, None)

    try:
        graph_data = graph_data.get_graph()
    except AttributeError:
        pass

    if isinstance(graph_data, nx.DiGraph):
        gi = ImmutableGraphIndex(None, None)
        gi.from_networkx(graph_data)
    elif isinstance(graph_data, sp.csr_matrix) or isinstance(graph_data, sp.coo_matrix):
        gi = ImmutableGraphIndex(None, None)
        gi.from_scipy_sparse_matrix(graph_data)
    elif isinstance(graph_data, mx.nd.sparse.CSRNDArray):
        gi = ImmutableGraphIndex(graph_data, None)
    else:
        raise Exception('cannot create an immutable graph index from unknown format')
    return gi
