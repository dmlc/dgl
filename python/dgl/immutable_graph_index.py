from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp

from ._ffi.function import _init_api
from . import backend as F
from . import utils

class ImmutableGraphIndex(object):
    """Graph index object on immutable graphs.

    Parameters
    ----------
    backend_csr: a csr array provided by the backend framework.
    """
    def __init__(self, backend_sparse):
        self._sparse = backend_sparse
        self._num_nodes = None
        self._num_edges = None
        self._in_deg = None
        self._out_deg = None
        self._cache = {}

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

    def is_multigraph(self):
        """Return whether the graph is a multigraph

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        # Immutable graph doesn't support multi-edge.
        return False

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        if self._num_nodes is None:
            self._num_nodes = self._sparse.number_of_nodes()
        return self._num_nodes

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        if self._num_edges is None:
            self._num_edges = self._sparse.number_of_edges()
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

    def has_edge_between(self, u, v):
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
        u = F.tensor([u], dtype=F.int64)
        v = F.tensor([v], dtype=F.int64)
        return self._sparse.has_edges(u, v).asnumpy()[0]

    def has_edges_between(self, u, v):
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
        ret = self._sparse.has_edges(u.tousertensor(), v.tousertensor())
        return utils.toindex(ret)

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
        pred = self._sparse.predecessors(v, radius)
        return utils.toindex(pred)

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
        succ = self._sparse.successors(v, radius)
        return utils.toindex(succ)

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
        u = F.tensor([u], dtype=F.int64)
        v = F.tensor([v], dtype=F.int64)
        _, _, id = self._sparse.edge_ids(u, v)
        return utils.toindex(id)

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
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        u = u.tousertensor()
        v = v.tousertensor()
        u, v, ids = self._sparse.edge_ids(u, v)
        return utils.toindex(u), utils.toindex(v), utils.toindex(ids)

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
        indptr, src, edges = self._sparse.in_edges(dst)
        off = utils.toindex(indptr)
        dst = _CAPI_DGLExpandIds(v.todgltensor(), off.todgltensor())
        return utils.toindex(src), utils.toindex(dst), utils.toindex(edges)

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
        indptr, dst, edges = self._sparse.out_edges(src)
        off = utils.toindex(indptr)
        src = _CAPI_DGLExpandIds(v.todgltensor(), off.todgltensor())
        return utils.toindex(src), utils.toindex(dst), utils.toindex(edges)

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
        if "all_edges" in self._cache:
            return self._cache["all_edges"]
        src, dst, edges = self._sparse.edges(sorted)
        self._cache["all_edges"] = (utils.toindex(src), utils.toindex(dst), utils.toindex(edges))
        return self._cache["all_edges"]

    def _get_in_degree(self):
        if 'in_deg' not in self._cache:
            self._cache['in_deg'] = self._sparse.get_in_degree()
        return self._cache['in_deg']

    def _get_out_degree(self):
        if 'out_deg' not in self._cache:
            self._cache['out_deg'] = self._sparse.get_out_degree()
        return self._cache['out_deg']

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
        v_array = v.tousertensor()
        deg = self._get_in_degree()
        return utils.toindex(F.gather_row(deg, v_array))

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
        v_array = v.tousertensor()
        deg = self._get_out_degree()
        return utils.toindex(F.gather_row(deg, v_array))

    def node_subgraph(self, v):
        """Return the induced node subgraph.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        ImmutableSubgraphIndex
            The subgraph index.
        """
        v = v.tousertensor()
        gi, induced_n, induced_e = self._sparse.node_subgraph(v)
        induced_nodes = utils.toindex(induced_n)
        induced_edges = utils.toindex(induced_e)
        return ImmutableSubgraphIndex(gi, self, induced_nodes, induced_edges)

    def node_subgraphs(self, vs_arr):
        """Return the induced node subgraphs.

        Parameters
        ----------
        vs_arr : a vector of utils.Index
            The nodes.

        Returns
        -------
        a vector of ImmutableSubgraphIndex
            The subgraph index.
        """
        vs_arr = [v.tousertensor() for v in vs_arr]
        gis, induced_nodes, induced_edges = self._sparse.node_subgraphs(vs_arr)
        induced_nodes = [utils.toindex(v) for v in induced_nodes]
        induced_edges = [utils.toindex(e) for e in induced_edges]
        return [ImmutableSubgraphIndex(gi, self, induced_n,
            induced_e) for gi, induced_n, induced_e in zip(gis, induced_nodes, induced_edges)]

    def neighbor_sampling(self, seed_ids, expand_factor, num_hops, node_prob,
                          max_subgraph_size):
        if len(seed_ids) == 0:
            return []
        seed_ids = [v.tousertensor() for v in seed_ids]
        gis, induced_nodes, induced_edges = self._sparse.neighbor_sampling(seed_ids, expand_factor,
                                                                           num_hops, node_prob,
                                                                           max_subgraph_size)
        induced_nodes = [utils.toindex(v) for v in induced_nodes]
        induced_edges = [utils.toindex(e) for e in induced_edges]
        return [ImmutableSubgraphIndex(gi, self, induced_n,
            induced_e) for gi, induced_n, induced_e in zip(gis, induced_nodes, induced_edges)]

    def adjacency_matrix(self, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to tranpose the returned adjacency matrix.

        Returns
        -------
        utils.CtxCachedObject
            An object that returns tensor given context.
        """
        def get_adj(ctx):
            new_mat = self._sparse.adjacency_matrix(transpose)
            return F.copy_to(new_mat, ctx)
        return self._sparse.adjacency_matrix(transpose, ctx)

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
        out_mat = nx.convert_matrix.to_scipy_sparse_matrix(nx_graph, format='coo')
        self._sparse.from_coo_matrix(out_mat)

    def from_scipy_sparse_matrix(self, adj):
        """Convert from scipy sparse matrix.

        Parameters
        ----------
        adj : scipy sparse matrix
        """
        assert isinstance(adj, sp.csr_matrix) or isinstance(adj, sp.coo_matrix), \
                "The input matrix has to be a SciPy sparse matrix."
        out_mat = adj.tocoo()
        self._sparse.from_coo_matrix(out_mat)

    def line_graph(self, backtracking=True):
        """Return the line graph of this graph.

        Parameters
        ----------
        backtracking : bool, optional (default=False)
          Whether (i, j) ~ (j, i) in L(G).
          (i, j) ~ (j, i) is the behavior of networkx.line_graph.

        Returns
        -------
        ImmutableGraphIndex
            The line graph of this graph.
        """
        raise Exception('immutable graph doesn\'t support line_graph')

class ImmutableSubgraphIndex(ImmutableGraphIndex):
    def __init__(self, backend_sparse, parent, induced_nodes, induced_edges):
        super(ImmutableSubgraphIndex, self).__init__(backend_sparse)

        self._parent = parent
        self._induced_nodes = induced_nodes
        self._induced_edges = induced_edges

    @property
    def induced_edges(self):
        return self._induced_edges

    @property
    def induced_nodes(self):
        return self._induced_nodes

def create_immutable_graph_index(graph_data=None):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    """
    if isinstance(graph_data, ImmutableGraphIndex):
        return graph_data
    assert F.create_immutable_graph_index is not None, \
            "The selected backend doesn't support read-only graph!"

    gi = ImmutableGraphIndex(F.create_immutable_graph_index())
    if graph_data is None:
        return gi

    # scipy format
    if isinstance(graph_data, sp.spmatrix):
        try:
            gi.from_scipy_sparse_matrix(graph_data)
            return gi
        except:
            raise Exception('Graph data is not a valid scipy sparse matrix.')

    # networkx - any format
    try:
        gi.from_networkx(graph_data)
    except:
        raise Exception('Error while creating graph from input of type "%s".'
                         % type(graph_data))

    return gi

_init_api("dgl.immutable_graph_index")
