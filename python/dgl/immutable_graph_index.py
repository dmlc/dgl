from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp

from ._ffi.function import _init_api
from . import backend as F
from . import utils
from .base import ALL, is_all, DGLError

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
        raise DGLError('Immutable graph doesn\'t support adding nodes')

    def add_edge(self, u, v):
        """Add one edge.
        
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        raise DGLError('Immutable graph doesn\'t support adding an edge')

    def add_edges(self, u, v):
        """Add many edges.
        
        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        raise DGLError('Immutable graph doesn\'t support adding edges')

    def clear(self):
        """Clear the graph."""
        raise DGLError('Immutable graph doesn\'t support clearing up')

    def is_multigraph(self):
        """Return whether the graph is a multigraph

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        # Immutable graph doesn't support multi-edge.
        return False

    def is_readonly(self):
        """Indicate whether the graph index is read-only.

        Returns
        -------
        bool
            True if it is a read-only graph, False otherwise.
        """
        return True

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

    def find_edges(self, eid):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        eid : utils.Index
            The edge ids.

        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        raise NotImplementedError('immutable graph doesn\'t implement find_edges for now.')

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
        deg = self._get_in_degree()
        if v.is_slice(0, self.number_of_nodes()):
            return utils.toindex(deg)
        else:
            v_array = v.tousertensor()
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
        deg = self._get_out_degree()
        if v.is_slice(0, self.number_of_nodes()):
            return utils.toindex(deg)
        else:
            v_array = v.tousertensor()
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
        return ImmutableSubgraphIndex(gi, self, induced_n, induced_e)

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
        return [ImmutableSubgraphIndex(gi, self, induced_n,
            induced_e) for gi, induced_n, induced_e in zip(gis, induced_nodes, induced_edges)]

    def edge_subgraph(self, e):
        """Return the induced edge subgraph.

        Parameters
        ----------
        e : utils.Index
            The edges.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        raise NotImplementedError('immutable graph doesn\'t implement edge_subgraph for now.')

    def neighbor_sampling(self, seed_ids, expand_factor, num_hops, neighbor_type,
                          node_prob, max_subgraph_size):
        if len(seed_ids) == 0:
            return []
        seed_ids = [v.tousertensor() for v in seed_ids]
        gis, induced_nodes, induced_edges = self._sparse.neighbor_sampling(seed_ids, expand_factor,
                                                                           num_hops, neighbor_type,
                                                                           node_prob,
                                                                           max_subgraph_size)
        induced_nodes = [utils.toindex(v) for v in induced_nodes]
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
        utils.Index
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        def get_adj(ctx):
            new_mat = self._sparse.adjacency_matrix(transpose)
            return F.copy_to(new_mat, ctx)
        return self._sparse.adjacency_matrix(transpose, ctx), None

    def incidence_matrix(self, type, ctx):
        """Return the incidence matrix representation of this graph.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix `I`:
        * "in":
          - I[v, e] = 1 if e is the in-edge of v (or v is the dst node of e);
          - I[v, e] = 0 otherwise.
        * "out":
          - I[v, e] = 1 if e is the out-edge of v (or v is the src node of e);
          - I[v, e] = 0 otherwise.
        * "both":
          - I[v, e] = 1 if e is the in-edge of v;
          - I[v, e] = -1 if e is the out-edge of v;
          - I[v, e] = 0 otherwise (including self-loop).

        Parameters
        ----------
        type : str
            Can be either "in", "out" or "both"
        ctx : context
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        utils.Index
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        raise NotImplementedError('immutable graph doesn\'t implement incidence_matrix for now.')

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
        if not isinstance(nx_graph, nx.Graph):
            nx_graph = (nx.MultiDiGraph(nx_graph) if self.is_multigraph()
                    else nx.DiGraph(nx_graph))
        else:
            nx_graph = nx_graph.to_directed()

        assert nx_graph.number_of_edges() > 0, "can't create an empty immutable graph"

        # nx_graph.edges(data=True) returns src, dst, attr_dict
        has_edge_id = 'id' in next(iter(nx_graph.edges(data=True)))[-1]
        if has_edge_id:
            num_edges = nx_graph.number_of_edges()
            src = np.zeros((num_edges,), dtype=np.int64)
            dst = np.zeros((num_edges,), dtype=np.int64)
            for u, v, attr in nx_graph.edges(data=True):
                eid = attr['id']
                src[eid] = u
                dst[eid] = v
        else:
            src = []
            dst = []
            for e in nx_graph.edges:
                src.append(e[0])
                dst.append(e[1])
            eid = np.arange(0, len(src), dtype=np.int64)

        num_nodes = nx_graph.number_of_nodes()
        # We store edge Ids as an edge attribute.
        eid = F.tensor(eid, dtype=np.int32)
        src = F.tensor(src, dtype=np.int64)
        dst = F.tensor(dst, dtype=np.int64)
        out_csr, _ = F.sparse_matrix(eid, ('coo', (src, dst)), (num_nodes, num_nodes))
        in_csr, _ = F.sparse_matrix(eid, ('coo', (dst, src)), (num_nodes, num_nodes))
        out_csr = out_csr.astype(np.int64)
        in_csr = in_csr.astype(np.int64)
        self._sparse = F.create_immutable_graph_index(in_csr, out_csr)

    def from_scipy_sparse_matrix(self, adj):
        """Convert from scipy sparse matrix.

        NOTE: we assume the row is src nodes and the col is dst nodes.

        Parameters
        ----------
        adj : scipy sparse matrix
        """
        assert isinstance(adj, sp.csr_matrix) or isinstance(adj, sp.coo_matrix), \
                "The input matrix has to be a SciPy sparse matrix."
        out_mat = adj.tocoo()
        self._sparse.from_coo_matrix(out_mat)

    def from_edge_list(self, elist):
        """Convert from an edge list.

        Paramters
        ---------
        elist : list
            List of (u, v) edge tuple.
        """
        self.clear()
        src, dst = zip(*elist)
        src = np.array(src)
        dst = np.array(dst)
        num_nodes = max(src.max(), dst.max()) + 1
        min_nodes = min(src.min(), dst.min())
        if min_nodes != 0:
            raise DGLError('Invalid edge list. Nodes must start from 0.')
        edge_ids = mx.nd.arange(0, len(src), step=1, repeat=1, dtype=np.int32)
        src = mx.nd.array(src, dtype=np.int64)
        dst = mx.nd.array(dst, dtype=np.int64)
        # TODO we can't generate a csr_matrix with np.int64 directly.
        in_csr = mx.nd.sparse.csr_matrix((edge_ids, (dst, src)),
                                         shape=(num_nodes, num_nodes)).astype(np.int64)
        out_csr = mx.nd.sparse.csr_matrix((edge_ids, (src, dst)),
                                          shape=(num_nodes, num_nodes)).astype(np.int64)
        self.__init__(in_csr, out_csr)

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
        raise NotImplementedError('immutable graph doesn\'t implement line_graph')

class ImmutableSubgraphIndex(ImmutableGraphIndex):
    """Graph index for an immutable subgraph.

    Parameters
    ----------
    backend_sparse : a sparse matrix from the backend framework.
        The sparse matrix that represents a subgraph.
    paranet : GraphIndex
        The parent graph index.
    induced_nodes : tensor
        The parent node ids in this subgraph.
    induced_edges : a lambda function that returns a tensor
        The parent edge ids in this subgraph.
    """
    def __init__(self, backend_sparse, parent, induced_nodes, induced_edges):
        super(ImmutableSubgraphIndex, self).__init__(backend_sparse)

        self._parent = parent
        self._induced_nodes = induced_nodes
        self._induced_edges = induced_edges

    @property
    def induced_edges(self):
        """Return parent edge ids.

        Returns
        -------
        A lambda function that returns utils.Index
            The parent edge ids.
        """
        return lambda: utils.toindex(self._induced_edges())

    @property
    def induced_nodes(self):
        """Return parent node ids.

        Returns
        -------
        utils.Index
            The parent node ids.
        """
        return utils.toindex(self._induced_nodes)

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
    raise NotImplementedError('immutable graph doesn\'t implement disjoint_union for now.')

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
    raise NotImplementedError('immutable graph doesn\'t implement disjoint_partition for now.')

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

    try:
        # Let's try using the graph data to generate an immutable graph index.
        # If we are successful, we can return the immutable graph index immediately.
        # If graph_data is None, we return an empty graph index.
        # If we can't create a graph index, we'll use the code below to handle the graph.
        return ImmutableGraphIndex(F.create_immutable_graph_index(graph_data))
    except:
        pass

    # Let's create an empty graph index first.
    gi = ImmutableGraphIndex(F.create_immutable_graph_index())

    # edge list
    if isinstance(graph_data, (list, tuple)):
        try:
            gi.from_edge_list(graph_data)
            return gi
        except:
            raise DGLError('Graph data is not a valid edge list.')

    # scipy format
    if isinstance(graph_data, sp.spmatrix):
        try:
            gi.from_scipy_sparse_matrix(graph_data)
            return gi
        except:
            raise DGLError('Graph data is not a valid scipy sparse matrix.')

    # networkx - any format
    try:
        gi.from_networkx(graph_data)
    except:
        raise DGLError('Error while creating graph from input of type "%s".'
                         % type(graph_data))

    return gi

_init_api("dgl.immutable_graph_index")
