"""Module for immutable graph index.

NOTE: this is currently a temporary solution.
"""
# pylint: disable=abstract-method,unused-argument

from __future__ import absolute_import

import numpy as np
import networkx as nx
import scipy.sparse as sp

from ._ffi.function import _init_api
from . import backend as F
from . import utils
from .base import DGLError

class ImmutableGraphIndex(object):
    """Graph index object on immutable graphs.

    Parameters
    ----------
    backend_csr: a csr array provided by the backend framework.
    """
    def __init__(self, handle):
        self._handle = handle
        self._num_nodes = None
        self._num_edges = None
        self._cache = {}

    def init(self, src_ids, dst_ids, edge_ids, num_nodes):
        """The actual init function"""
        self._handle = _CAPI_DGLGraphCreate(src_ids.todgltensor(), dst_ids.todgltensor(),
                                            edge_ids.todgltensor(), False, num_nodes)
        self._num_nodes = num_nodes
        self._num_edges = None

    def __del__(self):
        """Free this graph index object."""
        if self._handle is not None:
            _CAPI_DGLGraphFree(self._handle)

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
        return bool(_CAPI_DGLGraphIsMultigraph(self._handle))

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
            self._num_nodes = _CAPI_DGLGraphNumVertices(self._handle)
        return self._num_nodes

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        if self._num_edges is None:
            self._num_edges = _CAPI_DGLGraphNumEdges(self._handle)
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
        return bool(_CAPI_DGLGraphHasVertex(self._handle, vid))

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
        return bool(_CAPI_DGLGraphHasEdgeBetween(self._handle, u, v))

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
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphHasEdgesBetween(self._handle, u_array, v_array))

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
        return utils.toindex(_CAPI_DGLGraphEdgeId(self._handle, u, v))

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
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        edge_array = _CAPI_DGLGraphEdgeIds(self._handle, u_array, v_array)

        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))

        return src, dst, eid

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

    @utils.cached_member(cache='_cache', prefix='edges')
    def edges(self, return_sorted=False):
        """Return all the edges

        Parameters
        ----------
        return_sorted : bool
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
        key = 'edges_s%d' % return_sorted
        if key not in self._cache:
            edge_array = _CAPI_DGLGraphEdges(self._handle, return_sorted)
            src = utils.toindex(edge_array(0))
            dst = utils.toindex(edge_array(1))
            eid = utils.toindex(edge_array(2))
            self._cache[key] = (src, dst, eid)
        return self._cache[key]

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
        ImmutableSubgraphIndex
            The subgraph index.
        """
        v_array = v.todgltensor()
        rst = _CAPI_DGLGraphVertexSubgraph(self._handle, v_array)
        induced_edges = utils.toindex(rst(2))
        return ImmutableSubgraphIndex(rst(0), self, v, induced_edges)

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
        # TODO(zhengda) we should parallelize the computation here in CAPI.
        return [self.node_subgraph(v) for v in vs_arr]

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

    def neighbor_sampling(self, seed_ids, expand_factor, num_hops, neighbor_type, node_prob):
        """Neighborhood sampling"""
        if len(seed_ids) == 0:
            return []

        seed_ids = [v.todgltensor() for v in seed_ids]
        num_subgs = len(seed_ids)
        if node_prob is None:
            rst = _uniform_sampling(self, seed_ids, neighbor_type, num_hops, expand_factor)
        else:
            rst = _nonuniform_sampling(self, node_prob, seed_ids, neighbor_type, num_hops,
                                       expand_factor)

        return [ImmutableSubgraphIndex(rst(i), self, rst(num_subgs + i),
                                       rst(num_subgs * 2 + i)) for i in range(num_subgs)]

    def adjacency_matrix(self, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to transpose the returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        utils.Index
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        rst = _CAPI_DGLGraphGetCSR(self._handle, transpose)
        indptr = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
        indices = F.copy_to(utils.toindex(rst(1)).tousertensor(), ctx)
        shuffle = utils.toindex(rst(2))
        dat = F.ones(indices.shape, dtype=F.float32, ctx=ctx)
        return F.sparse_matrix(dat, ('csr', indices, indptr),
                               (self.number_of_nodes(), self.number_of_nodes()))[0], shuffle

    def incidence_matrix(self, typestr, ctx):
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
        typestr : str
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
        for u, v, e in zip(src, dst, eid):
            ret.add_edge(u, v, id=e)
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
            nx_graph = nx.DiGraph(nx_graph)
        else:
            if not nx_graph.is_directed():
                # to_directed creates a deep copy of the networkx graph even if
                # the original graph is already directed and we do not want to do it.
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
        eid = utils.toindex(eid)
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        self.init(src, dst, eid, num_nodes)

    def from_scipy_sparse_matrix(self, adj):
        """Convert from scipy sparse matrix.

        NOTE: we assume the row is src nodes and the col is dst nodes.

        Parameters
        ----------
        adj : scipy sparse matrix
        """
        assert isinstance(adj, (sp.csr_matrix, sp.coo_matrix)), \
                "The input matrix has to be a SciPy sparse matrix."
        num_nodes = max(adj.shape[0], adj.shape[1])
        out_mat = adj.tocoo()
        src_ids = utils.toindex(out_mat.row)
        dst_ids = utils.toindex(out_mat.col)
        edge_ids = utils.toindex(F.arange(0, len(out_mat.row)))
        self.init(src_ids, dst_ids, edge_ids, num_nodes)

    def from_edge_list(self, elist):
        """Convert from an edge list.

        Paramters
        ---------
        elist : list
            List of (u, v) edge tuple.
        """
        src, dst = zip(*elist)
        src = np.array(src)
        dst = np.array(dst)
        src_ids = utils.toindex(src)
        dst_ids = utils.toindex(dst)
        num_nodes = max(src.max(), dst.max()) + 1
        edge_ids = utils.toindex(F.arange(0, len(src)))
        # TODO we need to detect multigraph automatically.
        self.init(src_ids, dst_ids, edge_ids, num_nodes)

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
    def __init__(self, handle, parent, induced_nodes, induced_edges):
        super(ImmutableSubgraphIndex, self).__init__(handle)

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
        return utils.toindex(self._induced_edges)

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
    Nodes/Edges will be relabeled by adding the cumsum of the previous graph sizes
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

    # Let's create an empty graph index first.
    gidx = ImmutableGraphIndex(None)

    # edge list
    if isinstance(graph_data, (list, tuple)):
        try:
            gidx.from_edge_list(graph_data)
            return gidx
        except Exception:  # pylint: disable=broad-except
            raise DGLError('Graph data is not a valid edge list for immutable_graph_index.')

    # scipy format
    if isinstance(graph_data, sp.spmatrix):
        try:
            gidx.from_scipy_sparse_matrix(graph_data)
            return gidx
        except Exception:  # pylint: disable=broad-except
            raise DGLError('Graph data is not a valid scipy sparse matrix.')

    # networkx - any format
    try:
        gidx.from_networkx(graph_data)
    except Exception:  # pylint: disable=broad-except
        raise DGLError('Error while creating graph from input of type "%s".'
                       % type(graph_data))

    return gidx

_init_api("dgl.immutable_graph_index")

_NEIGHBOR_SAMPLING_APIS = {
    1: _CAPI_DGLGraphUniformSampling,
    2: _CAPI_DGLGraphUniformSampling2,
    4: _CAPI_DGLGraphUniformSampling4,
    8: _CAPI_DGLGraphUniformSampling8,
    16: _CAPI_DGLGraphUniformSampling16,
    32: _CAPI_DGLGraphUniformSampling32,
    64: _CAPI_DGLGraphUniformSampling64,
    128: _CAPI_DGLGraphUniformSampling128,
}

_EMPTY_ARRAYS = [utils.toindex(F.ones(shape=(0), dtype=F.int64, ctx=F.cpu()))]

def _uniform_sampling(gidx, seed_ids, neigh_type, num_hops, expand_factor):
    num_seeds = len(seed_ids)
    empty_ids = []
    if len(seed_ids) > 1 and len(seed_ids) not in _NEIGHBOR_SAMPLING_APIS.keys():
        remain = 2**int(math.ceil(math.log2(len(dgl_ids)))) - len(dgl_ids)
        empty_ids = _EMPTY_ARRAYS[0:remain]
        seed_ids.extend([empty.todgltensor() for empty in empty_ids])
    assert len(seed_ids) in _NEIGHBOR_SAMPLING_APIS.keys()
    return _NEIGHBOR_SAMPLING_APIS[len(seed_ids)](gidx._handle, *seed_ids, neigh_type,
                                                  num_hops, expand_factor, num_seeds)
