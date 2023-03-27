"""Module for graph index class definition."""
from __future__ import absolute_import

import networkx as nx
import numpy as np
import scipy

from . import backend as F, utils
from ._ffi.function import _init_api
from ._ffi.object import ObjectBase, register_object
from .base import dgl_warning, DGLError


class BoolFlag(object):
    """Bool flag with unknown value"""

    BOOL_UNKNOWN = -1
    BOOL_FALSE = 0
    BOOL_TRUE = 1


@register_object("graph.Graph")
class GraphIndex(ObjectBase):
    """Graph index object.

    Note
    ----
    Do not create GraphIndex directly, you can create graph index object using
    following functions:

    - `dgl.graph_index.from_edge_list`
    - `dgl.graph_index.from_scipy_sparse_matrix`
    - `dgl.graph_index.from_networkx`
    - `dgl.graph_index.from_shared_mem_csr_matrix`
    - `dgl.graph_index.from_csr`
    - `dgl.graph_index.from_coo`
    """

    def __new__(cls):
        obj = ObjectBase.__new__(cls)
        obj._readonly = None  # python-side cache of the flag
        obj._cache = {}
        return obj

    def __getstate__(self):
        src, dst, _ = self.edges()
        n_nodes = self.num_nodes()
        readonly = self.is_readonly()

        return n_nodes, readonly, src, dst

    def __setstate__(self, state):
        """The pickle state of GraphIndex is defined as a triplet
        (num_nodes, readonly, src_nodes, dst_nodes)
        """
        # Pickle compatibility check
        # TODO: we should store a storage version number in later releases.
        if isinstance(state, tuple) and len(state) == 5:
            dgl_warning(
                "The object is pickled pre-0.4.2.  Multigraph flag is ignored in 0.4.3"
            )
            num_nodes, _, readonly, src, dst = state
        elif isinstance(state, tuple) and len(state) == 4:
            # post-0.4.3.
            num_nodes, readonly, src, dst = state
        else:
            raise IOError("Unrecognized storage format.")

        self._cache = {}
        self._readonly = readonly
        self.__init_handle_by_constructor__(
            _CAPI_DGLGraphCreate,
            src.todgltensor(),
            dst.todgltensor(),
            int(num_nodes),
            readonly,
        )

    def add_nodes(self, num):
        """Add nodes.

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        """
        _CAPI_DGLGraphAddVertices(self, int(num))
        self.clear_cache()

    def add_edge(self, u, v):
        """Add one edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        _CAPI_DGLGraphAddEdge(self, int(u), int(v))
        self.clear_cache()

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
        _CAPI_DGLGraphAddEdges(self, u_array, v_array)
        self.clear_cache()

    def clear(self):
        """Clear the graph."""
        _CAPI_DGLGraphClear(self)
        self.clear_cache()

    def clear_cache(self):
        """Clear the cached graph structures."""
        self._cache.clear()

    def is_multigraph(self):
        """Return whether the graph is a multigraph
        The time cost will be O(E)

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        return bool(_CAPI_DGLGraphIsMultigraph(self))

    def is_readonly(self):
        """Indicate whether the graph index is read-only.

        Returns
        -------
        bool
            True if it is a read-only graph, False otherwise.
        """
        if self._readonly is None:
            self._readonly = bool(_CAPI_DGLGraphIsReadonly(self))
        return self._readonly

    def readonly(self, readonly_state=True):
        """Set the readonly state of graph index in-place.

        Parameters
        ----------
        readonly_state : bool
            New readonly state of current graph index.
        """
        # TODO(minjie): very ugly code, should fix this
        n_nodes, _, src, dst = self.__getstate__()
        self.clear_cache()
        state = (n_nodes, readonly_state, src, dst)
        self.__setstate__(state)

    def num_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes.
        """
        return _CAPI_DGLGraphNumVertices(self)

    def num_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges.
        """
        return _CAPI_DGLGraphNumEdges(self)

    # TODO(#5485): remove this method.
    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return _CAPI_DGLGraphNumVertices(self)

    # TODO(#5485): remove this method.
    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return _CAPI_DGLGraphNumEdges(self)

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists, False otherwise.
        """
        return bool(_CAPI_DGLGraphHasVertex(self, int(vid)))

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
        return utils.toindex(_CAPI_DGLGraphHasVertices(self, vid_array))

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
            True if the edge exists, False otherwise
        """
        return bool(_CAPI_DGLGraphHasEdgeBetween(self, int(u), int(v)))

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
        return utils.toindex(
            _CAPI_DGLGraphHasEdgesBetween(self, u_array, v_array)
        )

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
        return utils.toindex(
            _CAPI_DGLGraphPredecessors(self, int(v), int(radius))
        )

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
        return utils.toindex(
            _CAPI_DGLGraphSuccessors(self, int(v), int(radius))
        )

    def edge_id(self, u, v):
        """Return the id array of all edges between u and v.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        utils.Index
            The edge id array.
        """
        return utils.toindex(_CAPI_DGLGraphEdgeId(self, int(u), int(v)))

    def edge_ids(self, u, v):
        """Return a triplet of arrays that contains the edge IDs.

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
        edge_array = _CAPI_DGLGraphEdgeIds(self, u_array, v_array)

        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))

        return src, dst, eid

    def find_edge(self, eid):
        """Return the edge tuple of the given id.

        Parameters
        ----------
        eid : int
            The edge id.

        Returns
        -------
        int
            src node id
        int
            dst node id
        """
        ret = _CAPI_DGLGraphFindEdge(self, int(eid))
        return ret(0), ret(1)

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
        eid_array = eid.todgltensor()
        edge_array = _CAPI_DGLGraphFindEdges(self, eid_array)

        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))

        return src, dst, eid

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
            edge_array = _CAPI_DGLGraphInEdges_1(self, int(v[0]))
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphInEdges_2(self, v_array)
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
            edge_array = _CAPI_DGLGraphOutEdges_1(self, int(v[0]))
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphOutEdges_2(self, v_array)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    def sort_csr(self):
        """Sort the CSR matrix in the graph index.

        By default, when the CSR matrix is created, the edges may be stored
        in an arbitrary order. Sometimes, we want to sort them to accelerate
        some computation. For example, `has_edges_between` can be much faster
        on a giant adjacency matrix if the edges in the matrix is sorted.
        """
        _CAPI_DGLSortAdj(self)

    @utils.cached_member(cache="_cache", prefix="edges")
    def edges(self, order=None):
        """Return all the edges

        Parameters
        ----------
        order : string
            The order of the returned edges. Currently support:

            - 'srcdst' : sorted by their src and dst ids.
            - 'eid'    : sorted by edge Ids.
            - None     : the arbitrary order.

        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        if order is None:
            order = ""
        edge_array = _CAPI_DGLGraphEdges(self, order)
        src = edge_array(0)
        dst = edge_array(1)
        eid = edge_array(2)
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        eid = utils.toindex(eid)
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
        return _CAPI_DGLGraphInDegree(self, int(v))

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        tensor
            The in degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphInDegrees(self, v_array))

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
        return _CAPI_DGLGraphOutDegree(self, int(v))

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        tensor
            The out degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLGraphOutDegrees(self, v_array))

    def node_subgraph(self, v):
        """Return the induced node subgraph.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        v_array = v.todgltensor()
        return _CAPI_DGLGraphVertexSubgraph(self, v_array)

    def node_halo_subgraph(self, v, num_hops):
        """Return an induced subgraph with halo nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        num_hops : int
            The number of hops in which a HALO node can be accessed.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        DGLTensor
            Indicate if a node belongs to a partition.
        DGLTensor
            Indicate if an edge belongs to a partition.
        """
        v_array = v.todgltensor()
        subg = _CAPI_DGLGetSubgraphWithHalo(self, v_array, num_hops)
        inner_nodes = _CAPI_GetHaloSubgraphInnerNodes(subg)
        return subg, inner_nodes

    def node_subgraphs(self, vs_arr):
        """Return the induced node subgraphs.

        Parameters
        ----------
        vs_arr : a list of utils.Index
            The nodes.

        Returns
        -------
        a vector of SubgraphIndex
            The subgraph index.
        """
        gis = []
        for v in vs_arr:
            gis.append(self.node_subgraph(v))
        return gis

    def edge_subgraph(self, e, preserve_nodes=False):
        """Return the induced edge subgraph.

        Parameters
        ----------
        e : utils.Index
            The edges.
        preserve_nodes : bool
            Indicates whether to preserve all nodes or not.
            If true, keep the nodes which have no edge connected in the subgraph;
            If false, all nodes without edge connected to it would be removed.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        e_array = e.todgltensor()
        return _CAPI_DGLGraphEdgeSubgraph(self, e_array, preserve_nodes)

    @utils.cached_member(cache="_cache", prefix="scipy_adj")
    def adjacency_matrix_scipy(self, transpose, fmt, return_edge_ids=None):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        fmt : str
            Indicates the format of returned adjacency matrix.
        return_edge_ids : bool
            Indicates whether to return edge IDs or 1 as elements.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.
        """
        if not isinstance(transpose, bool):
            raise DGLError(
                'Expect bool value for "transpose" arg,'
                " but got %s." % (type(transpose))
            )

        if return_edge_ids is None:
            dgl_warning(
                "Adjacency matrix by default currently returns edge IDs."
                "  As a result there is one 0 entry which is not eliminated."
                "  In the next release it will return 1s by default,"
                " and 0 will be eliminated otherwise.",
                FutureWarning,
            )
            return_edge_ids = True

        rst = _CAPI_DGLGraphGetAdj(self, transpose, fmt)
        if fmt == "csr":
            indptr = utils.toindex(rst(0)).tonumpy()
            indices = utils.toindex(rst(1)).tonumpy()
            data = (
                utils.toindex(rst(2)).tonumpy()
                if return_edge_ids
                else np.ones_like(indices)
            )
            n = self.num_nodes()
            return scipy.sparse.csr_matrix(
                (data, indices, indptr), shape=(n, n)
            )
        elif fmt == "coo":
            idx = utils.toindex(rst(0)).tonumpy()
            n = self.num_nodes()
            m = self.num_edges()
            row, col = np.reshape(idx, (2, m))
            data = np.arange(0, m) if return_edge_ids else np.ones_like(row)
            return scipy.sparse.coo_matrix((data, (row, col)), shape=(n, n))
        else:
            raise Exception("unknown format")

    @utils.cached_member(cache="_cache", prefix="immu_gidx")
    def get_immutable_gidx(self, ctx):
        """Create an immutable graph index and copy to the given device context.

        Note: this internal function is for DGL scheduler use only

        Parameters
        ----------
        ctx : DGLContext
            The context of the returned graph.

        Returns
        -------
        GraphIndex
        """
        return self.to_immutable().asbits(self.bits_needed()).copy_to(ctx)

    def get_csr_shuffle_order(self):
        """Return the edge shuffling order when a coo graph is converted to csr format

        Returns
        -------
        tuple of two utils.Index
            The first element of the tuple is the shuffle order for outward graph
            The second element of the tuple is the shuffle order for inward graph
        """
        csr = _CAPI_DGLGraphGetAdj(self, True, "csr")
        order = csr(2)
        rev_csr = _CAPI_DGLGraphGetAdj(self, False, "csr")
        rev_order = rev_csr(2)
        return utils.toindex(order), utils.toindex(rev_order)

    def adjacency_matrix(self, transpose, ctx):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        ctx : context
            The context of the returned matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        utils.Index
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        if not isinstance(transpose, bool):
            raise DGLError(
                'Expect bool value for "transpose" arg,'
                " but got %s." % (type(transpose))
            )
        fmt = F.get_preferred_sparse_format()
        rst = _CAPI_DGLGraphGetAdj(self, transpose, fmt)
        if fmt == "csr":
            indptr = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            indices = F.copy_to(utils.toindex(rst(1)).tousertensor(), ctx)
            shuffle = utils.toindex(rst(2))
            dat = F.ones(indices.shape, dtype=F.float32, ctx=ctx)
            spmat = F.sparse_matrix(
                dat,
                ("csr", indices, indptr),
                (self.num_nodes(), self.num_nodes()),
            )[0]
            return spmat, shuffle
        elif fmt == "coo":
            ## FIXME(minjie): data type
            idx = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            m = self.num_edges()
            idx = F.reshape(idx, (2, m))
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            n = self.num_nodes()
            adj, shuffle_idx = F.sparse_matrix(dat, ("coo", idx), (n, n))
            shuffle_idx = (
                utils.toindex(shuffle_idx) if shuffle_idx is not None else None
            )
            return adj, shuffle_idx
        else:
            raise Exception("unknown format")

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
        src, dst, eid = self.edges()
        src = src.tousertensor(ctx)  # the index of the ctx will be cached
        dst = dst.tousertensor(ctx)  # the index of the ctx will be cached
        eid = eid.tousertensor(ctx)  # the index of the ctx will be cached
        n = self.num_nodes()
        m = self.num_edges()
        if typestr == "in":
            row = F.unsqueeze(dst, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ("coo", idx), (n, m))
        elif typestr == "out":
            row = F.unsqueeze(src, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ("coo", idx), (n, m))
        elif typestr == "both":
            # first remove entries for self loops
            mask = F.logical_not(F.equal(src, dst))
            src = F.boolean_mask(src, mask)
            dst = F.boolean_mask(dst, mask)
            eid = F.boolean_mask(eid, mask)
            n_entries = F.shape(src)[0]
            # create index
            row = F.unsqueeze(F.cat([src, dst], dim=0), 0)
            col = F.unsqueeze(F.cat([eid, eid], dim=0), 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            x = -F.ones((n_entries,), dtype=F.float32, ctx=ctx)
            y = F.ones((n_entries,), dtype=F.float32, ctx=ctx)
            dat = F.cat([x, y], dim=0)
            inc, shuffle_idx = F.sparse_matrix(dat, ("coo", idx), (n, m))
        else:
            raise DGLError("Invalid incidence matrix type: %s" % str(typestr))
        shuffle_idx = (
            utils.toindex(shuffle_idx) if shuffle_idx is not None else None
        )
        return inc, shuffle_idx

    def to_networkx(self):
        """Convert to networkx graph.

        The edge id will be saved as the 'id' edge attribute.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        src, dst, eid = self.edges()
        # xiangsx: Always treat graph as multigraph
        ret = nx.MultiDiGraph()
        ret.add_nodes_from(range(self.num_nodes()))
        for u, v, e in zip(src, dst, eid):
            ret.add_edge(u, v, id=e)
        return ret

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
        return _CAPI_DGLGraphLineGraph(self, backtracking)

    def to_immutable(self):
        """Convert this graph index to an immutable one.

        Returns
        -------
        GraphIndex
            An immutable graph index.
        """
        return _CAPI_DGLToImmutable(self)

    def ctx(self):
        """Return the context of this graph index.

        Returns
        -------
        DGLContext
            The context of the graph.
        """
        return _CAPI_DGLGraphContext(self)

    @property
    def dtype(self):
        """Return the index dtype

        Returns
        ----------
        str
            The dtype of graph index
        """
        bits = self.nbits()
        if bits == 32:
            return "int32"
        else:
            return "int64"

    def copy_to(self, ctx):
        """Copy this immutable graph index to the given device context.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        ctx : DGLContext
            The target device context.

        Returns
        -------
        GraphIndex
            The graph index on the given device context.
        """
        return _CAPI_DGLImmutableGraphCopyTo(
            self, ctx.device_type, ctx.device_id
        )

    def copyto_shared_mem(self, shared_mem_name):
        """Copy this immutable graph index to shared memory.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        shared_mem_name : string
            The name of the shared memory.

        Returns
        -------
        GraphIndex
            The graph index on the given device context.
        """
        return _CAPI_DGLImmutableGraphCopyToSharedMem(self, shared_mem_name)

    def nbits(self):
        """Return the number of integer bits used in the storage (32 or 64).

        Returns
        -------
        int
            The number of bits.
        """
        return _CAPI_DGLGraphNumBits(self)

    def bits_needed(self):
        """Return the number of integer bits needed to represent the graph

        Returns
        -------
        int
            The number of bits needed
        """
        if self.num_edges() >= 0x80000000 or self.num_nodes() >= 0x80000000:
            return 64
        else:
            return 32

    def asbits(self, bits):
        """Transform the graph to a new one with the given number of bits storage.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        bits : int
            The number of integer bits (32 or 64)

        Returns
        -------
        GraphIndex
            The graph index stored using the given number of bits.
        """
        return _CAPI_DGLImmutableGraphAsNumBits(self, int(bits))


@register_object("graph.Subgraph")
class SubgraphIndex(ObjectBase):
    """Subgraph data structure"""

    @property
    def graph(self):
        """The subgraph structure

        Returns
        -------
        GraphIndex
            The subgraph
        """
        return _CAPI_DGLSubgraphGetGraph(self)

    @property
    def induced_nodes(self):
        """Induced nodes for each node type. The return list
        length should be equal to the number of node types.

        Returns
        -------
        list of utils.Index
            Induced nodes
        """
        ret = _CAPI_DGLSubgraphGetInducedVertices(self)
        return utils.toindex(ret)

    @property
    def induced_edges(self):
        """Induced edges for each edge type. The return list
        length should be equal to the number of edge types.

        Returns
        -------
        list of utils.Index
            Induced edges
        """
        ret = _CAPI_DGLSubgraphGetInducedEdges(self)
        return utils.toindex(ret)


###############################################################
# Conversion functions
###############################################################
def from_coo(num_nodes, src, dst, readonly):
    """Convert from coo arrays.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    src : Tensor
        Src end nodes of the edges.
    dst : Tensor
        Dst end nodes of the edges.
    readonly : bool
        True if the returned graph is readonly.

    Returns
    -------
    GraphIndex
        The graph index.
    """
    src = utils.toindex(src)
    dst = utils.toindex(dst)
    if readonly:
        gidx = _CAPI_DGLGraphCreate(
            src.todgltensor(), dst.todgltensor(), int(num_nodes), readonly
        )
    else:
        gidx = _CAPI_DGLGraphCreateMutable()
        gidx.add_nodes(num_nodes)
        gidx.add_edges(src, dst)
    return gidx


def from_csr(indptr, indices, direction):
    """Load a graph from CSR arrays.

    Parameters
    ----------
    indptr : Tensor
        index pointer in the CSR format
    indices : Tensor
        column index array in the CSR format
    direction : str

    Returns
    ------
    GraphIndex
        The graph index
        the edge direction. Either "in" or "out".
    """
    indptr = utils.toindex(indptr)
    indices = utils.toindex(indices)
    gidx = _CAPI_DGLGraphCSRCreate(
        indptr.todgltensor(), indices.todgltensor(), direction
    )
    return gidx


def from_shared_mem_graph_index(shared_mem_name):
    """Load a graph index from the shared memory.

    Parameters
    ----------
    shared_mem_name : string
        the name of shared memory

    Returns
    ------
    GraphIndex
        The graph index
    """
    return _CAPI_DGLGraphCSRCreateMMap(shared_mem_name)


def from_networkx(nx_graph, readonly):
    """Convert from networkx graph.

    If 'id' edge attribute exists, the edge will be added follows
    the edge id order. Otherwise, order is undefined.

    Parameters
    ----------
    nx_graph : networkx.DiGraph
        The nx graph or any graph that can be converted to nx.DiGraph
    readonly : bool
        True if the returned graph is readonly.

    Returns
    -------
    GraphIndex
        The graph index.
    """
    if not isinstance(nx_graph, nx.Graph):
        nx_graph = nx.DiGraph(nx_graph)
    else:
        if not nx_graph.is_directed():
            # to_directed creates a deep copy of the networkx graph even if
            # the original graph is already directed and we do not want to do it.
            nx_graph = nx_graph.to_directed()
    num_nodes = nx_graph.number_of_nodes()

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    if nx_graph.number_of_edges() > 0:
        has_edge_id = "id" in next(iter(nx_graph.edges(data=True)))[-1]
    else:
        has_edge_id = False

    if has_edge_id:
        num_edges = nx_graph.number_of_edges()
        src = np.zeros((num_edges,), dtype=np.int64)
        dst = np.zeros((num_edges,), dtype=np.int64)
        for u, v, attr in nx_graph.edges(data=True):
            eid = attr["id"]
            src[eid] = u
            dst[eid] = v
    else:
        src = []
        dst = []
        for e in nx_graph.edges:
            src.append(e[0])
            dst.append(e[1])
    num_nodes = nx_graph.number_of_nodes()
    # We store edge Ids as an edge attribute.
    src = utils.toindex(src)
    dst = utils.toindex(dst)
    return from_coo(num_nodes, src, dst, readonly)


def from_scipy_sparse_matrix(adj, readonly):
    """Convert from scipy sparse matrix.

    Parameters
    ----------
    adj : scipy sparse matrix
    readonly : bool
        True if the returned graph is readonly.

    Returns
    -------
    GraphIndex
        The graph index.
    """
    if adj.getformat() != "csr" or not readonly:
        num_nodes = max(adj.shape[0], adj.shape[1])
        adj_coo = adj.tocoo()
        return from_coo(num_nodes, adj_coo.row, adj_coo.col, readonly)
    else:
        # If the input matrix is csr, we still treat it as multigraph.
        return from_csr(adj.indptr, adj.indices, "out")


def from_edge_list(elist, readonly):
    """Convert from an edge list.

    Parameters
    ---------
    elist : list, tuple
        List of (u, v) edge tuple, or a tuple of src/dst lists
    """
    if isinstance(elist, tuple):
        src, dst = elist
    else:
        src, dst = zip(*elist)
    src = np.asarray(src)
    dst = np.asarray(dst)
    src_ids = utils.toindex(src)
    dst_ids = utils.toindex(dst)
    num_nodes = max(src.max(), dst.max()) + 1
    return from_coo(num_nodes, src_ids, dst_ids, readonly)


def map_to_subgraph_nid(induced_nodes, parent_nids):
    """Map parent node Ids to the subgraph node Ids.

    Parameters
    ----------
    induced_nodes: utils.Index
        Induced nodes of the subgraph.

    parent_nids: utils.Index
        Node Ids in the parent graph.

    Returns
    -------
    utils.Index
        Node Ids in the subgraph.
    """
    return utils.toindex(
        _CAPI_DGLMapSubgraphNID(
            induced_nodes.todgltensor(), parent_nids.todgltensor()
        )
    )


def transform_ids(mapping, ids):
    """Transform ids by the given mapping.

    Parameters
    ----------
    mapping : utils.Index
        The id mapping. new_id = mapping[old_id]
    ids : utils.Index
        The old ids.

    Returns
    -------
    utils.Index
        The new ids.
    """
    return utils.toindex(
        _CAPI_DGLMapSubgraphNID(mapping.todgltensor(), ids.todgltensor())
    )


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
    return _CAPI_DGLDisjointUnion(list(graphs))


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
            graph, num_or_size_splits.todgltensor()
        )
    else:
        rst = _CAPI_DGLDisjointPartitionByNum(graph, int(num_or_size_splits))
    return rst


def create_graph_index(graph_data, readonly):
    """Create a graph index object.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    readonly : bool
        Whether the graph structure is read-only.
    """
    if isinstance(graph_data, GraphIndex):
        # FIXME(minjie): this return is not correct for mutable graph index
        return graph_data

    if graph_data is None:
        if readonly:
            raise Exception("can't create an empty immutable graph")
        return _CAPI_DGLGraphCreateMutable()
    elif isinstance(graph_data, (list, tuple)):
        # edge list
        return from_edge_list(graph_data, readonly)
    elif isinstance(graph_data, scipy.sparse.spmatrix):
        # scipy format
        return from_scipy_sparse_matrix(graph_data, readonly)
    else:
        # networkx - any format
        try:
            gidx = from_networkx(graph_data, readonly)
        except Exception:  # pylint: disable=broad-except
            raise DGLError(
                'Error while creating graph from input of type "%s".'
                % type(graph_data)
            )
        return gidx


def _get_halo_subgraph_inner_node(halo_subg):
    return _CAPI_GetHaloSubgraphInnerNodes(halo_subg)


_init_api("dgl.graph_index")
