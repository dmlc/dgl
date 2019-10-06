"""Module for heterogeneous graph index class definition."""
from __future__ import absolute_import

import numpy as np
import scipy

from ._ffi.object import register_object, ObjectBase
from ._ffi.function import _init_api
from .base import DGLError, dgl_warning
from . import backend as F
from . import utils

@register_object('graph.HeteroGraph')
class HeteroGraphIndex(ObjectBase):
    """HeteroGraph index object.

    Note
    ----
    Do not create GraphIndex directly.
    """
    def __new__(cls):
        obj = ObjectBase.__new__(cls)
        obj._cache = {}
        return obj

    def __getstate__(self):
        # TODO
        return

    def __setstate__(self, state):
        # TODO
        pass

    @property
    def metagraph(self):
        """Meta graph

        Returns
        -------
        GraphIndex
            The meta graph.
        """
        return _CAPI_DGLHeteroGetMetaGraph(self)

    def number_of_ntypes(self):
        """Return number of node types."""
        return self.metagraph.number_of_nodes()

    def number_of_etypes(self):
        """Return number of edge types."""
        return self.metagraph.number_of_edges()

    def get_relation_graph(self, etype):
        """Get the unitgraph graph of the given edge/relation type.

        Parameters
        ----------
        etype : int
            The edge/relation type.

        Returns
        -------
        HeteroGraphIndex
            The unitgraph graph.
        """
        return _CAPI_DGLHeteroGetRelationGraph(self, int(etype))

    def flatten_relations(self, etypes):
        """Convert the list of requested unitgraph graphs into a single unitgraph
        graph.

        Parameters
        ----------
        etypes : list[int]
            The edge/relation types.

        Returns
        -------
        FlattenedHeteroGraph
            A flattened heterograph object
        """
        return _CAPI_DGLHeteroGetFlattenedGraph(self, etypes)

    def add_nodes(self, ntype, num):
        """Add nodes.

        Parameters
        ----------
        ntype : int
            Node type
        num : int
            Number of nodes to be added.
        """
        _CAPI_DGLHeteroAddVertices(self, int(ntype), int(num))
        self.clear_cache()

    def add_edge(self, etype, u, v):
        """Add one edge.

        Parameters
        ----------
        etype : int
            Edge type
        u : int
            The src node.
        v : int
            The dst node.
        """
        _CAPI_DGLHeteroAddEdge(self, int(etype), int(u), int(v))
        self.clear_cache()

    def add_edges(self, etype, u, v):
        """Add many edges.

        Parameters
        ----------
        etype : int
            Edge type
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        _CAPI_DGLHeteroAddEdges(self, int(etype), u.todgltensor(), v.todgltensor())
        self.clear_cache()

    def clear(self):
        """Clear the graph."""
        _CAPI_DGLHeteroClear(self)
        self._cache.clear()

    def ctx(self):
        """Return the context of this graph index.

        Returns
        -------
        DGLContext
            The context of the graph.
        """
        return _CAPI_DGLHeteroContext(self)

    def nbits(self):
        """Return the number of integer bits used in the storage (32 or 64).

        Returns
        -------
        int
            The number of bits.
        """
        return _CAPI_DGLHeteroNumBits(self)

    def bits_needed(self, etype):
        """Return the number of integer bits needed to represent the unitgraph graph.

        Parameters
        ----------
        etype : int
            The edge type.

        Returns
        -------
        int
            The number of bits needed.
        """
        stype, dtype = self.metagraph.find_edge(etype)
        if (self.number_of_edges(etype) >= 0x80000000 or
                self.number_of_nodes(stype) >= 0x80000000 or
                self.number_of_nodes(dtype) >= 0x80000000):
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
        HeteroGraphIndex
            The graph index stored using the given number of bits.
        """
        return _CAPI_DGLHeteroAsNumBits(self, int(bits))

    def copy_to(self, ctx):
        """Copy this immutable graph index to the given device context.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        ctx : DGLContext
            The target device context.

        Returns
        -------
        HeteroGraphIndex
            The graph index on the given device context.
        """
        return _CAPI_DGLHeteroCopyTo(self, ctx.device_type, ctx.device_id)

    def is_multigraph(self):
        """Return whether the graph is a multigraph

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        return bool(_CAPI_DGLHeteroIsMultigraph(self))

    def is_readonly(self):
        """Return whether the graph index is read-only.

        Returns
        -------
        bool
            True if it is a read-only graph, False otherwise.
        """
        return bool(_CAPI_DGLHeteroIsReadonly(self))

    def number_of_nodes(self, ntype):
        """Return the number of nodes.

        Parameters
        ----------
        ntype : int
            Node type

        Returns
        -------
        int
            The number of nodes
        """
        return _CAPI_DGLHeteroNumVertices(self, int(ntype))

    def number_of_edges(self, etype):
        """Return the number of edges.

        Parameters
        ----------
        etype : int
            Edge type

        Returns
        -------
        int
            The number of edges
        """
        return _CAPI_DGLHeteroNumEdges(self, int(etype))

    def has_node(self, ntype, vid):
        """Return true if the node exists.

        Parameters
        ----------
        ntype : int
            Node type
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists, False otherwise.
        """
        return bool(_CAPI_DGLHeteroHasVertex(self, int(ntype), int(vid)))

    def has_nodes(self, ntype, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        ntype : int
            Node type
        vid : utils.Index
            The nodes

        Returns
        -------
        utils.Index
            0-1 array indicating existence
        """
        vid_array = vids.todgltensor()
        return utils.toindex(_CAPI_DGLHeteroHasVertices(self, int(ntype), vid_array))

    def has_edge_between(self, etype, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        etype : int
            Edge type
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists, False otherwise
        """
        return bool(_CAPI_DGLHeteroHasEdgeBetween(self, int(etype), int(u), int(v)))

    def has_edges_between(self, etype, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        etype : int
            Edge type
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
        return utils.toindex(_CAPI_DGLHeteroHasEdgesBetween(
            self, int(etype), u_array, v_array))

    def predecessors(self, etype, v):
        """Return the predecessors of the node.

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        utils.Index
            Array of predecessors
        """
        return utils.toindex(_CAPI_DGLHeteroPredecessors(
            self, int(etype), int(v)))

    def successors(self, etype, v):
        """Return the successors of the node.

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        utils.Index
            Array of successors
        """
        return utils.toindex(_CAPI_DGLHeteroSuccessors(
            self, int(etype), int(v)))

    def edge_id(self, etype, u, v):
        """Return the id array of all edges between u and v.

        Parameters
        ----------
        etype : int
            Edge type
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        utils.Index
            The edge id array.
        """
        return utils.toindex(_CAPI_DGLHeteroEdgeId(
            self, int(etype), int(u), int(v)))

    def edge_ids(self, etype, u, v):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
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
        edge_array = _CAPI_DGLHeteroEdgeIds(self, int(etype), u_array, v_array)

        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))

        return src, dst, eid

    def find_edges(self, etype, eid):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
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
        edge_array = _CAPI_DGLHeteroFindEdges(self, int(etype), eid_array)

        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))

        return src, dst, eid

    def in_edges(self, etype, v):
        """Return the in edges of the node(s).

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
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
            edge_array = _CAPI_DGLHeteroInEdges_1(self, int(etype), int(v[0]))
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLHeteroInEdges_2(self, int(etype), v_array)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    def out_edges(self, etype, v):
        """Return the out edges of the node(s).

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
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
            edge_array = _CAPI_DGLHeteroOutEdges_1(self, int(etype), int(v[0]))
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLHeteroOutEdges_2(self, int(etype), v_array)
        src = utils.toindex(edge_array(0))
        dst = utils.toindex(edge_array(1))
        eid = utils.toindex(edge_array(2))
        return src, dst, eid

    @utils.cached_member(cache='_cache', prefix='edges')
    def edges(self, etype, order=None):
        """Return all the edges

        Parameters
        ----------
        etype : int
            Edge type
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
        edge_array = _CAPI_DGLHeteroEdges(self, int(etype), order)
        src = edge_array(0)
        dst = edge_array(1)
        eid = edge_array(2)
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        eid = utils.toindex(eid)
        return src, dst, eid

    def in_degree(self, etype, v):
        """Return the in degree of the node.

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return _CAPI_DGLHeteroInDegree(self, int(etype), int(v))

    def in_degrees(self, etype, v):
        """Return the in degrees of the nodes.

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The in degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLHeteroInDegrees(self, int(etype), v_array))

    def out_degree(self, etype, v):
        """Return the out degree of the node.

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return _CAPI_DGLHeteroOutDegree(self, int(etype), int(v))

    def out_degrees(self, etype, v):
        """Return the out degrees of the nodes.

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The out degree array.
        """
        v_array = v.todgltensor()
        return utils.toindex(_CAPI_DGLHeteroOutDegrees(self, int(etype), v_array))

    def adjacency_matrix(self, etype, transpose, ctx):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        etype : int
            Edge type
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
            raise DGLError('Expect bool value for "transpose" arg,'
                           ' but got %s.' % (type(transpose)))
        fmt = F.get_preferred_sparse_format()
        rst = _CAPI_DGLHeteroGetAdj(self, int(etype), transpose, fmt)
        # convert to framework-specific sparse matrix
        srctype, dsttype = self.metagraph.find_edge(etype)
        nrows = self.number_of_nodes(srctype) if transpose else self.number_of_nodes(dsttype)
        ncols = self.number_of_nodes(dsttype) if transpose else self.number_of_nodes(srctype)
        nnz = self.number_of_edges(etype)
        if fmt == "csr":
            indptr = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            indices = F.copy_to(utils.toindex(rst(1)).tousertensor(), ctx)
            shuffle = utils.toindex(rst(2))
            dat = F.ones(nnz, dtype=F.float32, ctx=ctx)  # FIXME(minjie): data type
            spmat = F.sparse_matrix(dat, ('csr', indices, indptr), (nrows, ncols))[0]
            return spmat, shuffle
        elif fmt == "coo":
            idx = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            idx = F.reshape(idx, (2, nnz))
            dat = F.ones((nnz,), dtype=F.float32, ctx=ctx)
            adj, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (nrows, ncols))
            shuffle_idx = utils.toindex(shuffle_idx) if shuffle_idx is not None else None
            return adj, shuffle_idx
        else:
            raise Exception("unknown format")

    def adjacency_matrix_scipy(self, etype, transpose, fmt, return_edge_ids=None):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        etype : int
            Edge type
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
            raise DGLError('Expect bool value for "transpose" arg,'
                           ' but got %s.' % (type(transpose)))

        if return_edge_ids is None:
            dgl_warning(
                "Adjacency matrix by default currently returns edge IDs."
                "  As a result there is one 0 entry which is not eliminated."
                "  In the next release it will return 1s by default,"
                " and 0 will be eliminated otherwise.",
                FutureWarning)
            return_edge_ids = True

        rst = _CAPI_DGLHeteroGetAdj(self, int(etype), transpose, fmt)
        srctype, dsttype = self.metagraph.find_edge(etype)
        nrows = self.number_of_nodes(srctype) if transpose else self.number_of_nodes(dsttype)
        ncols = self.number_of_nodes(dsttype) if transpose else self.number_of_nodes(srctype)
        nnz = self.number_of_edges(etype)
        if fmt == "csr":
            indptr = utils.toindex(rst(0)).tonumpy()
            indices = utils.toindex(rst(1)).tonumpy()
            data = utils.toindex(rst(2)).tonumpy() if return_edge_ids else np.ones_like(indices)
            return scipy.sparse.csr_matrix((data, indices, indptr), shape=(nrows, ncols))
        elif fmt == 'coo':
            idx = utils.toindex(rst(0)).tonumpy()
            row, col = np.reshape(idx, (2, nnz))
            data = np.arange(0, nnz) if return_edge_ids else np.ones_like(row)
            return scipy.sparse.coo_matrix((data, (row, col)), shape=(nrows, ncols))
        else:
            raise Exception("unknown format")

    def incidence_matrix(self, etype, typestr, ctx):
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
        etype : int
            Edge type
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
        src, dst, eid = self.edges(etype)
        src = src.tousertensor(ctx)  # the index of the ctx will be cached
        dst = dst.tousertensor(ctx)  # the index of the ctx will be cached
        eid = eid.tousertensor(ctx)  # the index of the ctx will be cached
        srctype, dsttype = self.metagraph.find_edge(etype)

        m = self.number_of_edges(etype)
        if typestr == 'in':
            n = self.number_of_nodes(dsttype)
            row = F.unsqueeze(dst, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        elif typestr == 'out':
            n = self.number_of_nodes(srctype)
            row = F.unsqueeze(src, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        elif typestr == 'both':
            assert srctype == dsttype, \
                    "'both' is supported only if source and destination type are the same"
            n = self.number_of_nodes(srctype)
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
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        else:
            raise DGLError('Invalid incidence matrix type: %s' % str(typestr))
        shuffle_idx = utils.toindex(shuffle_idx) if shuffle_idx is not None else None
        return inc, shuffle_idx

    def node_subgraph(self, induced_nodes):
        """Return the induced node subgraph.

        Parameters
        ----------
        induced_nodes : list of utils.Index
            Induced nodes. The length should be equal to the number of
            node types in this heterograph.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        vids = [nodes.todgltensor() for nodes in induced_nodes]
        return _CAPI_DGLHeteroVertexSubgraph(self, vids)

    def edge_subgraph(self, induced_edges, preserve_nodes):
        """Return the induced edge subgraph.

        Parameters
        ----------
        induced_edges : list of utils.Index
            Induced edges. The length should be equal to the number of
            edge types in this heterograph.
        preserve_nodes : bool
            Indicates whether to preserve all nodes or not.
            If true, keep the nodes which have no edge connected in the subgraph;
            If false, all nodes without edge connected to it would be removed.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        eids = [edges.todgltensor() for edges in induced_edges]
        return _CAPI_DGLHeteroEdgeSubgraph(self, eids, preserve_nodes)

    @utils.cached_member(cache='_cache', prefix='unitgraph')
    def get_unitgraph(self, etype, ctx):
        """Create a unitgraph graph from given edge type and copy to the given device
        context.

        Note: this internal function is for DGL scheduler use only

        Parameters
        ----------
        etype : int
            If the graph index is a Bipartite graph index, this argument must be None.
            Otherwise, it represents the edge type.
        ctx : DGLContext
            The context of the returned graph.

        Returns
        -------
        HeteroGraphIndex
        """
        g = self.get_relation_graph(etype)
        return g.asbits(self.bits_needed(etype or 0)).copy_to(ctx)

    def get_csr_shuffle_order(self, etype):
        """Return the edge shuffling order when a coo graph is converted to csr format

        Parameters
        ----------
        etype : int
            The edge type

        Returns
        -------
        tuple of two utils.Index
            The first element of the tuple is the shuffle order for outward graph
            The second element of the tuple is the shuffle order for inward graph
        """
        csr = _CAPI_DGLHeteroGetAdj(self, int(etype), True, "csr")
        order = csr(2)
        rev_csr = _CAPI_DGLHeteroGetAdj(self, int(etype), False, "csr")
        rev_order = rev_csr(2)
        return utils.toindex(order), utils.toindex(rev_order)

@register_object('graph.HeteroSubgraph')
class HeteroSubgraphIndex(ObjectBase):
    """Hetero-subgraph data structure"""
    @property
    def graph(self):
        """The subgraph structure

        Returns
        -------
        HeteroGraphIndex
            The subgraph
        """
        return _CAPI_DGLHeteroSubgraphGetGraph(self)

    @property
    def induced_nodes(self):
        """Induced nodes for each node type. The return list
        length should be equal to the number of node types.

        Returns
        -------
        list of utils.Index
            Induced nodes
        """
        ret = _CAPI_DGLHeteroSubgraphGetInducedVertices(self)
        return [utils.toindex(v.data) for v in ret]

    @property
    def induced_edges(self):
        """Induced edges for each edge type. The return list
        length should be equal to the number of edge types.

        Returns
        -------
        list of utils.Index
            Induced edges
        """
        ret = _CAPI_DGLHeteroSubgraphGetInducedEdges(self)
        return [utils.toindex(v.data) for v in ret]

#################################################################
# Creators
#################################################################

def create_unitgraph_from_coo(num_ntypes, num_src, num_dst, row, col):
    """Create a unitgraph graph index from COO format

    Parameters
    ----------
    num_ntypes : int
        Number of node types (must be 1 or 2).
    num_src : int
        Number of nodes in the src type.
    num_dst : int
        Number of nodes in the dst type.
    row : utils.Index
        Row index.
    col : utils.Index
        Col index.

    Returns
    -------
    HeteroGraphIndex
    """
    return _CAPI_DGLHeteroCreateUnitGraphFromCOO(
        int(num_ntypes), int(num_src), int(num_dst), row.todgltensor(), col.todgltensor())

def create_unitgraph_from_csr(num_ntypes, num_src, num_dst, indptr, indices, edge_ids):
    """Create a unitgraph graph index from CSR format

    Parameters
    ----------
    num_ntypes : int
        Number of node types (must be 1 or 2).
    num_src : int
        Number of nodes in the src type.
    num_dst : int
        Number of nodes in the dst type.
    indptr : utils.Index
        CSR indptr.
    indices : utils.Index
        CSR indices.
    edge_ids : utils.Index
        Edge shuffle id.

    Returns
    -------
    HeteroGraphIndex
    """
    return _CAPI_DGLHeteroCreateUnitGraphFromCSR(
        int(num_ntypes), int(num_src), int(num_dst),
        indptr.todgltensor(), indices.todgltensor(), edge_ids.todgltensor())

def create_heterograph_from_relations(metagraph, rel_graphs):
    """Create a heterograph from metagraph and graphs of every relation.

    Parameters
    ----------
    metagraph : GraphIndex
        Meta-graph.
    rel_graphs : list of HeteroGraphIndex
        Bipartite graph of each relation.

    Returns
    -------
    HeteroGraphIndex
    """
    return _CAPI_DGLHeteroCreateHeteroGraph(metagraph, rel_graphs)

@register_object("graph.FlattenedHeteroGraph")
class FlattenedHeteroGraph(ObjectBase):
    """FlattenedHeteroGraph object class in C++ backend."""

_init_api("dgl.heterograph_index")
