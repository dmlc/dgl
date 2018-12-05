from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp
import mxnet as mx

class ImmutableGraphIndex(object):
    """Backend-specific graph index object on immutable graphs.
    We can use a CSR matrix to represent a graph structure. For functionality,
    one CSR matrix is sufficient. However, for efficient access
    to in-edges and out-edges of a directed graph, we need to use two CSR matrices.
    In these CSR matrices, both rows and columns represent vertices. In one CSR
    matrix, a row stores in-edges of a vertex (whose source vertex is a neighbor
    and destination vertex is the vertex itself). Thus, a non-zero entry is
    the neighbor Id and the value is the corresponding edge Id.
    The other CSR matrix stores the out-edges in the same fashion.

    Parameters
    ----------
    in_csr : a csr array that stores in-edges.
        MXNet CSRArray
    out_csr : a csr array that stores out-edges.
        MXNet CSRArray
    """
    def __init__(self, in_csr, out_csr):
        self._in_csr = in_csr
        self._out_csr = out_csr
        self._cached_adj = {}

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
        return self._in_csr.indices.shape[0]

    def has_edges(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : NDArray
            The src nodes.
        v : NDArray
            The dst nodes.

        Returns
        -------
        NDArray
            0-1 array indicating existence
        """
        ids = mx.nd.contrib.edge_id(self._in_csr, v, u)
        return ids >= 0

    def edge_ids(self, u, v):
        """Return the edge ids.

        Parameters
        ----------
        u : NDArray
            The src nodes.
        v : NDArray
            The dst nodes.

        Returns
        -------
        NDArray
            Teh edge id array.
        """
        if len(u) == 0 or len(v) == 0:
            return [], [], []
        ids = mx.nd.contrib.edge_id(self._in_csr, v, u)
        ids = ids.asnumpy()
        v = v.asnumpy()
        u = u.asnumpy()
        return u[ids >= 0], v[ids >= 0], ids[ids >= 0]

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
        NDArray
            Array of predecessors
        """
        if radius > 1:
            raise Exception('Immutable graph doesn\'t support predecessors with radius > 1 for now.')
        return self._in_csr[v].indices

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
        NDArray
            Array of successors
        """
        if radius > 1:
            raise Exception('Immutable graph doesn\'t support successors with radius > 1 for now.')
        return self._out_csr[v].indices

    def in_edges(self, v):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : NDArray
            The node(s).
        
        Returns
        -------
        NDArray
            index pointers
        NDArray
            The src nodes.
        NDArray
            The edge ids.
        """
        rows = mx.nd.take(self._in_csr, v)
        return rows.indptr, rows.indices, rows.data

    def out_edges(self, v):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : NDArray
            The node(s).
        
        Returns
        -------
        NDArray
            index pointers
        NDArray
            The dst nodes.
        NDArray
            The edge ids.
        """
        rows = mx.nd.take(self._out_csr, v)
        return rows.indptr, rows.indices, rows.data

    def edges(self, sorted=False):
        """Return all the edges

        Parameters
        ----------
        sorted : bool
            True if the returned edges are sorted by their src and dst ids.
        
        Returns
        -------
        NDArray
            The src nodes.
        NDArray
            The dst nodes.
        NDArray
            The edge ids.
        """
        #TODO(zhengda) we need to return NDArray directly
        # We don't need to take care of the sorted flag because the vertex Ids
        # are already sorted.
        coo = self._in_csr.asscipy().tocoo()
        return coo.col, coo.row, coo.data

    def get_in_degree(self):
        """Return the in degrees of all nodes.

        Returns
        -------
        NDArray
            degrees
        """
        return mx.nd.contrib.getnnz(self._in_csr, axis=1)

    def get_out_degree(self):
        """Return the out degrees of all nodes.

        Returns
        -------
        NDArray
            degrees
        """
        return mx.nd.contrib.getnnz(self._out_csr, axis=1)

    def node_subgraph(self, v):
        """Return the induced node subgraph.

        Parameters
        ----------
        v : NDArray
            The nodes.

        Returns
        -------
        ImmutableGraphIndex
            The subgraph index.
        NDArray
            Induced nodes
        NDArray
            Induced edges
        """
        v = mx.nd.sort(v)
        # when return_mapping is turned on, dgl_subgraph returns another CSRArray that
        # stores the edge Ids of the original graph.
        csr = mx.nd.contrib.dgl_subgraph(self._in_csr, v, return_mapping=True)
        induced_nodes = v
        induced_edges = lambda: csr[1].data
        return ImmutableGraphIndex(csr[0], None), induced_nodes, induced_edges

    def node_subgraphs(self, vs_arr):
        """Return the induced node subgraphs.

        Parameters
        ----------
        vs_arr : a vector of NDArray
            The nodes.

        Returns
        -------
        a vector of ImmutableGraphIndex
            The subgraph index.
        a vector of NDArrays
            Induced nodes of subgraphs.
        a vector of NDArrays
            Induced edges of subgraphs.
        """
        vs_arr = [mx.nd.sort(v) for v in vs_arr]
        res = mx.nd.contrib.dgl_subgraph(self._in_csr, *vs_arr, return_mapping=True)
        in_csrs = res[0:len(vs_arr)]
        induced_nodes = vs_arr
        induced_edges = [lambda: e.data for e in res[len(vs_arr):]]
        assert len(in_csrs) == len(induced_nodes)
        assert len(in_csrs) == len(induced_edges)
        gis = []
        induced_ns = []
        induced_es = []
        for in_csr, induced_n, induced_e in zip(in_csrs, induced_nodes, induced_edges):
            gis.append(ImmutableGraphIndex(in_csr, None))
            induced_ns.append(induced_n)
            induced_es.append(induced_e)
        return gis, induced_ns, induced_es

    def neighbor_sampling(self, seed_ids, expand_factor, num_hops, neighbor_type,
                          node_prob, max_subgraph_size):
        if neighbor_type == 'in':
            g = self._in_csr
        elif neighbor_type == 'out':
            g = self._out_csr
        else:
            raise NotImplementedError
        num_nodes = []
        num_subgs = len(seed_ids)
        if node_prob is None:
            res = mx.nd.contrib.dgl_csr_neighbor_uniform_sample(g, *seed_ids, num_hops=num_hops,
                                                                num_neighbor=expand_factor,
                                                                max_num_vertices=max_subgraph_size)
        else:
            res = mx.nd.contrib.dgl_csr_neighbor_non_uniform_sample(g, node_prob, *seed_ids, num_hops=num_hops,
                                                                    num_neighbor=expand_factor,
                                                                    max_num_vertices=max_subgraph_size)

        vertices, subgraphs = res[0:num_subgs], res[num_subgs:(2*num_subgs)]
        num_nodes = [subg_v[-1].asnumpy()[0] for subg_v in vertices]

        inputs = []
        inputs.extend(subgraphs)
        inputs.extend(vertices)
        compacts = mx.nd.contrib.dgl_graph_compact(*inputs, graph_sizes=num_nodes, return_mapping=False)

        if isinstance(compacts, mx.nd.sparse.CSRNDArray):
            compacts = [compacts]
        if neighbor_type == 'in':
            gis = [ImmutableGraphIndex(csr, None) for csr in compacts]
        elif neighbor_type == 'out':
            gis = [ImmutableGraphIndex(None, csr) for csr in compacts]
        parent_nodes = [v[0:size] for v, size in zip(vertices, num_nodes)]
        parent_edges = [lambda: e.data for e in subgraphs]
        return gis, parent_nodes, parent_edges

    def adjacency_matrix(self, transpose, ctx):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool
            A flag to tranpose the returned adjacency matrix.
        ctx : context
            The device context of the returned matrix.

        Returns
        -------
        NDArray
            An object that returns tensor given context.
        """
        if transpose:
            mat = self._out_csr
        else:
            mat = self._in_csr
        return mx.nd.contrib.dgl_adjacency(mat.as_in_context(ctx))

    def from_coo_matrix(self, out_coo):
        """construct the graph index from a SciPy coo matrix.

        Parameters
        ----------
        out_coo : SciPy coo matrix
            The non-zero entries indicate out-edges of the graph.
        """
        edge_ids = mx.nd.arange(0, len(out_coo.data), step=1, repeat=1, dtype=np.int32)
        src = mx.nd.array(out_coo.row, dtype=np.int64)
        dst = mx.nd.array(out_coo.col, dtype=np.int64)
        # TODO we can't generate a csr_matrix with np.int64 directly.
        size = max(out_coo.shape)
        self.__init__(mx.nd.sparse.csr_matrix((edge_ids, (dst, src)), shape=(size, size)).astype(np.int64),
                mx.nd.sparse.csr_matrix((edge_ids, (src, dst)), shape=(size, size)).astype(np.int64))

def create_immutable_graph_index(in_csr=None, out_csr=None):
    """ Create an empty backend-specific immutable graph index.

    Parameters
    ----------
    in_csr : MXNet CSRNDArray
        The in-edge CSR array.
    out_csr : MXNet CSRNDArray
        The out-edge CSR array.

    Returns
    -------
    ImmutableGraphIndex
        The backend-specific immutable graph index.
    """
    if in_csr is not None and not isinstance(in_csr, mx.nd.sparse.CSRNDArray):
        raise TypeError()
    if out_csr is not None and not isinstance(out_csr, mx.nd.sparse.CSRNDArray):
        raise TypeError()
    return ImmutableGraphIndex(in_csr, out_csr)
