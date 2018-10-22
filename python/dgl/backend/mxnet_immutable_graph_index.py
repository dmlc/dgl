from __future__ import absolute_import

import ctypes
import numpy as np
import networkx as nx
import scipy.sparse as sp
import mxnet as mx

from .mxnet import to_context

class ImmutableGraphIndex(object):
    """Backend-specific graph index object on immutable graphs.

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
        #TODO do we need to sort the array.
        #TODO we need to return NDArray directly
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
        induced_edges = csr[1].data
        return ImmutableGraphIndex(csr[0], None), induced_nodes, induced_edges

    def node_subgraphs(self, vs_arr):
        """Return the induced node subgraphs.

        Parameters
        ----------
        vs_arr : a vector of utils.Index
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
        induced_edges = [e.data for e in res[len(vs_arr):]]
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
        NDArray
            An object that returns tensor given context.
        """
        if edge_type == 'in':
            mat = self._in_csr
        else:
            mat = self._out_csr

        indices = mat.indices
        indptr = mat.indptr
        data = mx.nd.ones(indices.shape, dtype=np.float32)
        return mx.nd.sparse.csr_matrix((data, indices, indptr), shape=mat.shape)

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
        self.__init__(mx.nd.sparse.csr_matrix((edge_ids, (dst, src)), shape=out_coo.shape).astype(np.int64),
                mx.nd.sparse.csr_matrix((edge_ids, (src, dst)), shape=out_coo.shape).astype(np.int64))

def create_immutable_graph_index():
    """ Create an empty backend-specific immutable graph index.

    Returns
    -------
    ImmutableGraphIndex
        The backend-specific immutable graph index.
    """
    return ImmutableGraphIndex(None, None)
