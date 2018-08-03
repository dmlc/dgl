"""High-performance graph structure query component.

TODO: Currently implemented by igraph. Should replace with more efficient
solution later.
"""
from __future__ import absolute_import

import igraph

import dgl.backend as F
from dgl.backend import Tensor
import dgl.utils as utils

class CachedGraph:
    def __init__(self):
        self._graph = igraph.Graph(directed=True)
        self._adjmat = None  # cached adjacency matrix

    def add_nodes(self, num_nodes):
        self._graph.add_vertices(num_nodes)

    def add_edge(self, u, v):
        self._graph.add_edge(u, v)

    def add_edges(self, u, v):
        # The edge will be assigned ids equal to the order.
        for uu, vv in utils.edge_iter(u, v):
            self._graph.add_edge(uu, vv)

    def get_edge_id(self, u, v):
        uvs = list(utils.edge_iter(u, v))
        eids = self._graph.get_eids(uvs)
        return F.tensor(eids, dtype=F.int64)

    def in_edges(self, v):
        src = []
        dst = []
        for vv in utils.node_iter(v):
            uu = self._graph.predecessors(vv)
            src += uu
            dst += [vv] * len(uu)
        src = F.tensor(src, dtype=F.int64)
        dst = F.tensor(dst, dtype=F.int64)
        return src, dst

    def out_edges(self, u):
        src = []
        dst = []
        for uu in utils.node_iter(u):
            vv = self._graph.successors(uu)
            src += [uu] * len(vv)
            dst += vv
        src = F.tensor(src, dtype=F.int64)
        dst = F.tensor(dst, dtype=F.int64)
        return src, dst

    def edges(self):
        elist = self._graph.get_edgelist()
        src = [u for u, _ in elist]
        dst = [v for _, v in elist]
        src = F.tensor(src, dtype=F.int64)
        dst = F.tensor(dst, dtype=F.int64)
        return src, dst

    def in_degrees(self, v):
        degs = self._graph.indegree(list(v))
        return F.tensor(degs, dtype=F.int64)

    @property
    def adjmat(self):
        """Return a sparse adjacency matrix.

        The row dimension represents the dst nodes; the column dimension
        represents the src nodes.
        """
        if self._adjmat is None:
            elist = self._graph.get_edgelist()
            src = [u for u, _ in elist]
            dst = [v for _, v in elist]
            src = F.unsqueeze(F.tensor(src, dtype=F.int64), 0)
            dst = F.unsqueeze(F.tensor(dst, dtype=F.int64), 0)
            idx = F.pack([dst, src])
            n = self._graph.vcount()
            dat = F.ones((len(elist),))
            self._adjmat = F.sparse_tensor(idx, dat, [n, n])
        return self._adjmat

def create_cached_graph(dglgraph):
    cg = CachedGraph()
    cg.add_nodes(dglgraph.number_of_nodes())
    cg._graph.add_edges(dglgraph.edge_list)
    return cg
