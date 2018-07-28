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

    def add_nodes(self, num_nodes):
        self._graph.add_vertices(num_nodes)

    def add_edges(self, u, v):
        # The edge will be assigned ids equal to the order.
        # TODO(minjie): tensorize the loop
        for uu, vv in utils.edge_iter(u, v):
            self._graph.add_edge(uu, vv)

    def get_edge_id(self, u, v):
        # TODO(minjie): tensorize the loop
        uvs = list(utils.edge_iter(u, v))
        eids = self._graph.get_eids(uvs)
        return F.tensor(eids, dtype=F.int64)

    def in_edges(self, v):
        # TODO(minjie): tensorize the loop
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
        # TODO(minjie): tensorize the loop
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
        # TODO(minjie): return two tensors, src and dst.
        raise NotImplementedError()

    def in_degrees(self, v):
        degs = self._graph.indegree(list(v))
        return F.tensor(degs, dtype=F.int64)

def create_cached_graph(dglgraph):
    # TODO: tensorize the loop
    cg = CachedGraph()
    cg.add_nodes(dglgraph.number_of_nodes())
    for u, v in dglgraph.edges():
        cg.add_edges(u, v)
    return cg
