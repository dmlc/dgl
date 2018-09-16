"""High-performance graph structure query component.

TODO: Currently implemented by igraph. Should replace with more efficient
solution later.
"""
from __future__ import absolute_import

import igraph

from . import backend as F
from .backend import Tensor
from . import utils

class CachedGraph:
    def __init__(self):
        self._graph = igraph.Graph(directed=True)
        self._freeze = False

    def add_nodes(self, num_nodes):
        if self._freeze:
            raise RuntimeError('Freezed cached graph cannot be mutated.')
        self._graph.add_vertices(num_nodes)

    def add_edge(self, u, v):
        if self._freeze:
            raise RuntimeError('Freezed cached graph cannot be mutated.')
        self._graph.add_edge(u, v)

    def add_edges(self, u, v):
        if self._freeze:
            raise RuntimeError('Freezed cached graph cannot be mutated.')
        # The edge will be assigned ids equal to the order.
        uvs = list(utils.edge_iter(u, v))
        self._graph.add_edges(uvs)

    def get_edge_id(self, u, v):
        uvs = list(utils.edge_iter(u, v))
        eids = self._graph.get_eids(uvs)
        return utils.toindex(eids)

    def in_edges(self, v):
        """Get in-edges of the vertices.

        Parameters
        ----------
        v : utils.Index
            The vertex ids.

        Returns
        -------
        src : utils.Index
            The src vertex ids.
        dst : utils.Index
            The dst vertex ids.
        orphan : utils.Index
            The vertice that have no in-edges.
        """
        src = []
        dst = []
        orphan = []
        for vv in utils.node_iter(v):
            uu = self._graph.predecessors(vv)
            if len(uu) == 0:
                orphan.append(vv)
            else:
                src += uu
                dst += [vv] * len(uu)
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        orphan = utils.toindex(orphan)
        return src, dst, orphan

    def out_edges(self, u):
        """Get out-edges of the vertices.

        Parameters
        ----------
        v : utils.Index
            The vertex ids.

        Returns
        -------
        src : utils.Index
            The src vertex ids.
        dst : utils.Index
            The dst vertex ids.
        orphan : utils.Index
            The vertice that have no out-edges.
        """
        src = []
        dst = []
        orphan = []
        for uu in utils.node_iter(u):
            vv = self._graph.successors(uu)
            if len(vv) == 0:
                orphan.append(uu)
            else:
                src += [uu] * len(vv)
                dst += vv
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        orphan = utils.toindex(orphan)
        return src, dst, orphan

    def in_degrees(self, v):
        degs = self._graph.indegree(list(v))
        return utils.toindex(degs)

    def num_edges(self):
        return self._graph.ecount()

    @utils.cached_member
    def edges(self):
        elist = self._graph.get_edgelist()
        src = [u for u, _ in elist]
        dst = [v for _, v in elist]
        src = utils.toindex(src)
        dst = utils.toindex(dst)
        return src, dst

    @utils.cached_member
    def adjmat(self):
        """Return a sparse adjacency matrix.

        The row dimension represents the dst nodes; the column dimension
        represents the src nodes.
        """
        elist = self._graph.get_edgelist()
        src = F.tensor([u for u, _ in elist], dtype=F.int64)
        dst = F.tensor([v for _, v in elist], dtype=F.int64)
        src = F.unsqueeze(src, 0)
        dst = F.unsqueeze(dst, 0)
        idx = F.pack([dst, src])
        n = self._graph.vcount()
        dat = F.ones((len(elist),))
        mat = F.sparse_tensor(idx, dat, [n, n])
        return utils.CtxCachedObject(lambda ctx: F.to_context(mat, ctx))

    def freeze(self):
        self._freeze = True

def create_cached_graph(dglgraph):
    cg = CachedGraph()
    cg.add_nodes(dglgraph.number_of_nodes())
    cg._graph.add_edges(dglgraph.edge_list)
    cg.freeze()
    return cg
