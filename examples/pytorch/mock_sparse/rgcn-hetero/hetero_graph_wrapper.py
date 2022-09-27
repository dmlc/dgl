from dgl.mock_sparse import create_from_coo


class HeteroGraphWrapper(object):
    """A wrapper of DGLHeteroGraph, providing a sparse matrix representation for each relation"""

    def __init__(self, dgl_graph):
        self._graph = dgl_graph

        def create_adj(rel):
            stype, _, dtype = rel
            row, col = self._graph.edges(etype=rel)
            return create_from_coo(
                col,
                row,
                shape=(
                    dgl_graph.number_of_nodes(dtype),
                    dgl_graph.number_of_nodes(stype),
                ),
            )

        self.adj = {
            rel: create_adj(rel) for rel in self._graph.canonical_etypes
        }

    @property
    def canonical_etypes(self):
        return self._graph.canonical_etypes

    @property
    def ntypes(self):
        return self._graph.ntypes

    @property
    def etypes(self):
        return self._graph.etypes

    def number_of_nodes(self, ntype):
        return self._graph.number_of_nodes(ntype)
