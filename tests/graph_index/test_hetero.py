import numpy as np
import dgl
import dgl.ndarray as nd
import dgl.graph_index as dgl_gidx
from dgl.utils import toindex

"""
Test with a heterograph of three ntypes and three etypes
meta graph:
0 -> 1
1 -> 2
2 -> 1

Num nodes per ntype:
0 : 5
1 : 2
2 : 3

rel graph:
0->1 : [0 -> 0, 1 -> 0, 2 -> 0, 3 -> 0]
1->2 : [0 -> 0, 1 -> 1, 1 -> 2]
2->1 : [0 -> 1, 1 -> 1, 2 -> 1]
"""

def _array_equal(dglidx, l):
    return dglidx.tonumpy().tolist() == l

def rel1_from_coo():
    row = toindex([0, 1, 2, 3])
    col = toindex([0, 0, 0, 0])
    return dgl_gidx.create_bipartite_from_coo(5, 2, row, col)

def rel2_from_coo():
    row = toindex([0, 1, 1])
    col = toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_coo(2, 3, row, col)

def rel3_from_coo():
    row = toindex([0, 1, 2])
    col = toindex([1, 1, 1])
    return dgl_gidx.create_bipartite_from_coo(3, 2, row, col)

def rel1_from_csr():
    indptr = toindex([0, 1, 2, 3, 4, 4])
    indices = toindex([0, 0, 0, 0])
    edge_ids = toindex([0, 1, 2, 3])
    return dgl_gidx.create_bipartite_from_csr(5, 2, indptr, indices, edge_ids)

def rel2_from_csr():
    row = toindex([0, 1, 3])
    col = toindex([0, 1, 2])
    edge_ids = toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_csr(2, 3, indptr, indices, edge_ids)

def rel3_from_csr():
    row = toindex([0, 1, 2, 3])
    col = toindex([1, 1, 1])
    edge_ids = toindex([0, 1, 2])
    return dgl_gidx.create_bipartite_from_csr(3, 2, indptr, indices, edge_ids)

def gen_from_coo():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], is_multigraph=False, readonly=True)
    return dgl_gidx.create_heterograph(mg, [rel1_from_coo(), rel2_from_coo(), rel3_from_coo()])

def gen_from_csr():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], is_multigraph=False, readonly=True)
    return dgl_gidx.create_heterograph(mg, [rel1_from_csr(), rel2_from_csr(), rel3_from_csr()])

def test_query():
    def _test_g(g):
        assert g.number_of_ntypes() == 3
        assert g.number_of_etypes() == 3
        assert g.meta_graph.number_of_nodes() == 3
        assert g.meta_graph.number_of_edges() == 3
        assert g.ctx() == nd.cpu(0)
        assert g.nbits() == 64
        assert not g.is_multigraph()
        assert g.is_readonly()
        # relation graph 1
        assert g.number_of_nodes(0) == 5
        assert g.number_of_edges(0) == 4
        assert g.has_node(0, 0)
        assert not g.has_node(0, 10)
        assert _array_equal(g.has_nodes(0, toindex([0, 10])), [1, 0])
        assert g.has_edge_between(0, 3, 0)
        assert not g.has_edge_between(0, 4, 0)
        assert _array_equal(g.has_edges_between(0, toindex([3, 4]), toindex([0, 0])), [1, 0])

        assert g.number_of_nodes(1) == 2

    g = gen_from_coo()
    _test_g(g)

if __name__ == '__main__':
    test_query()
