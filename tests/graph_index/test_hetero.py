import numpy as np
import dgl
import dgl.ndarray as nd
import dgl.graph_index as dgl_gidx
import dgl.heterograph_index as dgl_hgidx
from dgl.utils import toindex
import backend as F

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
    return list(dglidx) == list(l)

def rel1_from_coo():
    row = toindex([0, 1, 2, 3])
    col = toindex([0, 0, 0, 0])
    return dgl_hgidx.create_bipartite_from_coo(5, 2, row, col)

def rel2_from_coo():
    row = toindex([0, 1, 1])
    col = toindex([0, 1, 2])
    return dgl_hgidx.create_bipartite_from_coo(2, 3, row, col)

def rel3_from_coo():
    row = toindex([0, 1, 2])
    col = toindex([1, 1, 1])
    return dgl_hgidx.create_bipartite_from_coo(3, 2, row, col)

def rel1_from_csr():
    indptr = toindex([0, 1, 2, 3, 4, 4])
    indices = toindex([0, 0, 0, 0])
    edge_ids = toindex([0, 1, 2, 3])
    return dgl_hgidx.create_bipartite_from_csr(5, 2, indptr, indices, edge_ids)

def rel2_from_csr():
    indptr = toindex([0, 1, 3])
    indices = toindex([0, 1, 2])
    edge_ids = toindex([0, 1, 2])
    return dgl_hgidx.create_bipartite_from_csr(2, 3, indptr, indices, edge_ids)

def rel3_from_csr():
    indptr = toindex([0, 1, 2, 3])
    indices = toindex([1, 1, 1])
    edge_ids = toindex([0, 1, 2])
    return dgl_hgidx.create_bipartite_from_csr(3, 2, indptr, indices, edge_ids)

def gen_from_coo():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], is_multigraph=False, readonly=True)
    return dgl_hgidx.create_heterograph(mg, [rel1_from_coo(), rel2_from_coo(), rel3_from_coo()])

def gen_from_csr():
    mg = dgl_gidx.from_edge_list([(0, 1), (1, 2), (2, 1)], is_multigraph=False, readonly=True)
    return dgl_hgidx.create_heterograph(mg, [rel1_from_csr(), rel2_from_csr(), rel3_from_csr()])

def test_query():
    R1 = 0
    R2 = 1
    R3 = 2
    def _test_g(g):
        assert g.number_of_ntypes() == 3
        assert g.number_of_etypes() == 3
        assert g.metagraph.number_of_nodes() == 3
        assert g.metagraph.number_of_edges() == 3
        assert g.ctx() == nd.cpu(0)
        assert g.nbits() == 64
        assert not g.is_multigraph()
        assert g.is_readonly()
        # relation graph 1
        assert g.number_of_nodes(R1) == 5
        assert g.number_of_edges(R1) == 4
        assert g.has_node(0, 0)
        assert not g.has_node(0, 10)
        assert _array_equal(g.has_nodes(0, toindex([0, 10])), [1, 0])
        assert g.has_edge_between(R1, 3, 0)
        assert not g.has_edge_between(R1, 4, 0)
        assert _array_equal(g.has_edges_between(R1, toindex([3, 4]), toindex([0, 0])), [1, 0])
        assert _array_equal(g.predecessors(R1, 0), [0, 1, 2, 3])
        assert _array_equal(g.predecessors(R1, 1), [])
        assert _array_equal(g.successors(R1, 3), [0])
        assert _array_equal(g.successors(R1, 4), [])
        assert _array_equal(g.edge_id(R1, 0, 0), [0])
        src, dst, eid = g.edge_ids(R1, toindex([0, 2, 1, 3]), toindex([0, 0, 0, 0]))
        assert _array_equal(src, [0, 2, 1, 3])
        assert _array_equal(dst, [0, 0, 0, 0])
        assert _array_equal(eid, [0, 2, 1, 3])
        src, dst, eid = g.find_edges(R1, toindex([3, 0]))
        assert _array_equal(src, [3, 0])
        assert _array_equal(dst, [0, 0])
        assert _array_equal(eid, [3, 0])
        src, dst, eid = g.in_edges(R1, toindex([0, 1]))
        assert _array_equal(src, [0, 1, 2, 3])
        assert _array_equal(dst, [0, 0, 0, 0])
        assert _array_equal(eid, [0, 1, 2, 3])
        src, dst, eid = g.out_edges(R1, toindex([1, 0, 4]))
        assert _array_equal(src, [1, 0])
        assert _array_equal(dst, [0, 0])
        assert _array_equal(eid, [1, 0])
        src, dst, eid = g.edges(R1, 'eid')
        assert _array_equal(src, [0, 1, 2, 3])
        assert _array_equal(dst, [0, 0, 0, 0])
        assert _array_equal(eid, [0, 1, 2, 3])
        assert g.in_degree(R1, 0) == 4
        assert g.in_degree(R1, 1) == 0
        assert _array_equal(g.in_degrees(R1, toindex([0, 1])), [4, 0])
        assert g.out_degree(R1, 2) == 1
        assert g.out_degree(R1, 4) == 0
        assert _array_equal(g.out_degrees(R1, toindex([4, 2])), [0, 1])
        # adjmat
        adj = g.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.],
                          [1., 0.],
                          [1., 0.],
                          [1., 0.],
                          [0., 0.]]))
        adj = g.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0., 0.],
                          [0., 1., 1.]]))
        adj = g.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 1.],
                          [0., 1.],
                          [0., 1.]]))

    g = gen_from_coo()
    _test_g(g)
    g = gen_from_csr()
    _test_g(g)

def test_subgraph():
    R1 = 0
    R2 = 1
    R3 = 2
    def _test_g(g):
        # node subgraph
        induced_nodes = [toindex([0, 1, 4]), toindex([0]), toindex([0, 2])]
        sub = g.node_subgraph(induced_nodes)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 3
        assert subg.number_of_nodes(1) == 1
        assert subg.number_of_nodes(2) == 2
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 1
        assert subg.number_of_edges(R3) == 0
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1.],
                          [1.],
                          [0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.]]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0.],
                          [0.]]))
        assert len(sub.induced_nodes) == 3
        assert _array_equal(sub.induced_nodes[0], induced_nodes[0])
        assert _array_equal(sub.induced_nodes[1], induced_nodes[1])
        assert _array_equal(sub.induced_nodes[2], induced_nodes[2])
        assert len(sub.induced_edges) == 3
        assert _array_equal(sub.induced_edges[0], [0, 1])
        assert _array_equal(sub.induced_edges[1], [0])
        assert _array_equal(sub.induced_edges[2], [])

        # node subgraph with empty type graph
        induced_nodes = [toindex([0, 1, 4]), toindex([0]), toindex([])]
        sub = g.node_subgraph(induced_nodes)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 3
        assert subg.number_of_nodes(1) == 1
        assert subg.number_of_nodes(2) == 0
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 0
        assert subg.number_of_edges(R3) == 0
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1.],
                          [1.],
                          [0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([]))

        # edge subgraph (preserve_nodes=False)
        induced_edges = [toindex([0, 2]), toindex([0]), toindex([0, 1, 2])]
        sub = g.edge_subgraph(induced_edges, False)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 2
        assert subg.number_of_nodes(1) == 2
        assert subg.number_of_nodes(2) == 3
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 1
        assert subg.number_of_edges(R3) == 3
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.],
                          [1., 0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0., 0.],
                          [0., 0., 0.]]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 1.],
                          [0., 1.],
                          [0., 1.]]))
        assert len(sub.induced_nodes) == 3
        assert _array_equal(sub.induced_nodes[0], [0, 2])
        assert _array_equal(sub.induced_nodes[1], [0, 1])
        assert _array_equal(sub.induced_nodes[2], [0, 1, 2])
        assert len(sub.induced_edges) == 3
        assert _array_equal(sub.induced_edges[0], induced_edges[0])
        assert _array_equal(sub.induced_edges[1], induced_edges[1])
        assert _array_equal(sub.induced_edges[2], induced_edges[2])

        # edge subgraph (preserve_nodes=True)
        induced_edges = [toindex([0, 2]), toindex([0]), toindex([0, 1, 2])]
        sub = g.edge_subgraph(induced_edges, True)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 5
        assert subg.number_of_nodes(1) == 2
        assert subg.number_of_nodes(2) == 3
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 1
        assert subg.number_of_edges(R3) == 3
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.],
                          [0., 0.],
                          [1., 0.],
                          [0., 0.],
                          [0., 0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0., 0.],
                          [0., 0., 0.]]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 1.],
                          [0., 1.],
                          [0., 1.]]))
        assert len(sub.induced_nodes) == 3
        assert _array_equal(sub.induced_nodes[0], [0, 1, 2, 3, 4])
        assert _array_equal(sub.induced_nodes[1], [0, 1])
        assert _array_equal(sub.induced_nodes[2], [0, 1, 2])
        assert len(sub.induced_edges) == 3
        assert _array_equal(sub.induced_edges[0], induced_edges[0])
        assert _array_equal(sub.induced_edges[1], induced_edges[1])
        assert _array_equal(sub.induced_edges[2], induced_edges[2])

        # edge subgraph with empty induced edges (preserve_nodes=False)
        induced_edges = [toindex([0, 2]), toindex([]), toindex([0, 1, 2])]
        sub = g.edge_subgraph(induced_edges, False)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 2
        assert subg.number_of_nodes(1) == 2
        assert subg.number_of_nodes(2) == 3
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 0
        assert subg.number_of_edges(R3) == 3
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.],
                          [1., 0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 0., 0.],
                          [0., 0., 0.]]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 1.],
                          [0., 1.],
                          [0., 1.]]))
        assert len(sub.induced_nodes) == 3
        assert _array_equal(sub.induced_nodes[0], [0, 2])
        assert _array_equal(sub.induced_nodes[1], [0, 1])
        assert _array_equal(sub.induced_nodes[2], [0, 1, 2])
        assert len(sub.induced_edges) == 3
        assert _array_equal(sub.induced_edges[0], induced_edges[0])
        assert _array_equal(sub.induced_edges[1], induced_edges[1])
        assert _array_equal(sub.induced_edges[2], induced_edges[2])

        # edge subgraph with empty induced edges (preserve_nodes=True)
        induced_edges = [toindex([0, 2]), toindex([]), toindex([0, 1, 2])]
        sub = g.edge_subgraph(induced_edges, True)
        subg = sub.graph
        assert subg.number_of_ntypes() == 3
        assert subg.number_of_etypes() == 3
        assert subg.number_of_nodes(0) == 5
        assert subg.number_of_nodes(1) == 2
        assert subg.number_of_nodes(2) == 3
        assert subg.number_of_edges(R1) == 2
        assert subg.number_of_edges(R2) == 0
        assert subg.number_of_edges(R3) == 3
        adj = subg.adjacency_matrix(R1, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[1., 0.],
                          [0., 0.],
                          [1., 0.],
                          [0., 0.],
                          [0., 0.]]))
        adj = subg.adjacency_matrix(R2, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 0., 0.],
                          [0., 0., 0.]]))
        adj = subg.adjacency_matrix(R3, True, F.cpu())[0]
        assert np.allclose(F.sparse_to_numpy(adj),
                np.array([[0., 1.],
                          [0., 1.],
                          [0., 1.]]))
        assert len(sub.induced_nodes) == 3
        assert _array_equal(sub.induced_nodes[0], [0, 1, 2, 3, 4])
        assert _array_equal(sub.induced_nodes[1], [0, 1])
        assert _array_equal(sub.induced_nodes[2], [0, 1, 2])
        assert len(sub.induced_edges) == 3
        assert _array_equal(sub.induced_edges[0], induced_edges[0])
        assert _array_equal(sub.induced_edges[1], induced_edges[1])
        assert _array_equal(sub.induced_edges[2], induced_edges[2])

    g = gen_from_coo()
    _test_g(g)
    g = gen_from_csr()
    _test_g(g)

if __name__ == '__main__':
    test_query()
    test_subgraph()
