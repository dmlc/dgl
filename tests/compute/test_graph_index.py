import os
import backend as F
import networkx as nx
import numpy as np
import scipy as sp
from scipy import sparse as spsp
import dgl
from dgl.graph_index import map_to_subgraph_nid, GraphIndex, create_graph_index
from dgl import utils

def generate_from_networkx():
    edges = [[2, 3], [2, 5], [3, 0], [1, 0], [4, 3], [4, 5]]
    nx_graph = nx.DiGraph()
    nx_graph.add_edges_from(edges)
    g = create_graph_index(nx_graph, multigraph=False, readonly=False)
    ig = create_graph_index(nx_graph, multigraph=False, readonly=True)
    return g, ig

def generate_from_edgelist():
    edges = [[2, 3], [2, 5], [3, 0], [6, 10], [10, 3], [10, 15]]
    g = create_graph_index(edges, multigraph=False, readonly=False)
    ig = create_graph_index(edges, multigraph=False, readonly=True)
    return g, ig

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = create_graph_index(arr, multigraph=False, readonly=False)
    ig = create_graph_index(arr, multigraph=False, readonly=True)
    return g, ig

def check_graph_equal(g1, g2):
    adj1 = g1.adjacency_matrix(False, F.cpu())[0]
    adj2 = g2.adjacency_matrix(False, F.cpu())[0]
    assert np.all(F.asnumpy(adj1) == F.asnumpy(adj2))

def test_graph_gen():
    g, ig = generate_from_edgelist()
    check_graph_equal(g, ig)
    g, ig = generate_rand_graph(10)
    check_graph_equal(g, ig)

def sort_edges(edges):
    edges = [e.tousertensor() for e in edges]
    if np.prod(edges[2].shape) > 0:
        val, idx = F.sort_1d(edges[2])
        return (F.gather_row(edges[0], idx), F.gather_row(edges[1], idx), F.gather_row(edges[2], idx))
    else:
        return (edges[0], edges[1], edges[2])

def check_basics(g, ig):
    assert g.number_of_nodes() == ig.number_of_nodes()
    assert g.number_of_edges() == ig.number_of_edges()

    edges = g.edges("srcdst")
    iedges = ig.edges("srcdst")
    assert F.array_equal(edges[0].tousertensor(), iedges[0].tousertensor())
    assert F.array_equal(edges[1].tousertensor(), iedges[1].tousertensor())
    assert F.array_equal(edges[2].tousertensor(), iedges[2].tousertensor())

    edges = g.edges("eid")
    iedges = ig.edges("eid")
    assert F.array_equal(edges[0].tousertensor(), iedges[0].tousertensor())
    assert F.array_equal(edges[1].tousertensor(), iedges[1].tousertensor())
    assert F.array_equal(edges[2].tousertensor(), iedges[2].tousertensor())

    for i in range(g.number_of_nodes()):
        assert g.has_node(i) == ig.has_node(i)

    for i in range(g.number_of_nodes()):
        assert F.array_equal(g.predecessors(i).tousertensor(), ig.predecessors(i).tousertensor())
        assert F.array_equal(g.successors(i).tousertensor(), ig.successors(i).tousertensor())

    randv = np.random.randint(0, g.number_of_nodes(), 10)
    randv = utils.toindex(randv)
    in_src1, in_dst1, in_eids1 = sort_edges(g.in_edges(randv))
    in_src2, in_dst2, in_eids2 = sort_edges(ig.in_edges(randv))
    nnz = in_src2.shape[0]
    assert F.array_equal(in_src1, in_src2)
    assert F.array_equal(in_dst1, in_dst2)
    assert F.array_equal(in_eids1, in_eids2)

    out_src1, out_dst1, out_eids1 = sort_edges(g.out_edges(randv))
    out_src2, out_dst2, out_eids2 = sort_edges(ig.out_edges(randv))
    nnz = out_dst2.shape[0]
    assert F.array_equal(out_dst1, out_dst2)
    assert F.array_equal(out_src1, out_src2)
    assert F.array_equal(out_eids1, out_eids2)

    num_v = len(randv)
    assert F.array_equal(g.in_degrees(randv).tousertensor(), ig.in_degrees(randv).tousertensor())
    assert F.array_equal(g.out_degrees(randv).tousertensor(), ig.out_degrees(randv).tousertensor())
    randv = randv.tousertensor()
    for v in F.asnumpy(randv):
        assert g.in_degree(v) == ig.in_degree(v)
        assert g.out_degree(v) == ig.out_degree(v)

    for u in F.asnumpy(randv):
        for v in F.asnumpy(randv):
            if len(g.edge_id(u, v)) == 1:
                assert g.edge_id(u, v).tonumpy() == ig.edge_id(u, v).tonumpy()
            assert g.has_edge_between(u, v) == ig.has_edge_between(u, v)
    randv = utils.toindex(randv)
    ids = g.edge_ids(randv, randv)[2].tonumpy()
    assert sum(ig.edge_ids(randv, randv)[2].tonumpy() == ids, 0) == len(ids)
    assert sum(g.has_edges_between(randv, randv).tonumpy() == ig.has_edges_between(randv, randv).tonumpy(), 0) == len(randv)


def test_basics():
    g, ig = generate_from_edgelist()
    check_basics(g, ig)
    g, ig = generate_from_networkx()
    check_basics(g, ig)
    g, ig = generate_rand_graph(100)
    check_basics(g, ig)

def test_node_subgraph():
    num_vertices = 100
    g, ig = generate_rand_graph(num_vertices)

    # node_subgraph
    randv1 = np.random.randint(0, num_vertices, 20)
    randv = np.unique(randv1)
    subg = g.node_subgraph(utils.toindex(randv))
    subig = ig.node_subgraph(utils.toindex(randv))
    check_basics(subg.graph, subig.graph)
    check_graph_equal(subg.graph, subig.graph)
    assert F.asnumpy(map_to_subgraph_nid(subg, utils.toindex(randv1[0:10])).tousertensor()
            == map_to_subgraph_nid(subig, utils.toindex(randv1[0:10])).tousertensor()).sum(0).item() == 10

    # node_subgraphs
    randvs = []
    subgs = []
    for i in range(4):
        randv = np.unique(np.random.randint(0, num_vertices, 20))
        randvs.append(utils.toindex(randv))
        subgs.append(g.node_subgraph(utils.toindex(randv)))
    subigs= ig.node_subgraphs(randvs)
    for i in range(4):
        check_basics(subg.graph, subig.graph)
        check_graph_equal(subgs[i].graph, subigs[i].graph)

def test_create_graph():
    elist = [(1, 2), (0, 1), (0, 2)]
    ig = dgl.DGLGraph(elist, readonly=True)
    g = dgl.DGLGraph(elist, readonly=False)
    for edge in elist:
        assert g.edge_id(edge[0], edge[1]) == ig.edge_id(edge[0], edge[1])

    data = [1, 2, 3]
    rows = [1, 0, 0]
    cols = [2, 1, 2]
    mat = sp.sparse.coo_matrix((data, (rows, cols)))
    g = dgl.DGLGraph(mat, readonly=False)
    ig = dgl.DGLGraph(mat, readonly=True)
    for edge in elist:
        assert g.edge_id(edge[0], edge[1]) == ig.edge_id(edge[0], edge[1])

def test_load_csr():
    n = 100
    csr = (sp.sparse.random(n, n, density=0.1, format='csr') != 0).astype(np.int64)

    # Load CSR normally.
    idx = dgl.graph_index.from_csr(
            utils.toindex(csr.indptr), utils.toindex(csr.indices), False, 'out')
    assert idx.number_of_nodes() == n
    assert idx.number_of_edges() == csr.nnz
    src, dst, eid = idx.edges()
    src, dst, eid = src.tousertensor(), dst.tousertensor(), eid.tousertensor()
    coo = csr.tocoo()
    assert np.all(F.asnumpy(src) == coo.row)
    assert np.all(F.asnumpy(dst) == coo.col)

    # Load CSR to shared memory.
    # Shared memory isn't supported in Windows.
    if os.name is not 'nt':
        idx = dgl.graph_index.from_csr(
                utils.toindex(csr.indptr), utils.toindex(csr.indices),
                False, 'out', '/test_graph_struct')
        assert idx.number_of_nodes() == n
        assert idx.number_of_edges() == csr.nnz
        src, dst, eid = idx.edges()
        src, dst, eid = src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        coo = csr.tocoo()
        assert np.all(F.asnumpy(src) == coo.row)
        assert np.all(F.asnumpy(dst) == coo.col)

def test_edge_ids():
    np.random.seed(0)
    csr = (spsp.random(20, 20, density=0.1, format='csr') != 0).astype(np.int64)
    #csr = csr.transpose()
    g = dgl.DGLGraph(csr, readonly=True)
    num_nodes = g.number_of_nodes()
    in_edges = g._graph.in_edges(v=dgl.utils.toindex([2]))
    src, dst, eids = g._graph.edge_ids(dgl.utils.toindex(in_edges[0]),
                                       dgl.utils.toindex(in_edges[1]))
    assert np.all(in_edges[0].tonumpy() == src.tonumpy())
    assert np.all(in_edges[1].tonumpy() == dst.tonumpy())

if __name__ == '__main__':
    test_basics()
    test_edge_ids()
    test_graph_gen()
    test_node_subgraph()
    test_create_graph()
    test_load_csr()
