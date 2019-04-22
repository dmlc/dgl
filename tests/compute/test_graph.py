import time
import math
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import backend as F
from dgl import DGLError

def test_graph_creation():
    g = dgl.DGLGraph()
    # test add nodes with data
    g.add_nodes(5)
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    ans = F.cat([F.zeros((5, 2)), F.ones((5, 2))], 0)
    assert F.allclose(ans, g.ndata['h'])
    g.ndata['w'] = 2 * F.ones((10, 2))
    assert F.allclose(2 * F.ones((10, 2)), g.ndata['w'])
    # test add edges with data
    g.add_edges([2, 3], [3, 4])
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    ans = F.cat([F.zeros((2, 2)), F.ones((2, 2))], 0)
    assert F.allclose(ans, g.edata['m'])
    # test clear and add again
    g.clear()
    g.add_nodes(5)
    g.ndata['h'] = 3 * F.ones((5, 2))
    assert F.allclose(3 * F.ones((5, 2)), g.ndata['h'])

def test_create_from_elist():
    elist = [(2, 1), (1, 0), (2, 0), (3, 0), (0, 2)]
    g = dgl.DGLGraph(elist)
    for i, (u, v) in enumerate(elist):
        assert g.edge_id(u, v) == i
    # immutable graph
    # XXX: not enabled for pytorch
    #g = dgl.DGLGraph(elist, readonly=True)
    #for i, (u, v) in enumerate(elist):
    #    assert g.edge_id(u, v) == i

def test_scipy_adjmat():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(range(9), range(1, 10))

    adj_0 = g.adjacency_matrix_scipy()
    adj_1 = g.adjacency_matrix_scipy(fmt='coo')
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

    adj_t0 = g.adjacency_matrix_scipy(transpose=True)
    adj_t_1 = g.adjacency_matrix_scipy(transpose=True, fmt='coo')
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

    g.readonly()
    adj_2 = g.adjacency_matrix_scipy()
    adj_3 = g.adjacency_matrix_scipy(fmt='coo')
    assert np.array_equal(adj_2.toarray(), adj_3.toarray())
    assert np.array_equal(adj_0.toarray(), adj_2.toarray())

    adj_t2 = g.adjacency_matrix_scipy(transpose=True)
    adj_t3 = g.adjacency_matrix_scipy(transpose=True, fmt='coo')
    assert np.array_equal(adj_t2.toarray(), adj_t3.toarray())
    assert np.array_equal(adj_t0.toarray(), adj_t2.toarray())

def test_adjmat_cache():
    n = 1000
    p = 10 * math.log(n) / n
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.DGLGraph(a)
    # the first call should contruct the adj
    t0 = time.time()
    adj1 = g.adjacency_matrix()
    dur1 = time.time() - t0
    # the second call should be cached and should be very fast
    t0 = time.time()
    adj2 = g.adjacency_matrix()
    dur2 = time.time() - t0
    print('first time {}, second time {}'.format(dur1, dur2))
    assert dur2 < dur1
    assert id(adj1) == id(adj2)
    # different arg should result in different cache
    adj3 = g.adjacency_matrix(transpose=True)
    assert id(adj3) != id(adj2)
    # manually clear the cache
    g.clear_cache()
    adj35 = g.adjacency_matrix()
    assert id(adj35) != id(adj2)
    # mutating the graph should invalidate the cache
    g.add_nodes(10)
    adj4 = g.adjacency_matrix()
    assert id(adj4) != id(adj35)

def test_incmat():
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edge(0, 1) # 0
    g.add_edge(0, 2) # 1
    g.add_edge(0, 3) # 2
    g.add_edge(2, 3) # 3
    g.add_edge(1, 1) # 4
    inc_in = F.sparse_to_numpy(g.incidence_matrix('in'))
    inc_out = F.sparse_to_numpy(g.incidence_matrix('out'))
    inc_both = F.sparse_to_numpy(g.incidence_matrix('both'))
    print(inc_in)
    print(inc_out)
    print(inc_both)
    assert np.allclose(
            inc_in,
            np.array([[0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 1.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 1., 0.]]))
    assert np.allclose(
            inc_out,
            np.array([[1., 1., 1., 0., 0.],
                      [0., 0., 0., 0., 1.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0.]]))
    assert np.allclose(
            inc_both,
            np.array([[-1., -1., -1., 0., 0.],
                      [1., 0., 0., 0., 0.],
                      [0., 1., 0., -1., 0.],
                      [0., 0., 1., 1., 0.]]))

def test_incmat_cache():
    n = 1000
    p = 10 * math.log(n) / n
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.DGLGraph(a)
    # the first call should contruct the inc
    t0 = time.time()
    inc1 = g.incidence_matrix("in")
    dur1 = time.time() - t0
    # the second call should be cached and should be very fast
    t0 = time.time()
    inc2 = g.incidence_matrix("in")
    dur2 = time.time() - t0
    print('first time {}, second time {}'.format(dur1, dur2))
    assert dur2 < dur1
    assert id(inc1) == id(inc2)
    # different arg should result in different cache
    inc3 = g.incidence_matrix("both")
    assert id(inc3) != id(inc2)
    # manually clear the cache
    g.clear_cache()
    inc35 = g.incidence_matrix("in")
    assert id(inc35) != id(inc2)
    # mutating the graph should invalidate the cache
    g.add_nodes(10)
    inc4 = g.incidence_matrix("in")
    assert id(inc4) != id(inc35)

def test_readonly():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([0, 1, 2, 3], [1, 2, 3, 4])
    g.ndata['x'] = F.zeros((5, 3))
    g.edata['x'] = F.zeros((4, 4))

    g.readonly(False)
    assert g._graph.is_readonly() == False
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    g.readonly()
    assert g._graph.is_readonly() == True 
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(5)
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly()
    assert g._graph.is_readonly() == True 
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(5)
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly(False)
    assert g._graph.is_readonly() == False
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4

    try:
        g.add_nodes(10)
        g.add_edges([4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert not fail
        assert g.number_of_nodes() == 15
        assert F.shape(g.ndata['x']) == (15, 3)
        assert g.number_of_edges() == 14
        assert F.shape(g.edata['x']) == (14, 4)

def test_find_edges():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(range(9), range(1, 10))
    e = g.find_edges([1, 3, 2, 4])
    assert e[0][0] == 1 and e[0][1] == 3 and e[0][2] == 2 and e[0][3] == 4
    assert e[1][0] == 2 and e[1][1] == 4 and e[1][2] == 3 and e[1][3] == 5

    try:
        g.find_edges([10])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

    g.readonly()
    e = g.find_edges([1, 3, 2, 4])
    assert e[0][0] == 1 and e[0][1] == 3 and e[0][2] == 2 and e[0][3] == 4
    assert e[1][0] == 2 and e[1][1] == 4 and e[1][2] == 3 and e[1][3] == 5

    try:
        g.find_edges([10])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

if __name__ == '__main__':
    test_graph_creation()
    test_create_from_elist()
    test_adjmat_cache()
    test_scipy_adjmat()
    test_incmat()
    test_incmat_cache()
    test_readonly()
    test_find_edges()
