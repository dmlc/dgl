import time
import math
import numpy as np
import scipy.sparse as sp
import torch as th
import dgl
import utils as U

def test_adjmat_speed():
    n = 1000
    p = 10 * math.log(n) / n
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.DGLGraph(a)
    # the first call should contruct the adj
    t0 = time.time()
    g.adjacency_matrix()
    dur1 = time.time() - t0
    # the second call should be cached and should be very fast
    t0 = time.time()
    g.adjacency_matrix()
    dur2 = time.time() - t0
    print('first time {}, second time {}'.format(dur1, dur2))
    assert dur2 < dur1

def test_incmat():
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edge(0, 1) # 0
    g.add_edge(0, 2) # 1
    g.add_edge(0, 3) # 2
    g.add_edge(2, 3) # 3
    g.add_edge(1, 1) # 4
    assert U.allclose(
            g.incidence_matrix('in').to_dense(),
            th.tensor([[0., 0., 0., 0., 0.],
                       [1., 0., 0., 0., 1.],
                       [0., 1., 0., 0., 0.],
                       [0., 0., 1., 1., 0.]]))
    assert U.allclose(
            g.incidence_matrix('out').to_dense(),
            th.tensor([[1., 1., 1., 0., 0.],
                       [0., 0., 0., 0., 1.],
                       [0., 0., 0., 1., 0.],
                       [0., 0., 0., 0., 0.]]))
    assert U.allclose(
            g.incidence_matrix('both').to_dense(),
            th.tensor([[-1., -1., -1., 0., 0.],
                       [1., 0., 0., 0., 0.],
                       [0., 1., 0., -1., 0.],
                       [0., 0., 1., 1., 0.]]))

def test_incmat_speed():
    n = 1000
    p = 2 * math.log(n) / n
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.DGLGraph(a)
    # the first call should contruct the adj
    t0 = time.time()
    g.incidence_matrix("in")
    dur1 = time.time() - t0
    # the second call should be cached and should be very fast
    t0 = time.time()
    g.incidence_matrix("in")
    dur2 = time.time() - t0
    print('first time {}, second time {}'.format(dur1, dur2))
    assert dur2 < dur1

if __name__ == '__main__':
    test_adjmat_speed()
    test_incmat()
    test_incmat_speed()
