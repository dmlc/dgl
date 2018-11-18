import time
import math
import numpy as np
import scipy.sparse as sp
import torch as th
import dgl

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
    assert dur2 < dur1 / 5

def test_incmat_speed():
    n = 1000
    p = 10 * math.log(n) / n
    a = sp.random(n, n, p, data_rvs=lambda n: np.ones(n))
    g = dgl.DGLGraph(a)
    # the first call should contruct the adj
    t0 = time.time()
    g.incidence_matrix()
    dur1 = time.time() - t0
    # the second call should be cached and should be very fast
    t0 = time.time()
    g.incidence_matrix()
    dur2 = time.time() - t0
    assert dur2 < dur1

if __name__ == '__main__':
    test_adjmat_speed()
    test_incmat_speed()
