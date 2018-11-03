import torch as th
import numpy as np
from dgl.graph import DGLGraph

def test_filter():
    g = DGLGraph()
    g.add_nodes(4)
    g.add_edges([0,1,2,3], [1,2,3,0])

    n_repr = th.zeros(4, 5)
    e_repr = th.zeros(4, 5)
    n_repr[[1, 3]] = 1
    e_repr[[1, 3]] = 1

    g.ndata['a'] = n_repr
    g.edata['a'] = e_repr

    def predicate(r):
        return r['a'].max(1)[0] > 0

    # full node filter
    n_idx = g.filter_nodes(predicate)
    assert set(n_idx.numpy()) == {1, 3}

    # partial node filter
    n_idx = g.filter_nodes(predicate, [0, 1])
    assert set(n_idx.numpy()) == {1}

    # full edge filter
    e_idx = g.filter_edges(predicate)
    assert set(e_idx.numpy()) == {1, 3}

    # partial edge filter
    e_idx = g.filter_edges(predicate, [0, 1])
    assert set(e_idx.numpy()) == {1}


if __name__ == '__main__':
    test_filter()
