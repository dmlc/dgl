from dgl.graph import DGLGraph
import backend as F
import numpy as np

def test_filter():
    g = DGLGraph()
    g.add_nodes(4)
    g.add_edges([0,1,2,3], [1,2,3,0])

    n_repr = np.zeros((4, 5))
    e_repr = np.zeros((4, 5))
    n_repr[[1, 3]] = 1
    e_repr[[1, 3]] = 1
    n_repr = F.copy_to(F.zerocopy_from_numpy(n_repr), F.ctx())
    e_repr = F.copy_to(F.zerocopy_from_numpy(e_repr), F.ctx())

    g.ndata['a'] = n_repr
    g.edata['a'] = e_repr

    def predicate(r):
        return F.max(r.data['a'], 1) > 0

    # full node filter
    n_idx = g.filter_nodes(predicate)
    assert set(F.zerocopy_to_numpy(n_idx)) == {1, 3}

    # partial node filter
    n_idx = g.filter_nodes(predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(n_idx)) == {1}

    # full edge filter
    e_idx = g.filter_edges(predicate)
    assert set(F.zerocopy_to_numpy(e_idx)) == {1, 3}

    # partial edge filter
    e_idx = g.filter_edges(predicate, [0, 1])
    assert set(F.zerocopy_to_numpy(e_idx)) == {1}


if __name__ == '__main__':
    test_filter()
