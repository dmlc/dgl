import torch as th
import dgl
import utils as U

def test_simple_readout():
    g1 = dgl.DGLGraph()
    g1.add_nodes(3)
    g2 = dgl.DGLGraph()
    g2.add_nodes(4) # no edges
    g1.add_edges([0, 1, 2], [2, 0, 1])

    n1 = th.randn(3, 5)
    n2 = th.randn(4, 5)
    e1 = th.randn(3, 5)
    s1 = n1.sum(0)      # node sums
    s2 = n2.sum(0)
    se1 = e1.sum(0)     # edge sums
    m1 = n1.mean(0)     # node means
    m2 = n2.mean(0)
    me1 = e1.mean(0)    # edge means
    w1 = th.randn(3)
    w2 = th.randn(4)
    ws1 = (n1 * w1[:, None]).sum(0)                         # weighted node sums
    ws2 = (n2 * w2[:, None]).sum(0)
    wm1 = (n1 * w1[:, None]).sum(0) / w1[:, None].sum(0)    # weighted node means
    wm2 = (n2 * w2[:, None]).sum(0) / w2[:, None].sum(0)
    g1.ndata['x'] = n1
    g2.ndata['x'] = n2
    g1.ndata['w'] = w1
    g2.ndata['w'] = w2
    g1.edata['x'] = e1

    assert U.allclose(dgl.sum_nodes(g1, 'x'), s1)
    assert U.allclose(dgl.sum_nodes(g1, 'x', 'w'), ws1)
    assert U.allclose(dgl.sum_edges(g1, 'x'), se1)
    assert U.allclose(dgl.mean_nodes(g1, 'x'), m1)
    assert U.allclose(dgl.mean_nodes(g1, 'x', 'w'), wm1)
    assert U.allclose(dgl.mean_edges(g1, 'x'), me1)

    g = dgl.batch([g1, g2])
    s = dgl.sum_nodes(g, 'x')
    m = dgl.mean_nodes(g, 'x')
    assert U.allclose(s, th.stack([s1, s2], 0))
    assert U.allclose(m, th.stack([m1, m2], 0))
    ws = dgl.sum_nodes(g, 'x', 'w')
    wm = dgl.mean_nodes(g, 'x', 'w')
    assert U.allclose(ws, th.stack([ws1, ws2], 0))
    assert U.allclose(wm, th.stack([wm1, wm2], 0))
    s = dgl.sum_edges(g, 'x')
    m = dgl.mean_edges(g, 'x')
    assert U.allclose(s, th.stack([se1, th.zeros(5)], 0))
    assert U.allclose(m, th.stack([me1, th.zeros(5)], 0))


if __name__ == '__main__':
    test_simple_readout()
