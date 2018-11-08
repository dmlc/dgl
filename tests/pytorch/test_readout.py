import torch as th
import dgl

def test_readout_sum():
    g1 = dgl.DGLGraph()
    g1.add_nodes(3)
    g2 = dgl.DGLGraph()
    g2.add_nodes(4) # no edges
    g1.add_edges([0, 1, 2], [2, 0, 1])

    n1 = th.randn(3, 5)
    n2 = th.randn(4, 5)
    e1 = th.randn(3, 5)
    s1 = n1.sum(0)
    s2 = n2.sum(0)
    se1 = e1.sum(0)
    w1 = th.randn(3)
    w2 = th.randn(4)
    ws1 = (n1 * w1[:, None]).sum(0)
    ws2 = (n2 * w2[:, None]).sum(0)
    g1.ndata['x'] = n1
    g2.ndata['x'] = n2
    g1.ndata['w'] = w1
    g2.ndata['w'] = w2
    g1.edata['x'] = e1

    assert th.allclose(dgl.sum_on(g1, 'nodes', 'x'), s1)
    assert th.allclose(dgl.sum_on(g1, 'nodes', 'x', 'w'), ws1)
    assert th.allclose(dgl.sum_on(g1, 'edges', 'x'), se1)

    g = dgl.batch([g1, g2])
    s = dgl.sum_on(g, 'nodes', 'x')
    assert th.allclose(s, th.stack([s1, s2], 0))
    ws = dgl.sum_on(g, 'nodes', 'x', 'w')
    assert th.allclose(ws, th.stack([ws1, ws2], 0))
    s = dgl.sum_on(g, 'edges', 'x')
    assert th.allclose(s, th.stack([se1, th.zeros(5)], 0))


if __name__ == '__main__':
    test_readout_sum()
