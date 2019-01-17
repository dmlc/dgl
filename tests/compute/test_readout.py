import dgl
import backend as F

def test_simple_readout():
    g1 = dgl.DGLGraph()
    g1.add_nodes(3)
    g2 = dgl.DGLGraph()
    g2.add_nodes(4) # no edges
    g1.add_edges([0, 1, 2], [2, 0, 1])

    n1 = F.randn((3, 5))
    n2 = F.randn((4, 5))
    e1 = F.randn((3, 5))
    s1 = F.sum(n1, 0)   # node sums
    s2 = F.sum(n2, 0)
    se1 = F.sum(e1, 0)  # edge sums
    m1 = F.mean(n1, 0)  # node means
    m2 = F.mean(n2, 0)
    me1 = F.mean(e1, 0) # edge means
    w1 = F.randn((3,))
    w2 = F.randn((4,))
    max1 = F.max(n1, 0)
    max2 = F.max(n2, 0)
    maxe1 = F.max(e1, 0)
    ws1 = F.sum(n1 * F.unsqueeze(w1, 1), 0)
    ws2 = F.sum(n2 * F.unsqueeze(w2, 1), 0)
    wm1 = F.sum(n1 * F.unsqueeze(w1, 1), 0) / F.sum(F.unsqueeze(w1, 1), 0)
    wm2 = F.sum(n2 * F.unsqueeze(w2, 1), 0) / F.sum(F.unsqueeze(w2, 1), 0)
    g1.ndata['x'] = n1
    g2.ndata['x'] = n2
    g1.ndata['w'] = w1
    g2.ndata['w'] = w2
    g1.edata['x'] = e1

    assert F.allclose(dgl.sum_nodes(g1, 'x'), s1)
    assert F.allclose(dgl.sum_nodes(g1, 'x', 'w'), ws1)
    assert F.allclose(dgl.sum_edges(g1, 'x'), se1)
    assert F.allclose(dgl.mean_nodes(g1, 'x'), m1)
    assert F.allclose(dgl.mean_nodes(g1, 'x', 'w'), wm1)
    assert F.allclose(dgl.mean_edges(g1, 'x'), me1)
    assert F.allclose(dgl.max_nodes(g1, 'x'), max1)
    assert F.allclose(dgl.max_edges(g1, 'x'), maxe1)

    g = dgl.batch([g1, g2])
    s = dgl.sum_nodes(g, 'x')
    m = dgl.mean_nodes(g, 'x')
    max_bg = dgl.max_nodes(g, 'x')
    assert F.allclose(s, F.stack([s1, s2], 0))
    assert F.allclose(m, F.stack([m1, m2], 0))
    assert F.allclose(max_bg, F.stack([max1, max2], 0))
    ws = dgl.sum_nodes(g, 'x', 'w')
    wm = dgl.mean_nodes(g, 'x', 'w')
    assert F.allclose(ws, F.stack([ws1, ws2], 0))
    assert F.allclose(wm, F.stack([wm1, wm2], 0))
    s = dgl.sum_edges(g, 'x')
    m = dgl.mean_edges(g, 'x')
    max_bg_e = dgl.max_edges(g, 'x')
    assert F.allclose(s, F.stack([se1, F.zeros(5)], 0))
    assert F.allclose(m, F.stack([me1, F.zeros(5)], 0))
    assert F.allclose(max_bg_e, F.stack([maxe1, F.zeros(5)], 0))	


if __name__ == '__main__':
    test_simple_readout()
