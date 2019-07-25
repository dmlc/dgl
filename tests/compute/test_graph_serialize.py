import backend as F

def construct_graph(n, readonly=True):
    from dgl import DGLGraph
    g_list = []
    for i in range(n):
        g = DGLGraph()
        g.add_nodes(10)
        g.add_edges(1, 2)
        g.add_edges(3, 2)
        g.add_edges(3, 3)


        g.edata['e1'] = F.randn((3, 5))
        g.edata['e2'] = F.ones((3, 5))
        g.ndata['n1'] = F.randn((10, 2))
        g.readonly(i % 2 == 0)
        g_list.append(g)
    return g_list

def test_graph_serialize():
    g_list = construct_graph(3)

    from dgl.graph_serialize import save_graphs, load_graphs
    save_graphs("/tmp/test.bin", g_list)
    loadg_list = load_graphs("/tmp/test.bin", [2])
    load_g = loadg_list[0]

    assert F.allclose(load_g.nodes(), g_list[2].nodes())
    assert F.allclose(load_g.edges()[0], g_list[2].edges()[0])
    assert F.allclose(load_g.edges()[1], g_list[2].edges()[1])
    assert F.allclose(load_g.edata['e1'], g_list[2].edata['e1'])
    assert F.allclose(load_g.edata['e2'], g_list[2].edata['e2'])
    assert F.allclose(load_g.ndata['n1'], g_list[2].ndata['n1'])

test_graph_serialize()
