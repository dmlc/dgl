import networkx as nx
import numpy as np
import dgl
import dgl.function as fn
import backend as F

D = 5

# line graph related
def test_line_graph():
    N = 5
    G = dgl.DGLGraph(nx.star_graph(N))
    G.edata['h'] = F.randn((2 * N, D))
    n_edges = G.number_of_edges()
    L = G.line_graph(shared=True)
    assert L.number_of_nodes() == 2 * N
    L.ndata['h'] = F.randn((2 * N, D))
    # update node features on line graph should reflect to edge features on
    # original graph.
    u = [0, 0, 2, 3]
    v = [1, 2, 0, 0]
    eid = G.edge_ids(u, v)
    L.nodes[eid].data['h'] = F.zeros((4, D))
    assert F.allclose(G.edges[u, v].data['h'], F.zeros((4, D)))

    # adding a new node feature on line graph should also reflect to a new
    # edge feature on original graph
    data = F.randn((n_edges, D))
    L.ndata['w'] = data
    assert F.allclose(G.edata['w'], data)

def test_no_backtracking():
    N = 5
    G = dgl.DGLGraph(nx.star_graph(N))
    L = G.line_graph(backtracking=False)
    assert L.number_of_nodes() == 2 * N
    for i in range(1, N):
        e1 = G.edge_id(0, i)
        e2 = G.edge_id(i, 0)
        assert not L.has_edge_between(e1, e2)
        assert not L.has_edge_between(e2, e1)

# reverse graph related
def test_reverse():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    # The graph need not to be completely connected.
    g.add_edges([0, 1, 2], [1, 2, 1])
    g.ndata['h'] = F.tensor([[0.], [1.], [2.], [3.], [4.]])
    g.edata['h'] = F.tensor([[5.], [6.], [7.]])
    rg = g.reverse()

    assert g.is_multigraph == rg.is_multigraph

    assert g.number_of_nodes() == rg.number_of_nodes()
    assert g.number_of_edges() == rg.number_of_edges()
    assert F.allclose(F.astype(rg.has_edges_between([1, 2, 1], [0, 1, 2]), F.float32), F.ones((3,)))
    assert g.edge_id(0, 1) == rg.edge_id(1, 0)
    assert g.edge_id(1, 2) == rg.edge_id(2, 1)
    assert g.edge_id(2, 1) == rg.edge_id(1, 2)

def test_reverse_shared_frames():
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 1, 2], [1, 2, 1])
    g.ndata['h'] = F.tensor([[0.], [1.], [2.]])
    g.edata['h'] = F.tensor([[3.], [4.], [5.]])

    rg = g.reverse(share_ndata=True, share_edata=True)
    assert F.allclose(g.ndata['h'], rg.ndata['h'])
    assert F.allclose(g.edata['h'], rg.edata['h'])
    assert F.allclose(g.edges[[0, 2], [1, 1]].data['h'],
                      rg.edges[[1, 1], [0, 2]].data['h'])

    rg.ndata['h'] = rg.ndata['h'] + 1
    assert F.allclose(rg.ndata['h'], g.ndata['h'])

    g.edata['h'] = g.edata['h'] - 1
    assert F.allclose(rg.edata['h'], g.edata['h'])

    src_msg = fn.copy_src(src='h', out='m')
    sum_reduce = fn.sum(msg='m', out='h')

    rg.update_all(src_msg, sum_reduce)
    assert F.allclose(g.ndata['h'], rg.ndata['h'])


if __name__ == '__main__':
    test_line_graph()
    test_no_backtracking()
    test_reverse()
    test_reverse_shared_frames()
