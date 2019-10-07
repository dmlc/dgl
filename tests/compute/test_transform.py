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
    assert F.allclose(F.astype(rg.has_edges_between(
        [1, 2, 1], [0, 1, 2]), F.float32), F.ones((3,)))
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


def test_simple_graph():
    elist = [(0, 1), (0, 2), (1, 2), (0, 1)]
    g = dgl.DGLGraph(elist, readonly=True)
    assert g.is_multigraph
    sg = dgl.to_simple_graph(g)
    assert not sg.is_multigraph
    assert sg.number_of_edges() == 3
    src, dst = sg.edges()
    eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
    assert eset == set(elist)


def test_bidirected_graph():
    def _test(in_readonly, out_readonly):
        elist = [(0, 0), (0, 1), (0, 1), (1, 0),
                 (1, 1), (2, 1), (2, 2), (2, 2)]
        g = dgl.DGLGraph(elist, readonly=in_readonly)
        elist.append((1, 2))
        elist = set(elist)
        big = dgl.to_bidirected(g, out_readonly)
        assert big.number_of_edges() == 10
        src, dst = big.edges()
        eset = set(zip(list(F.asnumpy(src)), list(F.asnumpy(dst))))
        assert eset == set(elist)

    _test(True, True)
    _test(True, False)
    _test(False, True)
    _test(False, False)


def test_khop_graph():
    N = 20
    feat = F.randn((N, 5))
    g = dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3))
    for k in range(4):
        g_k = dgl.khop_graph(g, k)
        # use original graph to do message passing for k times.
        g.ndata['h'] = feat
        for _ in range(k):
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h_0 = g.ndata.pop('h')
        # use k-hop graph to do message passing for one time.
        g_k.ndata['h'] = feat
        g_k.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h_1 = g_k.ndata.pop('h')
        assert F.allclose(h_0, h_1, rtol=1e-3, atol=1e-3)


def test_khop_adj():
    N = 20
    feat = F.randn((N, 5))
    g = dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3))
    for k in range(3):
        adj = F.tensor(dgl.khop_adj(g, k))
        # use original graph to do message passing for k times.
        g.ndata['h'] = feat
        for _ in range(k):
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h_0 = g.ndata.pop('h')
        # use k-hop adj to do message passing for one time.
        h_1 = F.matmul(adj, feat)
        assert F.allclose(h_0, h_1, rtol=1e-3, atol=1e-3)


def test_laplacian_lambda_max():
    N = 20
    eps = 1e-6
    # test DGLGraph
    g = dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3))
    l_max = dgl.laplacian_lambda_max(g)
    assert (l_max[0] < 2 + eps)
    # test BatchedDGLGraph
    N_arr = [20, 30, 10, 12]
    bg = dgl.batch([
        dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3))
        for N in N_arr
    ])
    l_max_arr = dgl.laplacian_lambda_max(bg)
    assert len(l_max_arr) == len(N_arr)
    for l_max in l_max_arr:
        assert l_max < 2 + eps


def test_add_self_loop():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([0, 1, 2], [1, 1, 2])
    # Nodes 0, 3, 4 don't have self-loop
    new_g = dgl.transform.add_self_loop(g)
    assert F.allclose(new_g.edges()[0], F.tensor([0, 0, 1, 2, 3, 4]))
    assert F.allclose(new_g.edges()[1], F.tensor([1, 0, 1, 2, 3, 4]))


def test_remove_self_loop():
    g = dgl.DGLGraph()
    g.add_nodes(5)
    g.add_edges([0, 1, 2], [1, 1, 2])
    new_g = dgl.transform.remove_self_loop(g)
    assert F.allclose(new_g.edges()[0], F.tensor([0]))
    assert F.allclose(new_g.edges()[1], F.tensor([1]))



if __name__ == '__main__':
    test_line_graph()
    test_no_backtracking()
    test_reverse()
    test_reverse_shared_frames()
    test_simple_graph()
    test_bidirected_graph()
    test_khop_adj()
    test_khop_graph()
    test_laplacian_lambda_max()
    test_remove_self_loop()
    test_add_self_loop()
