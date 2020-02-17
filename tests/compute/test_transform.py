from scipy import sparse as spsp
import unittest
import networkx as nx
import numpy as np
import dgl
import dgl.function as fn
import backend as F
from dgl.graph_index import from_scipy_sparse_matrix

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

def create_large_graph_index(num_nodes):
    row = np.random.choice(num_nodes, num_nodes * 10)
    col = np.random.choice(num_nodes, num_nodes * 10)
    spm = spsp.coo_matrix((np.ones(len(row)), (row, col)))
    return from_scipy_sparse_matrix(spm, True)

def get_nodeflow(g, node_ids, num_layers):
    batch_size = len(node_ids)
    expand_factor = g.number_of_nodes()
    sampler = dgl.contrib.sampling.NeighborSampler(g, batch_size,
            expand_factor=expand_factor, num_hops=num_layers,
            seed_nodes=node_ids)
    return next(iter(sampler))

def test_partition():
    g = dgl.DGLGraph(create_large_graph_index(1000), readonly=True)
    node_part = np.random.choice(4, g.number_of_nodes())
    subgs = dgl.transform.partition_graph_with_halo(g, node_part, 2)
    for part_id, subg in subgs.items():
        node_ids = np.nonzero(node_part == part_id)[0]
        lnode_ids = np.nonzero(F.asnumpy(subg.ndata['inner_node']))[0]
        nf = get_nodeflow(g, node_ids, 2)
        lnf = get_nodeflow(subg, lnode_ids, 2)
        for i in range(nf.num_layers):
            layer_nids1 = F.asnumpy(nf.layer_parent_nid(i))
            layer_nids2 = lnf.layer_parent_nid(i)
            layer_nids2 = F.asnumpy(F.gather_row(subg.parent_nid, layer_nids2))
            assert np.all(np.sort(layer_nids1) == np.sort(layer_nids2))

        for i in range(nf.num_blocks):
            block_eids1 = F.asnumpy(nf.block_parent_eid(i))
            block_eids2 = lnf.block_parent_eid(i)
            block_eids2 = F.asnumpy(F.gather_row(subg.parent_eid, block_eids2))
            assert np.all(np.sort(block_eids1) == np.sort(block_eids2))


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU compaction not implemented")
def test_compact():
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): [(1, 3), (3, 5)],
        ('user', 'plays', 'game'): [(2, 4), (3, 4), (2, 5)],
        ('game', 'wished-by', 'user'): [(6, 7), (5, 7)]},
        {'user': 20, 'game': 10})

    g2 = dgl.heterograph({
        ('game', 'clicked-by', 'user'): [(3, 1)],
        ('user', 'likes', 'user'): [(1, 8), (8, 9)]},
        {'user': 20, 'game': 10})

    g3 = dgl.graph([(0, 1), (1, 2)], card=10, ntype='user')
    g4 = dgl.graph([(1, 3), (3, 5)], card=10, ntype='user')

    def _check(g, new_g, induced_nodes):
        assert g.ntypes == new_g.ntypes
        assert g.canonical_etypes == new_g.canonical_etypes

        for ntype in g.ntypes:
            assert -1 not in induced_nodes[ntype]

        for etype in g.canonical_etypes:
            g_src, g_dst = g.all_edges(order='eid', etype=etype)
            g_src = F.asnumpy(g_src)
            g_dst = F.asnumpy(g_dst)
            new_g_src, new_g_dst = new_g.all_edges(order='eid', etype=etype)
            new_g_src_mapped = induced_nodes[etype[0]][F.asnumpy(new_g_src)]
            new_g_dst_mapped = induced_nodes[etype[2]][F.asnumpy(new_g_dst)]
            assert (g_src == new_g_src_mapped).all()
            assert (g_dst == new_g_dst_mapped).all()

    # Test default
    new_g1 = dgl.compact_graphs(g1)
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes['game']) == set([4, 5, 6])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a dict
    new_g1 = dgl.compact_graphs(
        g1, always_preserve={'game': F.tensor([4, 7], dtype=F.int64)})
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes['game']) == set([4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a tensor
    new_g3 = dgl.compact_graphs(
        g3, always_preserve=F.tensor([1, 7], dtype=F.int64))
    induced_nodes = {ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([0, 1, 2, 7])
    _check(g3, new_g3, induced_nodes)

    # Test multiple graphs
    new_g1, new_g2 = dgl.compact_graphs([g1, g2])
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes['game']) == set([3, 4, 5, 6])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a dict
    new_g1, new_g2 = dgl.compact_graphs(
        [g1, g2], always_preserve={'game': F.tensor([4, 7], dtype=F.int64)})
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes['game']) == set([3, 4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a tensor
    new_g3, new_g4 = dgl.compact_graphs(
        [g3, g4], always_preserve=F.tensor([1, 7], dtype=F.int64))
    induced_nodes = {ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([0, 1, 2, 3, 5, 7])
    _check(g3, new_g3, induced_nodes)
    _check(g4, new_g4, induced_nodes)


def test_to_simple():
    g = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 3), (2, 2), (1, 3), (1, 4), (1, 4)],
        ('user', 'plays', 'game'): [(3, 5), (2, 3), (1, 4), (1, 4), (3, 5), (2, 3), (2, 3)]})
    sg = dgl.to_simple(g, return_counts='weights', writeback_mapping=True)

    for etype in g.canonical_etypes:
        u, v = g.all_edges(form='uv', order='eid', etype=etype)
        u = F.asnumpy(u).tolist()
        v = F.asnumpy(v).tolist()
        uv = list(zip(u, v))
        eid_map = F.asnumpy(g.edges[etype].data[dgl.EID])

        su, sv = sg.all_edges(form='uv', order='eid', etype=etype)
        su = F.asnumpy(su).tolist()
        sv = F.asnumpy(sv).tolist()
        suv = list(zip(su, sv))
        sw = F.asnumpy(sg.edges[etype].data['weights'])

        assert set(uv) == set(suv)
        for i, e in enumerate(suv):
            assert sw[i] == sum(e == _e for _e in uv)
        for i, e in enumerate(uv):
            assert eid_map[i] == suv.index(e)


@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU array ops not implemented")
def test_select_topk():
    g = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (2, 1), (5, 1), (2, 2), (3, 2), (4, 2), (6, 2)],
        ('user', 'plays', 'game'): [(1, 0), (3, 1)]},
        {'user': 7, 'game': 3})     # include nodes with zero incoming edges
    follow_weights = np.array([2, 7, 3, 4, 6, 5, 1])
    plays_weights = np.array([2, 1])
    g.edges['follow'].data['weights'] = F.zerocopy_from_numpy(follow_weights)
    g.edges['plays'].data['weights'] = F.zerocopy_from_numpy(plays_weights)
    sg = dgl.select_topk(g, 'weights', 1)

    su, sv = sg.all_edges(form='uv', order='eid', etype='follow')
    su = F.asnumpy(su).tolist()
    sv = F.asnumpy(sv).tolist()
    suv = list(zip(su, sv))
    sw = F.asnumpy(sg.edges['follow'].data['weights'])
    induced_edges = F.asnumpy(sg.edges['follow'].data[dgl.EID])

    assert set(suv) == set([(2, 1), (3, 2)])
    for i, e in enumerate(suv):
        assert induced_edges[i] == g.edge_id(e[0], e[1], etype='follow')
        assert sw[i] == follow_weights[induced_edges[i]]

    su, sv = sg.all_edges(form='uv', order='eid', etype='plays')
    su = F.asnumpy(su).tolist()
    sv = F.asnumpy(sv).tolist()
    suv = list(zip(su, sv))
    sw = F.asnumpy(sg.edges['plays'].data['weights'])
    induced_edges = F.asnumpy(sg.edges['plays'].data[dgl.EID])

    assert set(suv) == set([(1, 0), (3, 1)])
    for i, e in enumerate(suv):
        assert induced_edges[i] == g.edge_id(e[0], e[1], etype='plays')
        assert sw[i] == plays_weights[induced_edges[i]]

    g = dgl.heterograph({
        ('user', 'follow', 'user'): [(1, 0), (1, 2), (1, 5), (2, 2), (2, 3), (2, 4), (2, 6)],
        ('user', 'plays', 'game'): [(0, 1), (1, 3)]},
        {'user': 7, 'game': 4})     # include nodes with zero incoming edges
    g.edges['follow'].data['weights'] = F.arange(1, 8)
    g.edges['plays'].data['weights'] = F.arange(1, 3)
    follow_weights = np.arange(1, 8)
    plays_weights = np.arange(1, 3)
    sg = dgl.select_topk(g, 'weights', 2, inbound=False)

    su, sv = sg.all_edges(form='uv', order='eid', etype='follow')
    su = F.asnumpy(su).tolist()
    sv = F.asnumpy(sv).tolist()
    suv = list(zip(su, sv))
    sw = F.asnumpy(sg.edges['follow'].data['weights'])
    induced_edges = F.asnumpy(sg.edges['follow'].data[dgl.EID])

    assert set(suv) == set([(1, 2), (1, 5), (2, 4), (2, 6)])
    for i, e in enumerate(suv):
        assert induced_edges[i] == g.edge_id(e[0], e[1], etype='follow')
        assert sw[i] == follow_weights[induced_edges[i]]

    su, sv = sg.all_edges(form='uv', order='eid', etype='plays')
    su = F.asnumpy(su).tolist()
    sv = F.asnumpy(sv).tolist()
    suv = list(zip(su, sv))
    sw = F.asnumpy(sg.edges['plays'].data['weights'])
    induced_edges = F.asnumpy(sg.edges['plays'].data[dgl.EID])

    assert set(suv) == set([(0, 1), (1, 3)])
    for i, e in enumerate(suv):
        assert induced_edges[i] == g.edge_id(e[0], e[1], etype='plays')
        assert sw[i] == plays_weights[induced_edges[i]]

    # random test
    x = spsp.random(10, 10, 0.5)
    w = np.random.permutation(50) + 1
    x.data = w
    g = dgl.graph(x)
    g.edata['w'] = F.zerocopy_from_numpy(w)

    sg = dgl.select_topk(g, 'w', 1)
    src, dst = sg.all_edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    sg_w = F.asnumpy(sg.edata['w'])
    ans_w = x.toarray().max(0)
    ans_src = x.toarray().argmax(0)

    assert np.array_equal(np.sort(dst), np.arange(10))
    assert np.array_equal(ans_src[dst], src)
    assert np.array_equal(ans_w[dst], sg_w)

    sg = dgl.select_topk(g, 'w', 1, inbound=False)
    src, dst = sg.all_edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    sg_w = F.asnumpy(sg.edata['w'])
    ans_w = x.toarray().max(1)
    ans_dst = x.toarray().argmax(1)

    assert np.array_equal(np.sort(src), np.arange(10))
    assert np.array_equal(ans_dst[src], dst)
    assert np.array_equal(ans_w[src], sg_w)

    neg_x = x.copy()
    neg_x.data = 100 - neg_x.data

    sg = dgl.select_topk(g, 'w', 1, smallest=True)
    src, dst = sg.all_edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    sg_w = F.asnumpy(sg.edata['w'])
    ans_w = 100 - neg_x.toarray().max(0)
    ans_src = neg_x.toarray().argmax(0)

    assert np.array_equal(np.sort(dst), np.arange(10))
    assert np.array_equal(ans_src[dst], src)
    assert np.array_equal(ans_w[dst], sg_w)

    sg = dgl.select_topk(g, 'w', 1, inbound=False, smallest=True)
    src, dst = sg.all_edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    sg_w = F.asnumpy(sg.edata['w'])
    ans_w = 100 - neg_x.toarray().max(1)
    ans_dst = neg_x.toarray().argmax(1)

    assert np.array_equal(np.sort(src), np.arange(10))
    assert np.array_equal(ans_dst[src], dst)
    assert np.array_equal(ans_w[src], sg_w)


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
    test_partition()
    test_compact()
    test_to_simple()
    test_select_topk()
