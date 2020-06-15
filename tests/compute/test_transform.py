from scipy import sparse as spsp
import unittest
import networkx as nx
import numpy as np
import dgl
import dgl.function as fn
import backend as F
from dgl.graph_index import from_scipy_sparse_matrix
import unittest
from utils import parametrize_dtype

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
        elist = [(0, 0), (0, 1), (1, 0),
                (1, 1), (2, 1), (2, 2)]
        num_edges = 7
        g = dgl.DGLGraph(elist, readonly=in_readonly)
        elist.append((1, 2))
        elist = set(elist)
        big = dgl.to_bidirected(g, out_readonly)
        assert big.number_of_edges() == num_edges
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

    def _test(g):
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

    # Test for random undirected graphs
    g = dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3))
    _test(g)
    # Test for random directed graphs
    g = dgl.DGLGraph(nx.erdos_renyi_graph(N, 0.3, directed=True))
    _test(g)

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
    # test batched DGLGraph
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

def test_partition_with_halo():
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

@unittest.skipIf(F._default_context_str == 'gpu', reason="METIS doesn't support GPU")
def test_metis_partition():
    # TODO(zhengda) Metis fails to partition a small graph.
    g = dgl.DGLGraph(create_large_graph_index(1000), readonly=True)
    check_metis_partition(g, 0)
    check_metis_partition(g, 1)
    check_metis_partition(g, 2)

def check_metis_partition(g, extra_hops):
    # partitions with 1-hop HALO nodes
    subgs = dgl.transform.metis_partition(g, 4, extra_cached_hops=extra_hops)
    num_inner_nodes = 0
    num_inner_edges = 0
    if subgs is not None:
        for part_id, subg in subgs.items():
            lnode_ids = np.nonzero(F.asnumpy(subg.ndata['inner_node']))[0]
            ledge_ids = np.nonzero(F.asnumpy(subg.edata['inner_edge']))[0]
            num_inner_nodes += len(lnode_ids)
            num_inner_edges += len(ledge_ids)
            assert np.sum(F.asnumpy(subg.ndata['part_id']) == part_id) == len(lnode_ids)
        assert num_inner_nodes == g.number_of_nodes()
        print(g.number_of_edges() - num_inner_edges)

    if extra_hops == 0:
        return

    # partitions with 1-hop HALO nodes and reshuffling nodes
    subgs = dgl.transform.metis_partition(g, 4, extra_cached_hops=extra_hops, reshuffle=True)
    num_inner_nodes = 0
    num_inner_edges = 0
    edge_cnts = np.zeros((g.number_of_edges(),))
    if subgs is not None:
        for part_id, subg in subgs.items():
            lnode_ids = np.nonzero(F.asnumpy(subg.ndata['inner_node']))[0]
            ledge_ids = np.nonzero(F.asnumpy(subg.edata['inner_edge']))[0]
            num_inner_nodes += len(lnode_ids)
            num_inner_edges += len(ledge_ids)
            assert np.sum(F.asnumpy(subg.ndata['part_id']) == part_id) == len(lnode_ids)
            nids = F.asnumpy(subg.ndata[dgl.NID])

            # ensure the local node Ids are contiguous.
            parent_ids = F.asnumpy(subg.ndata[dgl.NID])
            parent_ids = parent_ids[:len(lnode_ids)]
            assert np.all(parent_ids == np.arange(parent_ids[0], parent_ids[-1] + 1))

            # count the local edges.
            parent_ids = F.asnumpy(subg.edata[dgl.EID])[ledge_ids]
            edge_cnts[parent_ids] += 1

            orig_ids = subg.ndata['orig_id']
            inner_node = F.asnumpy(subg.ndata['inner_node'])
            for nid in range(subg.number_of_nodes()):
                neighs = subg.predecessors(nid)
                old_neighs1 = F.gather_row(orig_ids, neighs)
                old_nid = F.asnumpy(orig_ids[nid])
                old_neighs2 = g.predecessors(old_nid)
                # If this is an inner node, it should have the full neighborhood.
                if inner_node[nid]:
                    assert np.all(np.sort(F.asnumpy(old_neighs1)) == np.sort(F.asnumpy(old_neighs2)))
        # Normally, local edges are only counted once.
        assert np.all(edge_cnts == 1)

        assert num_inner_nodes == g.number_of_nodes()
        print(g.number_of_edges() - num_inner_edges)

@unittest.skipIf(F._default_context_str == 'gpu', reason="It doesn't support GPU")
def test_reorder_nodes():
    g = dgl.DGLGraph(create_large_graph_index(1000), readonly=True)
    new_nids = np.random.permutation(g.number_of_nodes())
    # TODO(zhengda) we need to test both CSR and COO.
    new_g = dgl.transform.reorder_nodes(g, new_nids)
    new_in_deg = new_g.in_degrees()
    new_out_deg = new_g.out_degrees()
    in_deg = g.in_degrees()
    out_deg = g.out_degrees()
    new_in_deg1 = F.scatter_row(in_deg, F.tensor(new_nids), in_deg)
    new_out_deg1 = F.scatter_row(out_deg, F.tensor(new_nids), out_deg)
    assert np.all(F.asnumpy(new_in_deg == new_in_deg1))
    assert np.all(F.asnumpy(new_out_deg == new_out_deg1))
    orig_ids = F.asnumpy(new_g.ndata['orig_id'])
    for nid in range(new_g.number_of_nodes()):
        neighs = F.asnumpy(new_g.successors(nid))
        old_neighs1 = orig_ids[neighs]
        old_nid = orig_ids[nid]
        old_neighs2 = g.successors(old_nid)
        assert np.all(np.sort(old_neighs1) == np.sort(F.asnumpy(old_neighs2)))

        neighs = F.asnumpy(new_g.predecessors(nid))
        old_neighs1 = orig_ids[neighs]
        old_nid = orig_ids[nid]
        old_neighs2 = g.predecessors(old_nid)
        assert np.all(np.sort(old_neighs1) == np.sort(F.asnumpy(old_neighs2)))

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
@parametrize_dtype
def test_in_subgraph(index_dtype):
    g1 = dgl.graph([(1,0),(2,0),(3,0),(0,1),(2,1),(3,1),(0,2)], 'user', 'follow', index_dtype=index_dtype)
    g2 = dgl.bipartite([(0,0),(0,1),(1,2),(3,2)], 'user', 'play', 'game', index_dtype=index_dtype)
    g3 = dgl.bipartite([(2,0),(2,1),(2,2),(1,0),(1,3),(0,0)], 'game', 'liked-by', 'user', index_dtype=index_dtype)
    g4 = dgl.bipartite([(0,0),(1,0),(2,0),(3,0)], 'user', 'flips', 'coin', index_dtype=index_dtype)
    hg = dgl.hetero_from_relations([g1, g2, g3, g4])
    subg = dgl.in_subgraph(hg, {'user' : [0,1], 'game' : 0})
    assert subg._idtype_str == index_dtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4
    u, v = subg['follow'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(hg['follow'].edge_ids(u, v), subg['follow'].edata[dgl.EID])
    assert edge_set == {(1,0),(2,0),(3,0),(0,1),(2,1),(3,1)}
    u, v = subg['play'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(hg['play'].edge_ids(u, v), subg['play'].edata[dgl.EID])
    assert edge_set == {(0,0)}
    u, v = subg['liked-by'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert F.array_equal(hg['liked-by'].edge_ids(u, v), subg['liked-by'].edata[dgl.EID])
    assert edge_set == {(2,0),(2,1),(1,0),(0,0)}
    assert subg['flips'].number_of_edges() == 0

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
@parametrize_dtype
def test_out_subgraph(index_dtype):
    g1 = dgl.graph([(1,0),(2,0),(3,0),(0,1),(2,1),(3,1),(0,2)], 'user', 'follow', index_dtype=index_dtype)
    g2 = dgl.bipartite([(0,0),(0,1),(1,2),(3,2)], 'user', 'play', 'game', index_dtype=index_dtype)
    g3 = dgl.bipartite([(2,0),(2,1),(2,2),(1,0),(1,3),(0,0)], 'game', 'liked-by', 'user', index_dtype=index_dtype)
    g4 = dgl.bipartite([(0,0),(1,0),(2,0),(3,0)], 'user', 'flips', 'coin', index_dtype=index_dtype)
    hg = dgl.hetero_from_relations([g1, g2, g3, g4])
    subg = dgl.out_subgraph(hg, {'user' : [0,1], 'game' : 0})
    assert subg._idtype_str == index_dtype
    assert len(subg.ntypes) == 3
    assert len(subg.etypes) == 4
    u, v = subg['follow'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(1,0),(0,1),(0,2)}
    assert F.array_equal(hg['follow'].edge_ids(u, v), subg['follow'].edata[dgl.EID])
    u, v = subg['play'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0,0),(0,1),(1,2)}
    assert F.array_equal(hg['play'].edge_ids(u, v), subg['play'].edata[dgl.EID])
    u, v = subg['liked-by'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0,0)}
    assert F.array_equal(hg['liked-by'].edge_ids(u, v), subg['liked-by'].edata[dgl.EID])
    u, v = subg['flips'].edges()
    edge_set = set(zip(list(F.asnumpy(u)), list(F.asnumpy(v))))
    assert edge_set == {(0,0),(1,0)}
    assert F.array_equal(hg['flips'].edge_ids(u, v), subg['flips'].edata[dgl.EID])

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU compaction not implemented")
@parametrize_dtype
def test_compact(index_dtype):
    g1 = dgl.heterograph({
        ('user', 'follow', 'user'): [(1, 3), (3, 5)],
        ('user', 'plays', 'game'): [(2, 4), (3, 4), (2, 5)],
        ('game', 'wished-by', 'user'): [(6, 7), (5, 7)]},
        {'user': 20, 'game': 10}, index_dtype=index_dtype)

    g2 = dgl.heterograph({
        ('game', 'clicked-by', 'user'): [(3, 1)],
        ('user', 'likes', 'user'): [(1, 8), (8, 9)]},
        {'user': 20, 'game': 10}, index_dtype=index_dtype)

    g3 = dgl.graph([(0, 1), (1, 2)], num_nodes=10, ntype='user', index_dtype=index_dtype)
    g4 = dgl.graph([(1, 3), (3, 5)], num_nodes=10, ntype='user', index_dtype=index_dtype)

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
    assert new_g1._idtype_str == index_dtype
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes['game']) == set([4, 5, 6])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a dict
    new_g1 = dgl.compact_graphs(
        g1, always_preserve={'game': F.tensor([4, 7], dtype=getattr(F, index_dtype))})
    assert new_g1._idtype_str == index_dtype
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7])
    assert set(induced_nodes['game']) == set([4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)

    # Test with always_preserve given a tensor
    new_g3 = dgl.compact_graphs(
        g3, always_preserve=F.tensor([1, 7], dtype=getattr(F, index_dtype)))
    induced_nodes = {ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    
    assert new_g3._idtype_str == index_dtype
    assert set(induced_nodes['user']) == set([0, 1, 2, 7])
    _check(g3, new_g3, induced_nodes)

    # Test multiple graphs
    new_g1, new_g2 = dgl.compact_graphs([g1, g2])
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    assert new_g1._idtype_str == index_dtype
    assert new_g2._idtype_str == index_dtype
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes['game']) == set([3, 4, 5, 6])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a dict
    new_g1, new_g2 = dgl.compact_graphs(
        [g1, g2], always_preserve={'game': F.tensor([4, 7], dtype=getattr(F, index_dtype))})
    induced_nodes = {ntype: new_g1.nodes[ntype].data[dgl.NID] for ntype in new_g1.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}    
    assert new_g1._idtype_str == index_dtype
    assert new_g2._idtype_str == index_dtype
    assert set(induced_nodes['user']) == set([1, 3, 5, 2, 7, 8, 9])
    assert set(induced_nodes['game']) == set([3, 4, 5, 6, 7])
    _check(g1, new_g1, induced_nodes)
    _check(g2, new_g2, induced_nodes)

    # Test multiple graphs with always_preserve given a tensor
    new_g3, new_g4 = dgl.compact_graphs(
        [g3, g4], always_preserve=F.tensor([1, 7], dtype=getattr(F, index_dtype)))
    induced_nodes = {ntype: new_g3.nodes[ntype].data[dgl.NID] for ntype in new_g3.ntypes}
    induced_nodes = {k: F.asnumpy(v) for k, v in induced_nodes.items()}
    
    assert new_g3._idtype_str == index_dtype
    assert new_g4._idtype_str == index_dtype
    assert set(induced_nodes['user']) == set([0, 1, 2, 3, 5, 7])
    _check(g3, new_g3, induced_nodes)
    _check(g4, new_g4, induced_nodes)

@parametrize_dtype
def test_to_simple(index_dtype):
    g = dgl.heterograph({
        ('user', 'follow', 'user'): [(0, 1), (1, 3), (2, 2), (1, 3), (1, 4), (1, 4)],
        ('user', 'plays', 'game'): [(3, 5), (2, 3), (1, 4), (1, 4), (3, 5), (2, 3), (2, 3)]}, index_dtype=index_dtype)
    sg = dgl.to_simple(g, return_counts='weights', writeback_mapping='new_eid')

    for etype in g.canonical_etypes:
        u, v = g.all_edges(form='uv', order='eid', etype=etype)
        u = F.asnumpy(u).tolist()
        v = F.asnumpy(v).tolist()
        uv = list(zip(u, v))
        eid_map = F.asnumpy(g.edges[etype].data['new_eid'])

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

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU compaction not implemented")
@parametrize_dtype
def test_to_block(index_dtype):
    def check(g, bg, ntype, etype, dst_nodes, include_dst_in_src=True):
        if dst_nodes is not None:
            assert F.array_equal(bg.dstnodes[ntype].data[dgl.NID], dst_nodes)
        n_dst_nodes = bg.number_of_nodes('DST/' + ntype)
        if include_dst_in_src:
            assert F.array_equal(
                bg.srcnodes[ntype].data[dgl.NID][:n_dst_nodes],
                bg.dstnodes[ntype].data[dgl.NID])

        g = g[etype]
        bg = bg[etype]
        induced_src = bg.srcdata[dgl.NID]
        induced_dst = bg.dstdata[dgl.NID]
        induced_eid = bg.edata[dgl.EID]
        bg_src, bg_dst = bg.all_edges(order='eid')
        src_ans, dst_ans = g.all_edges(order='eid')

        induced_src_bg = F.gather_row(induced_src, bg_src)
        induced_dst_bg = F.gather_row(induced_dst, bg_dst)
        induced_src_ans = F.gather_row(src_ans, induced_eid)
        induced_dst_ans = F.gather_row(dst_ans, induced_eid)

        assert F.array_equal(induced_src_bg, induced_src_ans)
        assert F.array_equal(induced_dst_bg, induced_dst_ans)

    def checkall(g, bg, dst_nodes, include_dst_in_src=True):
        for etype in g.etypes:
            ntype = g.to_canonical_etype(etype)[2]
            if dst_nodes is not None and ntype in dst_nodes:
                check(g, bg, ntype, etype, dst_nodes[ntype], include_dst_in_src)
            else:
                check(g, bg, ntype, etype, None, include_dst_in_src)

    g = dgl.heterograph({
        ('A', 'AA', 'A'): [(0, 1), (2, 3), (1, 2), (3, 4)],
        ('A', 'AB', 'B'): [(0, 1), (1, 3), (3, 5), (1, 6)],
        ('B', 'BA', 'A'): [(2, 3), (3, 2)]}, index_dtype=index_dtype)
    g_a = g['AA']

    bg = dgl.to_block(g_a)
    check(g_a, bg, 'A', 'AA', None)
    assert bg.number_of_src_nodes() == 5
    assert bg.number_of_dst_nodes() == 4

    bg = dgl.to_block(g_a, include_dst_in_src=False)
    check(g_a, bg, 'A', 'AA', None, False)
    assert bg.number_of_src_nodes() == 4
    assert bg.number_of_dst_nodes() == 4

    dst_nodes = F.tensor([3, 4], dtype=getattr(F, index_dtype))
    bg = dgl.to_block(g_a, dst_nodes)
    check(g_a, bg, 'A', 'AA', dst_nodes)

    dst_nodes = F.tensor([4, 3, 2, 1], dtype=getattr(F, index_dtype))
    bg = dgl.to_block(g_a, dst_nodes)
    check(g_a, bg, 'A', 'AA', dst_nodes)

    g_ab = g['AB']

    bg = dgl.to_block(g_ab)
    assert bg._idtype_str == index_dtype
    assert bg.number_of_nodes('SRC/B') == 4
    assert F.array_equal(bg.srcnodes['B'].data[dgl.NID], bg.dstnodes['B'].data[dgl.NID])
    assert bg.number_of_nodes('DST/A') == 0
    checkall(g_ab, bg, None)

    dst_nodes = {'B': F.tensor([5, 6], dtype=getattr(F, index_dtype))}
    bg = dgl.to_block(g, dst_nodes)
    assert bg.number_of_nodes('SRC/B') == 2
    assert F.array_equal(bg.srcnodes['B'].data[dgl.NID], bg.dstnodes['B'].data[dgl.NID])
    assert bg.number_of_nodes('DST/A') == 0
    checkall(g, bg, dst_nodes)

    dst_nodes = {'A': F.tensor([3, 4], dtype=getattr(F, index_dtype)), 'B': F.tensor([5, 6], dtype=getattr(F, index_dtype))}
    bg = dgl.to_block(g, dst_nodes)
    checkall(g, bg, dst_nodes)

    dst_nodes = {'A': F.tensor([4, 3, 2, 1], dtype=getattr(F, index_dtype)), 'B': F.tensor([3, 5, 6, 1], dtype=getattr(F, index_dtype))}
    bg = dgl.to_block(g, dst_nodes=dst_nodes)
    checkall(g, bg, dst_nodes)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
@parametrize_dtype
def test_remove_edges(index_dtype):
    def check(g1, etype, g, edges_removed):
        src, dst, eid = g.edges(etype=etype, form='all')
        src1, dst1 = g1.edges(etype=etype, order='eid')
        if etype is not None:
            eid1 = g1.edges[etype].data[dgl.EID]
        else:
            eid1 = g1.edata[dgl.EID]
        src1 = F.asnumpy(src1)
        dst1 = F.asnumpy(dst1)
        eid1 = F.asnumpy(eid1)
        src = F.asnumpy(src)
        dst = F.asnumpy(dst)
        eid = F.asnumpy(eid)
        sde_set = set(zip(src, dst, eid))

        for s, d, e in zip(src1, dst1, eid1):
            assert (s, d, e) in sde_set
        assert not np.isin(edges_removed, eid1).any()
        assert g1.idtype == g.idtype

    for fmt in ['coo', 'csr', 'csc']:
        for edges_to_remove in [[2], [2, 2], [3, 2], [1, 3, 1, 2]]:
            g = dgl.graph([(0, 1), (2, 3), (1, 2), (3, 4)], restrict_format=fmt, index_dtype=index_dtype)
            g1 = dgl.remove_edges(g, F.tensor(edges_to_remove, getattr(F, index_dtype)))
            check(g1, None, g, edges_to_remove)

            g = dgl.graph(
                spsp.csr_matrix(([1, 1, 1, 1], ([0, 2, 1, 3], [1, 3, 2, 4])), shape=(5, 5)),
                restrict_format=fmt, index_dtype=index_dtype)
            g1 = dgl.remove_edges(g, F.tensor(edges_to_remove, getattr(F, index_dtype)))
            check(g1, None, g, edges_to_remove)

    g = dgl.heterograph({
        ('A', 'AA', 'A'): [(0, 1), (2, 3), (1, 2), (3, 4)],
        ('A', 'AB', 'B'): [(0, 1), (1, 3), (3, 5), (1, 6)],
        ('B', 'BA', 'A'): [(2, 3), (3, 2)]}, index_dtype=index_dtype)
    g2 = dgl.remove_edges(g, {'AA': F.tensor([2], getattr(F, index_dtype)), 'AB': F.tensor([3], getattr(F, index_dtype)), 'BA': F.tensor([1], getattr(F, index_dtype))})
    check(g2, 'AA', g, [2])
    check(g2, 'AB', g, [3])
    check(g2, 'BA', g, [1])

    g3 = dgl.remove_edges(g, {'AA': F.tensor([], getattr(F, index_dtype)), 'AB': F.tensor([3], getattr(F, index_dtype)), 'BA': F.tensor([1], getattr(F, index_dtype))})
    check(g3, 'AA', g, [])
    check(g3, 'AB', g, [3])
    check(g3, 'BA', g, [1])

    g4 = dgl.remove_edges(g, {'AB': F.tensor([3, 1, 2, 0], getattr(F, index_dtype))})
    check(g4, 'AA', g, [])
    check(g4, 'AB', g, [3, 1, 2, 0])
    check(g4, 'BA', g, [])

def test_cast():
    m = spsp.coo_matrix(([1, 1], ([0, 1], [1, 2])), (4, 4))
    g = dgl.DGLGraph(m, readonly=True)
    gsrc, gdst = g.edges(order='eid')
    ndata = F.randn((4, 5))
    edata = F.randn((2, 4))
    g.ndata['x'] = ndata
    g.edata['y'] = edata

    hg = dgl.as_heterograph(g, 'A', 'AA')
    assert hg.ntypes == ['A']
    assert hg.etypes == ['AA']
    assert hg.canonical_etypes == [('A', 'AA', 'A')]
    assert hg.number_of_nodes() == 4
    assert hg.number_of_edges() == 2
    hgsrc, hgdst = hg.edges(order='eid')
    assert F.array_equal(gsrc, hgsrc)
    assert F.array_equal(gdst, hgdst)

    g2 = dgl.as_immutable_graph(hg)
    assert g2.number_of_nodes() == 4
    assert g2.number_of_edges() == 2
    g2src, g2dst = hg.edges(order='eid')
    assert F.array_equal(g2src, gsrc)
    assert F.array_equal(g2dst, gdst)

if __name__ == '__main__':
    test_reorder_nodes()
    # test_line_graph()
    # test_no_backtracking()
    # test_reverse()
    # test_reverse_shared_frames()
    # test_simple_graph()
    # test_bidirected_graph()
    # test_khop_adj()
    # test_khop_graph()
    # test_laplacian_lambda_max()
    # test_remove_self_loop()
    # test_add_self_loop()
    # test_partition_with_halo()
    test_metis_partition()
    # test_compact()
    # test_to_simple()
    # test_in_subgraph("int32")
    # test_out_subgraph()
    test_to_block("int32")
    # test_remove_edges()
