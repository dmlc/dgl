import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx

def create_test_heterograph():
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)

    follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
    plays_g = dgl.bipartite(plays_spmat, 'user', 'plays', 'game')
    wishes_g = dgl.bipartite(wishes_nx, 'user', 'wishes', 'game')
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')
    g = dgl.hetero_from_relations([follows_g, plays_g, wishes_g, develops_g])
    return g

def create_test_heterograph1():
    edges = []
    edges.extend([(0,1), (1,2)])  # follows
    edges.extend([(0,3), (1,3), (2,4), (1,4)])  # plays
    edges.extend([(0,4), (2,3)])  # wishes
    edges.extend([(5,3), (6,4)])  # develops
    ntypes = F.tensor([0, 0, 0, 1, 1, 2, 2])
    etypes = F.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    g0 = dgl.graph(edges)
    g0.ndata[dgl.NTYPE] = ntypes
    g0.edata[dgl.ETYPE] = etypes
    return dgl.to_hetero(g0, ['user', 'game', 'developer'], ['follows', 'plays', 'wishes', 'develops'])

def create_test_heterograph2():
    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')

    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): plays_spmat,
        ('user', 'wishes', 'game'): wishes_nx,
        ('developer', 'develops', 'game'): develops_g,
        })
    return g

def get_redfn(name):
    return getattr(F, name)

def test_create():
    g0 = create_test_heterograph()
    g1 = create_test_heterograph1()
    g2 = create_test_heterograph2()
    assert set(g0.ntypes) == set(g1.ntypes) == set(g2.ntypes)
    assert set(g0.canonical_etypes) == set(g1.canonical_etypes) == set(g2.canonical_etypes)

    # create from nx complete bipartite graph
    nxg = nx.complete_bipartite_graph(3, 4)
    g = dgl.bipartite(nxg, 'user', 'plays', 'game')
    assert g.ntypes == ['user', 'game']
    assert g.etypes == ['plays']
    assert g.number_of_edges() == 12

    # create from scipy
    spmat = ssp.coo_matrix(([1,1,1], ([0, 0, 1], [2, 3, 2])), shape=(4, 4))
    g = dgl.graph(spmat)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3

    # test inferring number of nodes for heterograph
    g = dgl.heterograph({
        ('l0', 'e0', 'l1'): [(0, 1), (0, 2)],
        ('l0', 'e1', 'l2'): [(2, 2)],
        ('l2', 'e2', 'l2'): [(1, 1), (3, 3)],
        })
    assert g.number_of_nodes('l0') == 3
    assert g.number_of_nodes('l1') == 3
    assert g.number_of_nodes('l2') == 4

def test_query():
    g = create_test_heterograph()

    ntypes = ['user', 'game', 'developer']
    canonical_etypes = [
        ('user', 'follows', 'user'),
        ('user', 'plays', 'game'),
        ('user', 'wishes', 'game'),
        ('developer', 'develops', 'game')]
    etypes = ['follows', 'plays', 'wishes', 'develops']

    # node & edge types
    assert set(ntypes) == set(g.ntypes)
    assert set(etypes) == set(g.etypes)
    assert set(canonical_etypes) == set(g.canonical_etypes)

    # metagraph
    mg = g.metagraph
    assert set(g.ntypes) == set(mg.nodes)
    etype_triplets = [(u, v, e) for u, v, e in mg.edges(keys=True)]
    assert set([
        ('user', 'user', 'follows'),
        ('user', 'game', 'plays'),
        ('user', 'game', 'wishes'),
        ('developer', 'game', 'develops')]) == set(etype_triplets)
    for i in range(len(etypes)):
        assert g.to_canonical_etype(etypes[i]) == canonical_etypes[i]

    # number of nodes
    assert [g.number_of_nodes(ntype) for ntype in ntypes] == [3, 2, 2]

    # number of edges
    assert [g.number_of_edges(etype) for etype in etypes] == [2, 4, 2, 2]

    assert not g.is_multigraph
    assert g.is_readonly

    # has_node & has_nodes
    for ntype in ntypes:
        n = g.number_of_nodes(ntype)
        for i in range(n):
            assert g.has_node(i, ntype)
        assert not g.has_node(n, ntype)
        assert np.array_equal(
            F.asnumpy(g.has_nodes([0, n], ntype)).astype('int32'), [1, 0])

    def _test(g):
        for etype in etypes:
            srcs, dsts = edges[etype]
            for src, dst in zip(srcs, dsts):
                assert g.has_edge_between(src, dst, etype)
            assert F.asnumpy(g.has_edges_between(srcs, dsts, etype)).all()

            srcs, dsts = negative_edges[etype]
            for src, dst in zip(srcs, dsts):
                assert not g.has_edge_between(src, dst, etype)
            assert not F.asnumpy(g.has_edges_between(srcs, dsts, etype)).any()

            srcs, dsts = edges[etype]
            n_edges = len(srcs)

            # predecessors & in_edges & in_degree
            pred = [s for s, d in zip(srcs, dsts) if d == 0]
            assert set(F.asnumpy(g.predecessors(0, etype)).tolist()) == set(pred)
            u, v = g.in_edges([0], etype=etype)
            assert F.asnumpy(v).tolist() == [0] * len(pred)
            assert set(F.asnumpy(u).tolist()) == set(pred)
            assert g.in_degree(0, etype) == len(pred)

            # successors & out_edges & out_degree
            succ = [d for s, d in zip(srcs, dsts) if s == 0]
            assert set(F.asnumpy(g.successors(0, etype)).tolist()) == set(succ)
            u, v = g.out_edges([0], etype=etype)
            assert F.asnumpy(u).tolist() == [0] * len(succ)
            assert set(F.asnumpy(v).tolist()) == set(succ)
            assert g.out_degree(0, etype) == len(succ)

            # edge_id & edge_ids
            for i, (src, dst) in enumerate(zip(srcs, dsts)):
                assert g.edge_id(src, dst, etype=etype) == i
                assert F.asnumpy(g.edge_id(src, dst, etype=etype, force_multi=True)).tolist() == [i]
            assert F.asnumpy(g.edge_ids(srcs, dsts, etype=etype)).tolist() == list(range(n_edges))
            u, v, e = g.edge_ids(srcs, dsts, etype=etype, force_multi=True)
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts
            assert F.asnumpy(e).tolist() == list(range(n_edges))

            # find_edges
            u, v = g.find_edges(list(range(n_edges)), etype)
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts

            # all_edges.
            for order in ['eid']:
                u, v, e = g.all_edges('all', order, etype)
                assert F.asnumpy(u).tolist() == srcs
                assert F.asnumpy(v).tolist() == dsts
                assert F.asnumpy(e).tolist() == list(range(n_edges))

            # in_degrees & out_degrees
            in_degrees = F.asnumpy(g.in_degrees(etype=etype))
            out_degrees = F.asnumpy(g.out_degrees(etype=etype))
            src_count = Counter(srcs)
            dst_count = Counter(dsts)
            utype, _, vtype = g.to_canonical_etype(etype)
            for i in range(g.number_of_nodes(utype)):
                assert out_degrees[i] == src_count[i]
            for i in range(g.number_of_nodes(vtype)):
                assert in_degrees[i] == dst_count[i]

    edges = {
        'follows': ([0, 1], [1, 2]),
        'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
        'wishes': ([0, 2], [1, 0]),
        'develops': ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        'follows': ([0, 1], [0, 1]),
        'plays': ([0, 2], [1, 0]),
        'wishes': ([0, 1], [0, 1]),
        'develops': ([0, 1], [1, 0]),
    }
    g = create_test_heterograph()
    _test(g)
    g = create_test_heterograph1()
    _test(g)

    etypes = canonical_etypes
    edges = {
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        ('user', 'follows', 'user'): ([0, 1], [0, 1]),
        ('user', 'plays', 'game'): ([0, 2], [1, 0]),
        ('user', 'wishes', 'game'): ([0, 1], [0, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [1, 0]),
        }
    g = create_test_heterograph()
    _test(g)
    g = create_test_heterograph1()
    _test(g)

    # test repr
    print(g)

def test_adj():
    g = create_test_heterograph()
    adj = F.sparse_to_numpy(g.adj(etype='follows'))
    assert np.allclose(
            adj,
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = F.sparse_to_numpy(g.adj(transpose=True, etype='follows'))
    assert np.allclose(
            adj,
            np.array([[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]]))
    adj = F.sparse_to_numpy(g.adj(etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = F.sparse_to_numpy(g.adj(transpose=True, etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 0.],
                      [1., 1.],
                      [0., 1.]]))

    adj = g.adj(scipy_fmt='csr', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(scipy_fmt='coo', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(scipy_fmt='csr', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = g.adj(scipy_fmt='coo', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = F.sparse_to_numpy(g['follows'].adj())
    assert np.allclose(
            adj,
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))

def test_inc():
    g = create_test_heterograph()
    #follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
    adj = F.sparse_to_numpy(g['follows'].inc('in'))
    assert np.allclose(
            adj,
            np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.]]))
    adj = F.sparse_to_numpy(g['follows'].inc('out'))
    assert np.allclose(
            adj,
            np.array([[1., 0.],
                      [0., 1.],
                      [0., 0.]]))
    adj = F.sparse_to_numpy(g['follows'].inc('both'))
    assert np.allclose(
            adj,
            np.array([[-1., 0.],
                      [1., -1.],
                      [0., 1.]]))
    adj = F.sparse_to_numpy(g.inc('in', etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 1., 0., 0.],
                      [0., 0., 1., 1.]]))
    adj = F.sparse_to_numpy(g.inc('out', etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]))
    adj = F.sparse_to_numpy(g.inc('both', etype='follows'))
    assert np.allclose(
            adj,
            np.array([[-1., 0.],
                      [1., -1.],
                      [0., 1.]]))
    
def test_view():
    # test data view
    g = create_test_heterograph()

    f1 = F.randn((3, 6))
    g.nodes['user'].data['h'] = f1       # ok
    f2 = g.nodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(F.tensor(g.nodes('user')), F.arange(0, 3))

    f3 = F.randn((2, 4))
    g.edges['user', 'follows', 'user'].data['h'] = f3
    f4 = g.edges['user', 'follows', 'user'].data['h']
    f5 = g.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(f3, f5)
    assert F.array_equal(F.tensor(g.edges(etype='follows', form='eid')), F.arange(0, 2))

def test_view1():
    # test relation view
    HG = create_test_heterograph()
    ntypes = ['user', 'game', 'developer']
    canonical_etypes = [
        ('user', 'follows', 'user'),
        ('user', 'plays', 'game'),
        ('user', 'wishes', 'game'),
        ('developer', 'develops', 'game')]
    etypes = ['follows', 'plays', 'wishes', 'develops']

    def _test_query():
        for etype in etypes:
            utype, _, vtype = HG.to_canonical_etype(etype)
            g = HG[etype]
            srcs, dsts = edges[etype]
            for src, dst in zip(srcs, dsts):
                assert g.has_edge_between(src, dst)
            assert F.asnumpy(g.has_edges_between(srcs, dsts)).all()

            srcs, dsts = negative_edges[etype]
            for src, dst in zip(srcs, dsts):
                assert not g.has_edge_between(src, dst)
            assert not F.asnumpy(g.has_edges_between(srcs, dsts)).any()

            srcs, dsts = edges[etype]
            n_edges = len(srcs)

            # predecessors & in_edges & in_degree
            pred = [s for s, d in zip(srcs, dsts) if d == 0]
            assert set(F.asnumpy(g.predecessors(0)).tolist()) == set(pred)
            u, v = g.in_edges([0])
            assert F.asnumpy(v).tolist() == [0] * len(pred)
            assert set(F.asnumpy(u).tolist()) == set(pred)
            assert g.in_degree(0) == len(pred)

            # successors & out_edges & out_degree
            succ = [d for s, d in zip(srcs, dsts) if s == 0]
            assert set(F.asnumpy(g.successors(0)).tolist()) == set(succ)
            u, v = g.out_edges([0])
            assert F.asnumpy(u).tolist() == [0] * len(succ)
            assert set(F.asnumpy(v).tolist()) == set(succ)
            assert g.out_degree(0) == len(succ)

            # edge_id & edge_ids
            for i, (src, dst) in enumerate(zip(srcs, dsts)):
                assert g.edge_id(src, dst) == i
                assert F.asnumpy(g.edge_id(src, dst, force_multi=True)).tolist() == [i]
            assert F.asnumpy(g.edge_ids(srcs, dsts)).tolist() == list(range(n_edges))
            u, v, e = g.edge_ids(srcs, dsts, force_multi=True)
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts
            assert F.asnumpy(e).tolist() == list(range(n_edges))

            # find_edges
            u, v = g.find_edges(list(range(n_edges)))
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts

            # all_edges.
            for order in ['eid']:
                u, v, e = g.all_edges(form='all', order=order)
                assert F.asnumpy(u).tolist() == srcs
                assert F.asnumpy(v).tolist() == dsts
                assert F.asnumpy(e).tolist() == list(range(n_edges))

            # in_degrees & out_degrees
            in_degrees = F.asnumpy(g.in_degrees())
            out_degrees = F.asnumpy(g.out_degrees())
            src_count = Counter(srcs)
            dst_count = Counter(dsts)
            for i in range(g.number_of_nodes(utype)):
                assert out_degrees[i] == src_count[i]
            for i in range(g.number_of_nodes(vtype)):
                assert in_degrees[i] == dst_count[i]   

    edges = {
        'follows': ([0, 1], [1, 2]),
        'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
        'wishes': ([0, 2], [1, 0]),
        'develops': ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        'follows': ([0, 1], [0, 1]),
        'plays': ([0, 2], [1, 0]),
        'wishes': ([0, 1], [0, 1]),
        'develops': ([0, 1], [1, 0]),
    }
    _test_query()
    etypes = canonical_etypes
    edges = {
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        ('user', 'follows', 'user'): ([0, 1], [0, 1]),
        ('user', 'plays', 'game'): ([0, 2], [1, 0]),
        ('user', 'wishes', 'game'): ([0, 1], [0, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [1, 0]),
        }
    _test_query()

    # test features
    HG.nodes['user'].data['h'] = F.ones((HG.number_of_nodes('user'), 5))
    HG.nodes['game'].data['m'] = F.ones((HG.number_of_nodes('game'), 3)) * 2

    # test only one node type
    g = HG['follows']
    assert g.number_of_nodes() == 3

    # test ndata and edata
    f1 = F.randn((3, 6))
    g.ndata['h'] = f1       # ok
    f2 = HG.nodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(F.tensor(g.nodes()), F.arange(0, 3))

    f3 = F.randn((2, 4))
    g.edata['h'] = f3
    f4 = HG.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(F.tensor(g.edges(form='eid')), F.arange(0, 2))

    # test fail case
    # fail due to multiple types
    fail = False
    try:
        HG.ndata['h']
    except dgl.DGLError:
        fail = True
    assert fail

    fail = False
    try:
        HG.edata['h']
    except dgl.DGLError:
        fail = True
    assert fail

def test_flatten():
    def check_mapping(g, fg):
        if len(fg.ntypes) == 1:
            SRC = DST = fg.ntypes[0]
        else:
            SRC = fg.ntypes[0]
            DST = fg.ntypes[1]

        etypes = F.asnumpy(fg.edata[dgl.ETYPE]).tolist()
        eids = F.asnumpy(fg.edata[dgl.EID]).tolist()

        for i, (etype, eid) in enumerate(zip(etypes, eids)):
            src_g, dst_g = g.find_edges([eid], g.canonical_etypes[etype])
            src_fg, dst_fg = fg.find_edges([i])
            # TODO(gq): I feel this code is quite redundant; can we just add new members (like
            # "induced_srcid") to returned heterograph object and not store them as features?
            assert src_g == fg.nodes[SRC].data[dgl.NID][src_fg]
            tid = F.asnumpy(fg.nodes[SRC].data[dgl.NTYPE][src_fg])[0]
            assert g.canonical_etypes[etype][0] == g.ntypes[tid]
            assert dst_g == fg.nodes[DST].data[dgl.NID][dst_fg]
            tid = F.asnumpy(fg.nodes[DST].data[dgl.NTYPE][dst_fg])[0]
            assert g.canonical_etypes[etype][2] == g.ntypes[tid]

    # check for wildcard slices
    g = create_test_heterograph()
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.nodes['game'].data['i'] = F.ones((2, 5))
    g.edges['plays'].data['e'] = F.ones((4, 4))
    g.edges['wishes'].data['e'] = F.ones((2, 4))
    g.edges['wishes'].data['f'] = F.ones((2, 4))

    fg = g['user', :, 'game']   # user--plays->game and user--wishes->game
    assert len(fg.ntypes) == 2
    assert fg.ntypes == ['user', 'game']
    assert fg.etypes == ['plays+wishes']

    assert F.array_equal(fg.nodes['user'].data['h'], F.ones((3, 5)))
    assert F.array_equal(fg.nodes['game'].data['i'], F.ones((2, 5)))
    assert F.array_equal(fg.edata['e'], F.ones((6, 4)))
    assert 'f' not in fg.edata

    etypes = F.asnumpy(fg.edata[dgl.ETYPE]).tolist()
    eids = F.asnumpy(fg.edata[dgl.EID]).tolist()
    assert set(zip(etypes, eids)) == set([(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)])

    check_mapping(g, fg)

    fg = g['user', :, 'user']
    # NOTE(gq): The node/edge types from the parent graph is returned if there is only one
    # node/edge type.  This differs from the behavior above.
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows']
    u1, v1 = g.edges(etype='follows', order='eid')
    u2, v2 = fg.edges(etype='follows', order='eid')
    assert F.array_equal(u1, u2)
    assert F.array_equal(v1, v2)

    fg = g['developer', :, 'game']
    assert fg.ntypes == ['developer', 'game']
    assert fg.etypes == ['develops']
    u1, v1 = g.edges(etype='develops', order='eid')
    u2, v2 = fg.edges(etype='develops', order='eid')
    assert F.array_equal(u1, u2)
    assert F.array_equal(v1, v2)

    fg = g[:, :, :]
    assert fg.ntypes == ['developer+user', 'game+user']
    assert fg.etypes == ['develops+follows+plays+wishes']
    check_mapping(g, fg)

    # Test another heterograph
    g_x = dgl.graph(([0, 1, 2], [1, 2, 3]), 'user', 'follows')
    g_y = dgl.graph(([0, 2], [2, 3]), 'user', 'knows')
    g_x.nodes['user'].data['h'] = F.randn((4, 3))
    g_x.edges['follows'].data['w'] = F.randn((3, 2))
    g_y.nodes['user'].data['hh'] = F.randn((4, 5))
    g_y.edges['knows'].data['ww'] = F.randn((2, 10))
    g = dgl.hetero_from_relations([g_x, g_y])

    assert F.array_equal(g.ndata['h'], g_x.ndata['h'])
    assert F.array_equal(g.ndata['hh'], g_y.ndata['hh'])
    assert F.array_equal(g.edges['follows'].data['w'], g_x.edata['w'])
    assert F.array_equal(g.edges['knows'].data['ww'], g_y.edata['ww'])

    fg = g['user', :, 'user']
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows+knows']
    check_mapping(g, fg)

    fg = g['user', :, :]
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows+knows']
    check_mapping(g, fg)

def test_convert():
    hg = create_test_heterograph()
    hs = []
    for ntype in hg.ntypes:
        h = F.randn((hg.number_of_nodes(ntype), 5))
        hg.nodes[ntype].data['h'] = h
        hs.append(h)
    hg.nodes['user'].data['x'] = F.randn((3, 3))
    ws = []
    for etype in hg.canonical_etypes:
        w = F.randn((hg.number_of_edges(etype), 5))
        hg.edges[etype].data['w'] = w
        ws.append(w)
    hg.edges['plays'].data['x'] = F.randn((4, 3))

    g = dgl.to_homo(hg)
    assert F.array_equal(F.cat(hs, dim=0), g.ndata['h'])
    assert 'x' not in g.ndata
    assert F.array_equal(F.cat(ws, dim=0), g.edata['w'])
    assert 'x' not in g.edata

    src, dst = g.all_edges(order='eid')
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    etype_id, eid = F.asnumpy(g.edata[dgl.ETYPE]), F.asnumpy(g.edata[dgl.EID])
    ntype_id, nid = F.asnumpy(g.ndata[dgl.NTYPE]), F.asnumpy(g.ndata[dgl.NID])
    for i in range(g.number_of_edges()):
        srctype = hg.ntypes[ntype_id[src[i]]]
        dsttype = hg.ntypes[ntype_id[dst[i]]]
        etype = hg.etypes[etype_id[i]]
        src_i, dst_i = hg.find_edges([eid[i]], (srctype, etype, dsttype))
        assert np.asscalar(F.asnumpy(src_i)) == nid[src[i]]
        assert np.asscalar(F.asnumpy(dst_i)) == nid[dst[i]]

    mg = nx.MultiDiGraph([
        ('user', 'user', 'follows'),
        ('user', 'game', 'plays'),
        ('user', 'game', 'wishes'),
        ('developer', 'game', 'develops')])

    for _mg in [None, mg]:
        hg2 = dgl.to_hetero(
                g, ['user', 'game', 'developer'], ['follows', 'plays', 'wishes', 'develops'],
                ntype_field=dgl.NTYPE, etype_field=dgl.ETYPE, metagraph=_mg)
        assert set(hg.ntypes) == set(hg2.ntypes)
        assert set(hg.canonical_etypes) == set(hg2.canonical_etypes)
        for ntype in hg.ntypes:
            assert hg.number_of_nodes(ntype) == hg2.number_of_nodes(ntype)
            assert F.array_equal(hg.nodes[ntype].data['h'], hg2.nodes[ntype].data['h'])
        for canonical_etype in hg.canonical_etypes:
            src, dst = hg.all_edges(etype=canonical_etype, order='eid')
            src2, dst2 = hg2.all_edges(etype=canonical_etype, order='eid')
            assert F.array_equal(src, src2)
            assert F.array_equal(dst, dst2)
            assert F.array_equal(hg.edges[canonical_etype].data['w'], hg2.edges[canonical_etype].data['w'])

    # hetero_from_homo test case 2
    g = dgl.graph([(0, 2), (1, 2), (2, 3), (0, 3)])
    g.ndata[dgl.NTYPE] = F.tensor([0, 0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0, 1, 2])
    hg = dgl.to_hetero(g, ['l0', 'l1', 'l2'], ['e0', 'e1', 'e2'])
    assert set(hg.canonical_etypes) == set(
        [('l0', 'e0', 'l1'), ('l1', 'e1', 'l2'), ('l0', 'e2', 'l2')])
    assert hg.number_of_nodes('l0') == 2
    assert hg.number_of_nodes('l1') == 1
    assert hg.number_of_nodes('l2') == 1
    assert hg.number_of_edges('e0') == 2
    assert hg.number_of_edges('e1') == 1
    assert hg.number_of_edges('e2') == 1

    # hetero_from_homo test case 3
    mg = nx.MultiDiGraph([
        ('user', 'movie', 'watches'),
        ('user', 'TV', 'watches')])
    g = dgl.graph([(0, 1), (0, 2)])
    g.ndata[dgl.NTYPE] = F.tensor([0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0])
    for _mg in [None, mg]:
        hg = dgl.to_hetero(g, ['user', 'TV', 'movie'], ['watches'], metagraph=_mg)
        assert set(hg.canonical_etypes) == set(
            [('user', 'watches', 'movie'), ('user', 'watches', 'TV')])
        assert hg.number_of_nodes('user') == 1
        assert hg.number_of_nodes('TV') == 1
        assert hg.number_of_nodes('movie') == 1
        assert hg.number_of_edges(('user', 'watches', 'TV')) == 1
        assert hg.number_of_edges(('user', 'watches', 'movie')) == 1
        assert len(hg.etypes) == 2

    # hetero_to_homo test case 2
    hg = dgl.bipartite([(0, 0), (1, 1)], card=(2, 3))
    g = dgl.to_homo(hg)
    assert g.number_of_nodes() == 5

def test_transform():
    g = create_test_heterograph()
    x = F.randn((3, 5))
    g.nodes['user'].data['h'] = x

    new_g = dgl.metapath_reachable_graph(g, ['follows', 'plays'])

    assert new_g.ntypes == ['user', 'game']
    assert new_g.number_of_edges() == 3
    assert F.asnumpy(new_g.has_edges_between([0, 0, 1], [0, 1, 1])).all()

    new_g = dgl.metapath_reachable_graph(g, ['follows'])

    assert new_g.ntypes == ['user']
    assert new_g.number_of_edges() == 2
    assert F.asnumpy(new_g.has_edges_between([0, 1], [1, 2])).all()

def test_subgraph():
    g = create_test_heterograph()
    x = F.randn((3, 5))
    y = F.randn((2, 4))
    g.nodes['user'].data['h'] = x
    g.edges['follows'].data['h'] = y

    def _check_subgraph(g, sg):
        assert sg.ntypes == ['user', 'game', 'developer']
        assert sg.etypes == ['follows', 'plays', 'wishes', 'develops']
        assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                             F.tensor([1, 2], F.int64))
        assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                             F.tensor([0], F.int64))
        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['wishes'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert sg.number_of_nodes('developer') == 0
        assert sg.number_of_edges('develops') == 0
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'][1:3])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'][1:2])

    sg1 = g.subgraph({'user': [1, 2], 'game': [0]})
    _check_subgraph(g, sg1)
    sg2 = g.edge_subgraph({'follows': [1], 'plays': [1], 'wishes': [1]})
    _check_subgraph(g, sg2)

    def _check_typed_subgraph(g, sg):
        assert set(sg.ntypes) == {'user', 'game'}
        assert set(sg.etypes) == {'follows', 'plays', 'wishes'}
        for ntype in sg.ntypes:
            assert sg.number_of_nodes(ntype) == g.number_of_nodes(ntype)
        for etype in sg.etypes:
            src_sg, dst_sg = sg.all_edges(etype=etype, order='eid')
            src_g, dst_g = g.all_edges(etype=etype, order='eid')
            assert F.array_equal(src_sg, src_g)
            assert F.array_equal(dst_sg, dst_g)
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'])
        g.nodes['user'].data['h'][2] = F.randn((5,))
        g.edges['follows'].data['h'][1] = F.randn((4,))
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'])

    sg3 = g.node_type_subgraph(['user', 'game'])
    _check_typed_subgraph(g, sg3)
    sg4 = g.edge_type_subgraph(['follows', 'plays', 'wishes'])
    _check_typed_subgraph(g, sg4)

def test_apply():
    def node_udf(nodes):
        return {'h': nodes.data['h'] * 2}
    def edge_udf(edges):
        return {'h': edges.data['h'] * 2 + edges.src['h']}

    g = create_test_heterograph()
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.apply_nodes(node_udf, ntype='user')
    assert F.array_equal(g.nodes['user'].data['h'], F.ones((3, 5)) * 2)

    g['plays'].edata['h'] = F.ones((4, 5))
    g.apply_edges(edge_udf, etype=('user', 'plays', 'game'))
    assert F.array_equal(g['plays'].edata['h'], F.ones((4, 5)) * 4)

    # test apply on graph with only one type
    g['follows'].apply_nodes(node_udf)
    assert F.array_equal(g.nodes['user'].data['h'], F.ones((3, 5)) * 4)

    g['plays'].apply_edges(edge_udf)
    assert F.array_equal(g['plays'].edata['h'], F.ones((4, 5)) * 12)

    # test fail case
    # fail due to multiple types
    fail = False
    try:
        g.apply_nodes(node_udf)
    except dgl.DGLError:
        fail = True
    assert fail

    fail = False
    try:
        g.apply_edges(edge_udf)
    except dgl.DGLError:
        fail = True
    assert fail

def test_level1():
    #edges = {
    #    'follows': ([0, 1], [1, 2]),
    #    'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
    #    'wishes': ([0, 2], [1, 0]),
    #    'develops': ([0, 1], [0, 1]),
    #}
    g = create_test_heterograph()
    def rfunc(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def rfunc2(nodes):
        return {'y': F.max(nodes.mailbox['m'], 1)}
    def mfunc(edges):
        return {'m': edges.src['h']}
    def afunc(nodes):
        return {'y' : nodes.data['y'] + 1}
    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.send([2, 3], mfunc, etype='plays')
    g.recv([0, 1], rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))
    g.nodes['game'].data.pop('y')

    # only one type
    play_g = g['plays']
    play_g.send([2, 3], mfunc)
    play_g.recv([0, 1], rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))
    # TODO(minjie): following codes will fail because messages are
    #   not shared with the base graph. However, since send and recv
    #   are rarely used, no fix at the moment.
    # g['plays'].send([2, 3], mfunc)
    # g['plays'].recv([0, 1], mfunc)

    # test fail case
    # fail due to multiple types
    fail = False
    try:
        g.send([2, 3], mfunc)
    except dgl.DGLError:
        fail = True
    assert fail

    fail = False
    try:
        g.recv([0, 1], rfunc)
    except dgl.DGLError:
        fail = True
    assert fail

    # test multi recv
    g.send(g.edges(etype='plays'), mfunc, etype='plays')
    g.send(g.edges(etype='wishes'), mfunc, etype='wishes')
    g.multi_recv([0, 1], {'plays' : rfunc, ('user', 'wishes', 'game'): rfunc2}, 'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))

    # test multi recv with apply function
    g.send(g.edges(etype='plays'), mfunc, etype='plays')
    g.send(g.edges(etype='wishes'), mfunc, etype='wishes')
    g.multi_recv([0, 1], {'plays' : (rfunc, afunc), ('user', 'wishes', 'game'): rfunc2}, 'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean']:
        g.send(g.edges(etype='plays'), mfunc, etype='plays')
        g.send(g.edges(etype='wishes'), mfunc, etype='wishes')
        g.multi_recv([0, 1], {'plays' : (rfunc, afunc), 'wishes': rfunc2}, cred, afunc)
        y = g.nodes['game'].data['y']
        g1 = g['plays']
        g2 = g['wishes']
        g1.send(g1.edges(), mfunc)
        g1.recv(g1.nodes('game'), rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g2.send(g2.edges(), mfunc)
        g2.recv(g2.nodes('game'), rfunc2)
        y2 = g.nodes['game'].data['y']
        yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
        yy = yy + 1  # final afunc
        assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    fail = False
    try:
        g.multi_recv([0, 1], {'plays' : rfunc, 'follows': rfunc2}, 'sum')
    except dgl.DGLError:
        fail = True
    assert fail

def test_level2():
    #edges = {
    #    'follows': ([0, 1], [1, 2]),
    #    'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
    #    'wishes': ([0, 2], [1, 0]),
    #    'develops': ([0, 1], [0, 1]),
    #}
    g = create_test_heterograph()
    def rfunc(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def rfunc2(nodes):
        return {'y': F.max(nodes.mailbox['m'], 1)}
    def mfunc(edges):
        return {'m': edges.src['h']}
    def afunc(nodes):
        return {'y' : nodes.data['y'] + 1}

    #############################################################
    #  send_and_recv
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.send_and_recv([2, 3], mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # only one type
    g['plays'].send_and_recv([2, 3], mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))
    
    # test fail case
    # fail due to multiple types
    fail = False
    try:
        g.send_and_recv([2, 3], mfunc, rfunc)
    except dgl.DGLError:
        fail = True
    assert fail

    # test multi
    g.multi_send_and_recv(
        {'plays' : (g.edges(etype='plays'), mfunc, rfunc),
         ('user', 'wishes', 'game'): (g.edges(etype='wishes'), mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))

    # test multi
    g.multi_send_and_recv(
        {'plays' : (g.edges(etype='plays'), mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (g.edges(etype='wishes'), mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean']:
        g.multi_send_and_recv(
            {'plays' : (g.edges(etype='plays'), mfunc, rfunc, afunc),
             'wishes': (g.edges(etype='wishes'), mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].send_and_recv(g.edges(etype='plays'), mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].send_and_recv(g.edges(etype='wishes'), mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
        yy = yy + 1  # final afunc
        assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    fail = False
    try:
        g.multi_send_and_recv(
            {'plays' : (g.edges(etype='plays'), mfunc, rfunc),
             'follows': (g.edges(etype='follows'), mfunc, rfunc2)},
            'sum')
    except dgl.DGLError:
        fail = True
    assert fail

    g.nodes['game'].data.clear()

    #############################################################
    #  pull
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.pull(1, mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # only one type
    g['plays'].pull(1, mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # test fail case
    fail = False
    try:
        g.pull(1, mfunc, rfunc)
    except dgl.DGLError:
        fail = True
    assert fail

    # test multi
    g.multi_pull(
        1,
        {'plays' : (mfunc, rfunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[0., 0.], [3., 3.]]))

    # test multi
    g.multi_pull(
        1,
        {'plays' : (mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[0., 0.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean']:
        g.multi_pull(
            1,
            {'plays' : (mfunc, rfunc, afunc),
             'wishes': (mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].pull(1, mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].pull(1, mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        g.nodes['game'].data['y'] = get_redfn(cred)(F.stack([y1, y2], 0), 0)
        g.apply_nodes(afunc, 1, ntype='game')
        yy = g.nodes['game'].data['y']
        assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    fail = False
    try:
        g.multi_pull(
            1,
            {'plays' : (mfunc, rfunc),
             'follows': (mfunc, rfunc2)},
            'sum')
    except dgl.DGLError:
        fail = True
    assert fail

    g.nodes['game'].data.clear()

    #############################################################
    #  update_all
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.update_all(mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))

    # only one type
    g['plays'].update_all(mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))

    # test fail case
    # fail due to multiple types
    fail = False
    try:
        g.update_all(mfunc, rfunc)
    except dgl.DGLError:
        fail = True
    assert fail

    # test multi
    g.multi_update_all(
        {'plays' : (mfunc, rfunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))

    # test multi
    g.multi_update_all(
        {'plays' : (mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean', 'stack']:
        g.multi_update_all(
            {'plays' : (mfunc, rfunc, afunc),
             'wishes': (mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].update_all(mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].update_all(mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        if cred == 'stack':
            # stack has two both correct outcomes
            yy1 = F.stack([F.unsqueeze(y1, 1), F.unsqueeze(y2, 1)], 1)
            yy1 = yy1 + 1  # final afunc
            yy2 = F.stack([F.unsqueeze(y2, 1), F.unsqueeze(y1, 1)], 1)
            yy2 = yy2 + 1  # final afunc
            assert F.array_equal(y, yy1) or F.array_equal(y, yy2)
        else:
            yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
            yy = yy + 1  # final afunc
            assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    fail = False
    try:
        g.update_all(
            {'plays' : (mfunc, rfunc),
             'follows': (mfunc, rfunc2)},
            'sum')
    except dgl.DGLError:
        fail = True
    assert fail

    g.nodes['game'].data.clear()

def test_updates():
    def msg_func(edges):
        return {'m': edges.src['h']}
    def reduce_func(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def apply_func(nodes):
        return {'y': nodes.data['y'] * 2}
    g = create_test_heterograph()
    x = F.randn((3, 5))
    g.nodes['user'].data['h'] = x

    for msg, red, apply in itertools.product(
            [fn.copy_u('h', 'm'), msg_func], [fn.sum('m', 'y'), reduce_func],
            [None, apply_func]):
        multiplier = 1 if apply is None else 2

        g['user', 'plays', 'game'].update_all(msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g.nodes['game'].data['y']

        g['user', 'plays', 'game'].send_and_recv(([0, 1, 2], [0, 1, 1]), msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g.nodes['game'].data['y']

        plays_g = g['user', 'plays', 'game']
        plays_g.send(([0, 1, 2], [0, 1, 1]), msg)
        plays_g.recv([0, 1], red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g.nodes['game'].data['y']

        # pulls from destination (game) node 0
        g['user', 'plays', 'game'].pull(0, msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        del g.nodes['game'].data['y']

        # pushes from source (user) node 0
        g['user', 'plays', 'game'].push(0, msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        del g.nodes['game'].data['y']

def test_backward():
    g = create_test_heterograph()
    x = F.randn((3, 5))
    F.attach_grad(x)
    g.nodes['user'].data['h'] = x
    with F.record_grad():
        g.multi_update_all(
            {'plays' : (fn.copy_u('h', 'm'), fn.sum('m', 'y')),
             'wishes': (fn.copy_u('h', 'm'), fn.sum('m', 'y'))},
            'sum')
        y = g.nodes['game'].data['y']
        F.backward(y, F.ones(y.shape))
    print(F.grad(x))
    assert F.array_equal(F.grad(x), F.tensor([[2., 2., 2., 2., 2.],
                                              [2., 2., 2., 2., 2.],
                                              [2., 2., 2., 2., 2.]]))

if __name__ == '__main__':
    test_create()
    test_query()
    test_adj()
    test_inc()
    test_view()
    test_view1()
    test_flatten()
    test_convert()
    test_transform()
    test_subgraph()
    test_apply()
    test_level1()
    test_level2()
    test_updates()
    test_backward()
