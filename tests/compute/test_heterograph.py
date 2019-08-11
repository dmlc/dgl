
import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F

def create_test_heterograph():
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    follows = dgl.bipartite_from_edge_list(
        'user', 'user', 'follows', [0, 1], [1, 2], 3, 3)

    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    plays = dgl.bipartite_from_scipy('user', 'game', 'plays', plays_spmat)

    wishes_spmat = ssp.coo_matrix(([1, 0], ([2, 0], [0, 1])))
    wishes = dgl.bipartite_from_scipy('user', 'game', 'wishes', wishes_spmat, True)

    develops = dgl.bipartite_from_edge_list(
        'developer', 'game', 'develops', [0, 1], [0, 1], 2, 2)

    g = dgl.DGLHeteroGraph([follows, plays, wishes, develops])
    return g

def test_query():
    g = create_test_heterograph()

    ntypes = ['user', 'game', 'developer']
    etypes = ['follows', 'plays', 'wishes', 'develops']
    edges = {
        'follows': ([0, 1], [1, 2]),
        'plays': ([0, 1, 1, 2], [0, 0, 1, 1]),
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

    # node & edge types
    assert set(ntypes) == set(g.all_node_types)
    assert set(etypes) == set(g.all_edge_types)

    # metagraph
    mg = g.metagraph
    assert set(g.all_node_types) == set(mg.nodes)
    etype_triplets = [(u, v, e['type']) for u, v, e in mg.edges(data=True)]
    assert set([
        ('user', 'user', 'follows'),
        ('user', 'game', 'plays'),
        ('user', 'game', 'wishes'),
        ('developer', 'game', 'develops')]) == set(etype_triplets)

    # endpoint types
    for u, v, e in etype_triplets:
        assert (u, v) == g.endpoint_types(e)

    # number of nodes
    assert [g[ntype].number_of_nodes() for ntype in ntypes] == [3, 2, 2]
    assert [g[etype].number_of_nodes() for etype in etypes] == [3, 5, 5, 4]

    # number of edges
    assert [g[etype].number_of_edges() for etype in etypes] == [2, 4, 2, 2]

    # has_node & has_nodes
    for ntype in ntypes:
        n = g[ntype].number_of_nodes()
        for i in range(n):
            assert g[ntype].has_node(i)
        assert not g[ntype].has_node(n)
        assert np.array_equal(
            F.asnumpy(g[ntype].has_nodes([0, n])).astype('int32'), [1, 0])

    for etype in etypes:
        srcs, dsts = edges[etype]
        for src, dst in zip(srcs, dsts):
            assert g[etype].has_edge_between(src, dst)
        assert F.asnumpy(g[etype].has_edges_between(srcs, dsts)).all()

        srcs, dsts = negative_edges[etype]
        for src, dst in zip(srcs, dsts):
            assert not g[etype].has_edge_between(src, dst)
        assert not F.asnumpy(g[etype].has_edges_between(srcs, dsts)).any()

        srcs, dsts = edges[etype]
        n_edges = len(srcs)

        # predecessors & in_edges & in_degree
        pred = [s for s, d in zip(srcs, dsts) if d == 0]
        assert set(F.asnumpy(g[etype].predecessors(0)).tolist()) == set(pred)
        u, v = g[etype].in_edges([0])
        assert F.asnumpy(v).tolist() == [0] * len(pred)
        assert set(F.asnumpy(u).tolist()) == set(pred)
        assert g[etype].in_degree(0) == len(pred)

        # successors & out_edges & out_degree
        succ = [d for s, d in zip(srcs, dsts) if s == 0]
        assert set(F.asnumpy(g[etype].successors(0)).tolist()) == set(succ)
        u, v = g[etype].out_edges([0])
        assert F.asnumpy(u).tolist() == [0] * len(succ)
        assert set(F.asnumpy(v).tolist()) == set(succ)
        assert g[etype].out_degree(0) == len(succ)

        # edge_id & edge_ids
        for i, (src, dst) in enumerate(zip(srcs, dsts)):
            assert g[etype].edge_id(src, dst) == i
            assert F.asnumpy(g[etype].edge_id(src, dst, force_multi=True)).tolist() == [i]
        assert F.asnumpy(g[etype].edge_ids(srcs, dsts)).tolist() == list(range(n_edges))
        u, v, e = g[etype].edge_ids(srcs, dsts, force_multi=True)
        assert F.asnumpy(u).tolist() == srcs
        assert F.asnumpy(v).tolist() == dsts
        assert F.asnumpy(e).tolist() == list(range(n_edges))

        # find_edges
        u, v = g[etype].find_edges(list(range(n_edges)))
        assert F.asnumpy(u).tolist() == srcs
        assert F.asnumpy(v).tolist() == dsts

        # all_edges.  edges are already in srcdst order
        for order in ['srcdst', 'eid']:
            u, v, e = g[etype].all_edges('all', order)
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts
            assert F.asnumpy(e).tolist() == list(range(n_edges))

        # in_degrees & out_degrees
        in_degrees = F.asnumpy(g[etype].in_degrees())
        out_degrees = F.asnumpy(g[etype].out_degrees())
        src_count = Counter(srcs)
        dst_count = Counter(dsts)
        utype, vtype = g.endpoint_types(etype)
        for i in range(g[utype].number_of_nodes()):
            assert out_degrees[i] == src_count[i]
        for i in range(g[vtype].number_of_nodes()):
            assert in_degrees[i] == dst_count[i]

def test_frame():
    g = create_test_heterograph()

    f1 = F.randn((3, 6))
    f2 = F.randn((2, 4))
    g['user'].ndata['h'] = f1       # ok
    g['user'].edata['x'] = f2       # ok - only one edge type in users subgraph
    f3 = g['follows'].edata['x']    # ok
    f4 = g['follows'].ndata['h']    # ok - only one node type in follows subgraph
    assert F.array_equal(f2, f3)    # they are shared
    assert F.array_equal(f1, f4)
    F.scatter_row_inplace(f3, 0, 0)
    f2 = g['user'].edata['x']
    assert F.array_equal(f2, f3)

def test_apply():
    def node_udf(nodes):
        return {'h': nodes.data['h'] * 2}
    def edge_udf(edges):
        return {'h': edges.data['h'] * 2 + edges.src['h']}

    g = create_test_heterograph()
    g['user'].ndata['h'] = F.ones((3, 5))
    g['user'].apply_nodes(node_udf)
    assert F.array_equal(g['user'].ndata['h'], F.ones((3, 5)) * 2)

    g['plays'].edata['h'] = F.ones((4, 5))
    g['plays'].apply_edges(edge_udf)
    assert F.array_equal(g['plays'].edata['h'], F.ones((4, 5)) * 4)

def test_updates():
    def msg_func(edges):
        return {'m': edges.src['h']}
    def reduce_func(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def apply_func(nodes):
        return {'y': nodes.data['y'] * 2}
    g = create_test_heterograph()
    x = F.randn((3, 5))
    g['user'].ndata['h'] = x

    for msg, red, apply in itertools.product(
            [fn.copy_u('h', 'm'), msg_func], [fn.sum('m', 'y'), reduce_func],
            [None, apply_func]):
        multiplier = 1 if apply is None else 2

        g['plays'].update_all(msg, red, apply)
        y = g['game'].ndata['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g['game'].ndata['y']

        g['plays'].send_and_recv(([0, 1, 2], [0, 1, 1]), msg, red, apply)
        y = g['game'].ndata['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g['game'].ndata['y']

        g['plays'].send(([0, 1, 2], [0, 1, 1]), msg)
        g['plays'].recv([0, 1], red, apply)
        y = g['game'].ndata['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g['game'].ndata['y']

        # pulls from destination (game) node 0
        g['plays'].pull(0, msg, red, apply)
        y = g['game'].ndata['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        del g['game'].ndata['y']

        # pushes from source (user) node 0
        g['plays'].push(0, msg, red, apply)
        y = g['game'].ndata['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        del g['game'].ndata['y']

if __name__ == '__main__':
    test_query()
    test_frame()
    test_apply()
    test_updates()
