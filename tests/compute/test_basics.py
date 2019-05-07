# Currently readonly graph construction only accepts sparse tensor in MXNet,
# and pytorch doesn't support readonly graph or graph creation from sparse
# tensor.  For now, readonly graph test is postponed until we have better
# readonly graph support.
import backend as F
import dgl
import networkx as nx
from dgl import DGLGraph
from collections import defaultdict as ddict

D = 5
reduce_msg_shapes = set()

def message_func(edges):
    assert F.ndim(edges.src['h']) == 2
    assert F.shape(edges.src['h'])[1] == D
    return {'m' : edges.src['h']}

def reduce_func(nodes):
    msgs = nodes.mailbox['m']
    reduce_msg_shapes.add(tuple(msgs.shape))
    assert F.ndim(msgs) == 3
    assert F.shape(msgs)[2] == D
    return {'accum' : F.sum(msgs, 1)}

def apply_node_func(nodes):
    return {'h' : nodes.data['h'] + nodes.data['accum']}

def generate_graph(grad=False):
    g = DGLGraph()
    g.add_nodes(10) # 10 nodes
    # create a graph where 0 is the source and 9 is the sink
    # 17 edges
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    ncol = F.randn((10, D))
    ecol = F.randn((17, D))
    if grad:
        ncol = F.attach_grad(ncol)
        ecol = F.attach_grad(ecol)

    g.ndata['h'] = ncol
    g.edata['w'] = ecol
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g

def test_batch_setter_getter():
    def _pfc(x):
        return list(F.zerocopy_to_numpy(x)[:,0])
    g = generate_graph()
    # set all nodes
    g.ndata['h'] = F.zeros((10, D))
    assert F.allclose(g.ndata['h'], F.zeros((10, D)))
    # pop nodes
    old_len = len(g.ndata)
    assert _pfc(g.pop_n_repr('h')) == [0.] * 10
    assert len(g.ndata) == old_len - 1
    g.ndata['h'] = F.zeros((10, D))
    # set partial nodes
    u = F.tensor([1, 3, 5])
    g.nodes[u].data['h'] = F.ones((3, D))
    assert _pfc(g.ndata['h']) == [0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]
    # get partial nodes
    u = F.tensor([1, 2, 3])
    assert _pfc(g.nodes[u].data['h']) == [1., 0., 1.]

    '''
    s, d, eid
    0, 1, 0
    1, 9, 1
    0, 2, 2
    2, 9, 3
    0, 3, 4
    3, 9, 5
    0, 4, 6
    4, 9, 7
    0, 5, 8
    5, 9, 9
    0, 6, 10
    6, 9, 11
    0, 7, 12
    7, 9, 13
    0, 8, 14
    8, 9, 15
    9, 0, 16
    '''
    # set all edges
    g.edata['l'] = F.zeros((17, D))
    assert _pfc(g.edata['l']) == [0.] * 17
    # pop edges
    old_len = len(g.edata)
    assert _pfc(g.pop_e_repr('l')) == [0.] * 17
    assert len(g.edata) == old_len - 1
    g.edata['l'] = F.zeros((17, D))
    # set partial edges (many-many)
    u = F.tensor([0, 0, 2, 5, 9])
    v = F.tensor([1, 3, 9, 9, 0])
    g.edges[u, v].data['l'] = F.ones((5, D))
    truth = [0.] * 17
    truth[0] = truth[4] = truth[3] = truth[9] = truth[16] = 1.
    assert _pfc(g.edata['l']) == truth
    # set partial edges (many-one)
    u = F.tensor([3, 4, 6])
    v = F.tensor([9])
    g.edges[u, v].data['l'] = F.ones((3, D))
    truth[5] = truth[7] = truth[11] = 1.
    assert _pfc(g.edata['l']) == truth
    # set partial edges (one-many)
    u = F.tensor([0])
    v = F.tensor([4, 5, 6])
    g.edges[u, v].data['l'] = F.ones((3, D))
    truth[6] = truth[8] = truth[10] = 1.
    assert _pfc(g.edata['l']) == truth
    # get partial edges (many-many)
    u = F.tensor([0, 6, 0])
    v = F.tensor([6, 9, 7])
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 0.]
    # get partial edges (many-one)
    u = F.tensor([5, 6, 7])
    v = F.tensor([9])
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 0.]
    # get partial edges (one-many)
    u = F.tensor([0])
    v = F.tensor([3, 4, 5])
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 1.]

def test_batch_setter_autograd():
    g = generate_graph(grad=True)
    h1 = g.ndata['h']
    # partial set
    v = F.tensor([1, 2, 8])
    hh = F.attach_grad(F.zeros((len(v), D)))
    with F.record_grad():
        g.nodes[v].data['h'] = hh
        h2 = g.ndata['h']
    F.backward(h2, F.ones((10, D)) * 2)
    assert F.array_equal(F.grad(h1)[:,0], F.tensor([2., 0., 0., 2., 2., 2., 2., 2., 0., 2.]))
    assert F.array_equal(F.grad(hh)[:,0], F.tensor([2., 2., 2.]))

def test_nx_conversion():
    # check conversion between networkx and DGLGraph

    def _check_nx_feature(nxg, nf, ef):
        # check node and edge feature of nxg
        # this is used to check to_networkx
        num_nodes = len(nxg)
        num_edges = nxg.size()
        if num_nodes > 0:
            node_feat = ddict(list)
            for nid, attr in nxg.nodes(data=True):
                assert len(attr) == len(nf)
                for k in nxg.nodes[nid]:
                    node_feat[k].append(F.unsqueeze(attr[k], 0))
            for k in node_feat:
                feat = F.cat(node_feat[k], 0)
                assert F.allclose(feat, nf[k])
        else:
            assert len(nf) == 0
        if num_edges > 0:
            edge_feat = ddict(lambda: [0] * num_edges)
            for u, v, attr in nxg.edges(data=True):
                assert len(attr) == len(ef) + 1 # extra id
                eid = attr['id']
                for k in ef:
                    edge_feat[k][eid] = F.unsqueeze(attr[k], 0)
            for k in edge_feat:
                feat = F.cat(edge_feat[k], 0)
                assert F.allclose(feat, ef[k])
        else:
            assert len(ef) == 0

    n1 = F.randn((5, 3))
    n2 = F.randn((5, 10))
    n3 = F.randn((5, 4))
    e1 = F.randn((4, 5))
    e2 = F.randn((4, 7))
    g = DGLGraph(multigraph=True)
    g.add_nodes(5)
    g.add_edges([0,1,3,4], [2,4,0,3])
    g.ndata.update({'n1': n1, 'n2': n2, 'n3': n3})
    g.edata.update({'e1': e1, 'e2': e2})

    # convert to networkx
    nxg = g.to_networkx(node_attrs=['n1', 'n3'], edge_attrs=['e1', 'e2'])
    assert len(nxg) == 5
    assert nxg.size() == 4
    _check_nx_feature(nxg, {'n1': n1, 'n3': n3}, {'e1': e1, 'e2': e2})

    # convert to DGLGraph, nx graph has id in edge feature
    # use id feature to test non-tensor copy
    g.from_networkx(nxg, node_attrs=['n1'], edge_attrs=['e1', 'id'])
    # check graph size
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 4
    # check number of features
    # test with existing dglgraph (so existing features should be cleared)
    assert len(g.ndata) == 1
    assert len(g.edata) == 2
    # check feature values
    assert F.allclose(g.ndata['n1'], n1)
    # with id in nx edge feature, e1 should follow original order
    assert F.allclose(g.edata['e1'], e1)
    assert F.array_equal(g.get_e_repr()['id'], F.arange(0, 4))

    # test conversion after modifying DGLGraph
    g.pop_e_repr('id') # pop id so we don't need to provide id when adding edges
    new_n = F.randn((2, 3))
    new_e = F.randn((3, 5))
    g.add_nodes(2, data={'n1': new_n})
    # add three edges, one is a multi-edge
    g.add_edges([3, 6, 0], [4, 5, 2], data={'e1': new_e})
    n1 = F.cat((n1, new_n), 0)
    e1 = F.cat((e1, new_e), 0)
    # convert to networkx again
    nxg = g.to_networkx(node_attrs=['n1'], edge_attrs=['e1'])
    assert len(nxg) == 7
    assert nxg.size() == 7
    _check_nx_feature(nxg, {'n1': n1}, {'e1': e1})

    # now test convert from networkx without id in edge feature
    # first pop id in edge feature
    for _, _, attr in nxg.edges(data=True):
        attr.pop('id')
    # test with a new graph
    g = DGLGraph(multigraph=True)
    g.from_networkx(nxg, node_attrs=['n1'], edge_attrs=['e1'])
    # check graph size
    assert g.number_of_nodes() == 7
    assert g.number_of_edges() == 7
    # check number of features
    assert len(g.ndata) == 1
    assert len(g.edata) == 1
    # check feature values
    assert F.allclose(g.ndata['n1'], n1)
    # edge feature order follows nxg.edges()
    edge_feat = []
    for _, _, attr in nxg.edges(data=True):
        edge_feat.append(F.unsqueeze(attr['e1'], 0))
    edge_feat = F.cat(edge_feat, 0)
    assert F.allclose(g.edata['e1'], edge_feat)

    # Test converting from a networkx graph whose nodes are
    # not labeled with consecutive-integers.
    nxg = nx.cycle_graph(5)
    nxg.remove_nodes_from([0, 4])
    for u in nxg.nodes():
        nxg.node[u]['h'] = F.tensor([u])
    for u, v, d in nxg.edges(data=True):
        d['h'] = F.tensor([u, v])

    g = dgl.DGLGraph()
    g.from_networkx(nxg, node_attrs=['h'], edge_attrs=['h'])
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 4
    assert g.has_edge_between(0, 1)
    assert g.has_edge_between(1, 2)
    assert F.allclose(g.ndata['h'], F.tensor([[1.], [2.], [3.]]))
    assert F.allclose(g.edata['h'], F.tensor([[1., 2.], [1., 2.],
                                              [2., 3.], [2., 3.]]))

def test_batch_send():
    g = generate_graph()
    def _fmsg(edges):
        assert tuple(F.shape(edges.src['h'])) == (5, D)
        return {'m' : edges.src['h']}
    g.register_message_func(_fmsg)
    # many-many send
    u = F.tensor([0, 0, 0, 0, 0])
    v = F.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # one-many send
    u = F.tensor([0])
    v = F.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # many-one send
    u = F.tensor([1, 2, 3, 4, 5])
    v = F.tensor([9])
    g.send((u, v))

def test_batch_recv():
    # basic recv test
    g = generate_graph()
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)
    u = F.tensor([0, 0, 0, 4, 5, 6])
    v = F.tensor([1, 2, 3, 9, 9, 9])
    reduce_msg_shapes.clear()
    g.send((u, v))
    g.recv(F.unique(v))
    assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    reduce_msg_shapes.clear()

def test_apply_nodes():
    def _upd(nodes):
        return {'h' : nodes.data['h'] * 2}
    g = generate_graph()
    g.register_apply_node_func(_upd)
    old = g.ndata['h']
    g.apply_nodes()
    assert F.allclose(old * 2, g.ndata['h'])
    u = F.tensor([0, 3, 4, 6])
    g.apply_nodes(lambda nodes : {'h' : nodes.data['h'] * 0.}, u)
    assert F.allclose(F.gather_row(g.ndata['h'], u), F.zeros((4, D)))

def test_apply_edges():
    def _upd(edges):
        return {'w' : edges.data['w'] * 2}
    g = generate_graph()
    g.register_apply_edge_func(_upd)
    old = g.edata['w']
    g.apply_edges()
    assert F.allclose(old * 2, g.edata['w'])
    u = F.tensor([0, 0, 0, 4, 5, 6])
    v = F.tensor([1, 2, 3, 9, 9, 9])
    g.apply_edges(lambda edges : {'w' : edges.data['w'] * 0.}, (u, v))
    eid = F.tensor(g.edge_ids(u, v))
    assert F.allclose(F.gather_row(g.edata['w'], eid), F.zeros((6, D)))

def test_update_routines():
    g = generate_graph()
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)

    # send_and_recv
    reduce_msg_shapes.clear()
    u = [0, 0, 0, 4, 5, 6]
    v = [1, 2, 3, 9, 9, 9]
    g.send_and_recv((u, v))
    assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    reduce_msg_shapes.clear()
    try:
        g.send_and_recv([u, v])
        assert False
    except:
        pass

    # pull
    v = F.tensor([1, 2, 3, 9])
    reduce_msg_shapes.clear()
    g.pull(v)
    assert(reduce_msg_shapes == {(1, 8, D), (3, 1, D)})
    reduce_msg_shapes.clear()

    # push
    v = F.tensor([0, 1, 2, 3])
    reduce_msg_shapes.clear()
    g.push(v)
    assert(reduce_msg_shapes == {(1, 3, D), (8, 1, D)})
    reduce_msg_shapes.clear()

    # update_all
    reduce_msg_shapes.clear()
    g.update_all()
    assert(reduce_msg_shapes == {(1, 8, D), (9, 1, D)})
    reduce_msg_shapes.clear()

def test_recv_0deg():
    # test recv with 0deg nodes;
    g = DGLGraph()
    g.add_nodes(2)
    g.add_edge(0, 1)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.data['h'] + F.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + F.zeros(shape, dtype, ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(_init2, 'h')
    # test#1: recv both 0deg and non-0deg nodes
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv([0, 1])
    new = g.ndata.pop('h')
    # 0deg check: initialized with the func and got applied
    assert F.allclose(new[0], F.full_1d(5, 4, F.float32))
    # non-0deg check
    assert F.allclose(new[1], F.sum(old, 0) * 2)

    # test#2: recv only 0deg node is equal to apply
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv(0)
    new = g.ndata.pop('h')
    # 0deg check: equal to apply_nodes
    assert F.allclose(new[0], 2 * old[0])
    # non-0deg check: untouched
    assert F.allclose(new[1], old[1])

def test_recv_0deg_newfld():
    # test recv with 0deg nodes; the reducer also creates a new field
    g = DGLGraph()
    g.add_nodes(2)
    g.add_edge(0, 1)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h1' : nodes.data['h'] + F.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h1' : nodes.data['h1'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + F.zeros(shape, dtype=dtype, ctx=ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    # test#1: recv both 0deg and non-0deg nodes
    old = F.randn((2, 5))
    g.set_n_initializer(_init2, 'h1')
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv([0, 1])
    new = g.ndata.pop('h1')
    # 0deg check: initialized with the func and got applied
    assert F.allclose(new[0], F.full_1d(5, 4, dtype=F.float32))
    # non-0deg check
    assert F.allclose(new[1], F.sum(old, 0) * 2)

    # test#2: recv only 0deg node
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.ndata['h1'] = F.full((2, 5), -1, F.int64)  # this is necessary
    g.send((0, 1))
    g.recv(0)
    new = g.ndata.pop('h1')
    # 0deg check: fallback to apply
    assert F.allclose(new[0], F.full_1d(5, -2, F.int64))
    # non-0deg check: not changed
    assert F.allclose(new[1], F.full_1d(5, -1, F.int64))

def test_update_all_0deg():
    # test#1
    g = DGLGraph()
    g.add_nodes(5)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    g.add_edge(3, 0)
    g.add_edge(4, 0)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.data['h'] + F.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + F.zeros(shape, dtype, ctx)
    g.set_n_initializer(_init2, 'h')
    old_repr = F.randn((5, 5))
    g.ndata['h'] = old_repr
    g.update_all(_message, _reduce, _apply)
    new_repr = g.ndata['h']
    # the first row of the new_repr should be the sum of all the node
    # features; while the 0-deg nodes should be initialized by the
    # initializer and applied with UDF.
    assert F.allclose(new_repr[1:], 2*(2+F.zeros((4,5))))
    assert F.allclose(new_repr[0], 2 * F.sum(old_repr, 0))

    # test#2: graph with no edge
    g = DGLGraph()
    g.add_nodes(5)
    g.set_n_initializer(_init2, 'h')
    g.ndata['h'] = old_repr
    g.update_all(_message, _reduce, _apply)
    new_repr = g.ndata['h']
    # should fallback to apply
    assert F.allclose(new_repr, 2*old_repr)

def test_pull_0deg():
    g = DGLGraph()
    g.add_nodes(2)
    g.add_edge(0, 1)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.data['h'] + F.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + F.zeros(shape, dtype, ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(_init2, 'h')
    # test#1: pull both 0deg and non-0deg nodes
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.pull([0, 1])
    new = g.ndata.pop('h')
    # 0deg check: initialized with the func and got applied
    assert F.allclose(new[0], F.full_1d(5, 4, dtype=F.float32))
    # non-0deg check
    assert F.allclose(new[1], F.sum(old, 0) * 2)

    # test#2: pull only 0deg node
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.pull(0)
    new = g.ndata.pop('h')
    # 0deg check: fallback to apply
    assert F.allclose(new[0], 2*old[0])
    # non-0deg check: not touched
    assert F.allclose(new[1], old[1])

def test_send_multigraph():
    g = DGLGraph(multigraph=True)
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    g.add_edge(0, 1)
    g.add_edge(2, 1)

    def _message_a(edges):
        return {'a': edges.data['a']}
    def _message_b(edges):
        return {'a': edges.data['a'] * 3}
    def _reduce(nodes):
        return {'a': F.max(nodes.mailbox['a'], 1)}

    def answer(*args):
        return F.max(F.stack(args, 0), 0)

    # send by eid
    old_repr = F.randn((4, 5))
    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send([0, 2], message_func=_message_a)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], answer(old_repr[0], old_repr[2]))

    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send([0, 2, 3], message_func=_message_a)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], answer(old_repr[0], old_repr[2], old_repr[3]))

    # send on multigraph
    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send(([0, 2], [1, 1]), _message_a)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], F.max(old_repr, 0))

    # consecutive send and send_on
    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send((2, 1), _message_a)
    g.send([0, 1], message_func=_message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], answer(old_repr[0] * 3, old_repr[1] * 3, old_repr[3]))

    # consecutive send_on
    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send(0, message_func=_message_a)
    g.send(1, message_func=_message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], answer(old_repr[0], old_repr[1] * 3))

    # send_and_recv_on
    g.ndata['a'] = F.zeros((3, 5))
    g.edata['a'] = old_repr
    g.send_and_recv([0, 2, 3], message_func=_message_a, reduce_func=_reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], answer(old_repr[0], old_repr[2], old_repr[3]))
    assert F.allclose(new_repr[[0, 2]], F.zeros((2, 5)))

def test_dynamic_addition():
    N = 3
    D = 1

    g = DGLGraph()

    # Test node addition
    g.add_nodes(N)
    g.ndata.update({'h1': F.randn((N, D)),
                    'h2': F.randn((N, D))})
    g.add_nodes(3)
    assert g.ndata['h1'].shape[0] == g.ndata['h2'].shape[0] == N + 3

    # Test edge addition
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.edata.update({'h1': F.randn((2, D)),
                    'h2': F.randn((2, D))})
    assert g.edata['h1'].shape[0] == g.edata['h2'].shape[0] == 2

    g.add_edges([0, 2], [2, 0])
    g.edata['h1'] = F.randn((4, D))
    assert g.edata['h1'].shape[0] == g.edata['h2'].shape[0] == 4

    g.add_edge(1, 2)
    g.edges[4].data['h1'] = F.randn((1, D))
    assert g.edata['h1'].shape[0] == g.edata['h2'].shape[0] == 5

    # test add edge with part of the features
    g.add_edge(2, 1, {'h1': F.randn((1, D))})
    assert len(g.edata['h1']) == len(g.edata['h2'])


def test_repr():
    G = dgl.DGLGraph()
    G.add_nodes(10)
    G.add_edge(0, 1)
    repr_string = G.__repr__()
    print(repr_string)
    G.ndata['x'] = F.zeros((10, 5))
    G.add_edges([0, 1], 2)
    G.edata['y'] = F.zeros((3, 4))
    repr_string = G.__repr__()
    print(repr_string)


def test_group_apply_edges():
    def edge_udf(edges):
        h = F.sum(edges.data['feat'] * (edges.src['h'] + edges.dst['h']), dim=2)
        normalized_feat = F.softmax(h, dim=1)
        return {"norm_feat": normalized_feat}

    g = DGLGraph()
    g.add_nodes(10)
    g.add_edges(0, [1, 2, 3, 4, 5, 6, 7, 8])
    g.add_edges(1, [2, 3, 4, 6, 7, 8])
    g.add_edges(2, [2, 3, 4, 5, 6, 7, 8])

    g.ndata['h'] = F.randn((g.number_of_nodes(), D))
    g.edata['feat'] = F.randn((g.number_of_edges(), D))

    def _test(group_by):
        g.group_apply_edges(group_by=group_by, func=edge_udf)
        if group_by == 'src':
            u, v, eid = g.out_edges(1, form='all')
        else:
            u, v, eid = g.in_edges(5, form='all')
        out_feat = g.edata['norm_feat'][eid]
        result = (g.ndata['h'][u] + g.ndata['h'][v]) * g.edata['feat'][eid]
        result = F.softmax(F.sum(result, dim=1), dim=0)
        assert F.allclose(out_feat, result)

    # test group by source nodes
    _test('src')

    # test group by destination nodes
    _test('dst')


if __name__ == '__main__':
    test_nx_conversion()
    test_batch_setter_getter()
    test_batch_setter_autograd()
    test_batch_send()
    test_batch_recv()
    test_apply_nodes()
    test_apply_edges()
    test_update_routines()
    test_recv_0deg()
    test_recv_0deg_newfld()
    test_update_all_0deg()
    test_pull_0deg()
    test_send_multigraph()
    test_dynamic_addition()
    test_repr()
    test_group_apply_edges()
