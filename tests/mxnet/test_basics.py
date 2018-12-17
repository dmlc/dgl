import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
from dgl.graph import DGLGraph
import dgl
import scipy.sparse as spsp

D = 5
reduce_msg_shapes = set()

def check_eq(a, b):
    assert a.shape == b.shape
    assert mx.nd.sum(a == b).asnumpy() == int(np.prod(list(a.shape)))

def message_func(edges):
    assert len(edges.src['h'].shape) == 2
    assert edges.src['h'].shape[1] == D
    return {'m' : edges.src['h']}

def reduce_func(nodes):
    msgs = nodes.mailbox['m']
    reduce_msg_shapes.add(tuple(msgs.shape))
    assert len(msgs.shape) == 3
    assert msgs.shape[2] == D
    return {'m' : mx.nd.sum(msgs, 1)}

def apply_node_func(nodes):
    return {'h' : nodes.data['h'] + nodes.data['m']}

def generate_graph(grad=False, readonly=False):
    if readonly:
        row_idx = []
        col_idx = []
        for i in range(1, 9):
            row_idx.append(0)
            col_idx.append(i)
            row_idx.append(i)
            col_idx.append(9)
        row_idx.append(9)
        col_idx.append(0)
        ones = np.ones(shape=(len(row_idx)))
        csr = spsp.csr_matrix((ones, (row_idx, col_idx)), shape=(10, 10))
        g = DGLGraph(csr, readonly=True)
        ncol = mx.nd.random.normal(shape=(10, D))
        ecol = mx.nd.random.normal(shape=(17, D))
        if grad:
            ncol.attach_grad()
            ecol.attach_grad()
        g.ndata['h'] = ncol
        g.edata['w'] = ecol
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        return g
    else:
        g = DGLGraph()
        g.add_nodes(10) # 10 nodes.
        # create a graph where 0 is the source and 9 is the sink
        for i in range(1, 9):
            g.add_edge(0, i)
            g.add_edge(i, 9)
        # add a back flow from 9 to 0
        g.add_edge(9, 0)
        ncol = mx.nd.random.normal(shape=(10, D))
        ecol = mx.nd.random.normal(shape=(17, D))
        if grad:
            ncol.attach_grad()
            ecol.attach_grad()
        g.ndata['h'] = ncol
        g.edata['w'] = ecol
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        return g

def test_batch_setter_getter():
    def _pfc(x):
        return list(x.asnumpy()[:,0])
    g = generate_graph()
    # set all nodes
    g.set_n_repr({'h' : mx.nd.zeros((10, D))})
    assert _pfc(g.ndata['h']) == [0.] * 10
    # pop nodes
    assert _pfc(g.pop_n_repr('h')) == [0.] * 10
    assert len(g.ndata) == 0
    g.set_n_repr({'h' : mx.nd.zeros((10, D))})
    # set partial nodes
    u = mx.nd.array([1, 3, 5], dtype='int64')
    g.set_n_repr({'h' : mx.nd.ones((3, D))}, u)
    assert _pfc(g.ndata['h']) == [0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]
    # get partial nodes
    u = mx.nd.array([1, 2, 3], dtype='int64')
    assert _pfc(g.get_n_repr(u)['h']) == [1., 0., 1.]

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
    g.edata['l'] = mx.nd.zeros((17, D))
    assert _pfc(g.edata['l']) == [0.] * 17
    # pop edges
    old_len = len(g.edata)
    assert _pfc(g.pop_e_repr('l')) == [0.] * 17
    assert len(g.edata) == old_len - 1
    g.edata['l'] = mx.nd.zeros((17, D))
    # set partial edges (many-many)
    u = mx.nd.array([0, 0, 2, 5, 9], dtype='int64')
    v = mx.nd.array([1, 3, 9, 9, 0], dtype='int64')
    g.edges[u, v].data['l'] = mx.nd.ones((5, D))
    truth = [0.] * 17
    truth[0] = truth[4] = truth[3] = truth[9] = truth[16] = 1.
    assert _pfc(g.edata['l']) == truth
    # set partial edges (many-one)
    u = mx.nd.array([3, 4, 6], dtype='int64')
    v = mx.nd.array([9], dtype='int64')
    g.edges[u, v].data['l'] = mx.nd.ones((3, D))
    truth[5] = truth[7] = truth[11] = 1.
    assert _pfc(g.edata['l']) == truth
    # set partial edges (one-many)
    u = mx.nd.array([0], dtype='int64')
    v = mx.nd.array([4, 5, 6], dtype='int64')
    g.edges[u, v].data['l'] = mx.nd.ones((3, D))
    truth[6] = truth[8] = truth[10] = 1.
    assert _pfc(g.edata['l']) == truth
    # get partial edges (many-many)
    u = mx.nd.array([0, 6, 0], dtype='int64')
    v = mx.nd.array([6, 9, 7], dtype='int64')
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 0.]
    # get partial edges (many-one)
    u = mx.nd.array([5, 6, 7], dtype='int64')
    v = mx.nd.array([9], dtype='int64')
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 0.]
    # get partial edges (one-many)
    u = mx.nd.array([0], dtype='int64')
    v = mx.nd.array([3, 4, 5], dtype='int64')
    assert _pfc(g.edges[u, v].data['l']) == [1., 1., 1.]

def test_batch_setter_autograd():
    with mx.autograd.record():
        g = generate_graph(grad=True, readonly=True)
        h1 = g.ndata['h']
        h1.attach_grad()
        # partial set
        v = mx.nd.array([1, 2, 8], dtype='int64')
        hh = mx.nd.zeros((len(v), D))
        hh.attach_grad()
        g.set_n_repr({'h' : hh}, v)
        h2 = g.ndata['h']
    h2.backward(mx.nd.ones((10, D)) * 2)
    check_eq(h1.grad[:,0], mx.nd.array([2., 0., 0., 2., 2., 2., 2., 2., 0., 2.]))
    check_eq(hh.grad[:,0], mx.nd.array([2., 2., 2.]))

def test_batch_send():
    g = generate_graph()
    def _fmsg(edges):
        assert edges.src['h'].shape == (5, D)
        return {'m' : edges.src['h']}
    g.register_message_func(_fmsg)
    # many-many send
    u = mx.nd.array([0, 0, 0, 0, 0], dtype='int64')
    v = mx.nd.array([1, 2, 3, 4, 5], dtype='int64')
    g.send((u, v))
    # one-many send
    u = mx.nd.array([0], dtype='int64')
    v = mx.nd.array([1, 2, 3, 4, 5], dtype='int64')
    g.send((u, v))
    # many-one send
    u = mx.nd.array([1, 2, 3, 4, 5], dtype='int64')
    v = mx.nd.array([9], dtype='int64')
    g.send((u, v))

def check_batch_recv(readonly):
    # basic recv test
    g = generate_graph(readonly=readonly)
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)
    u = mx.nd.array([0, 0, 0, 4, 5, 6], dtype='int64')
    v = mx.nd.array([1, 2, 3, 9, 9, 9], dtype='int64')
    reduce_msg_shapes.clear()
    g.send((u, v))
    #g.recv(th.unique(v))
    #assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    #reduce_msg_shapes.clear()

def test_batch_recv():
    check_batch_recv(True)
    check_batch_recv(False)

def test_apply_nodes():
    def _upd(nodes):
        return {'h' : nodes.data['h'] * 2}
    g = generate_graph()
    g.register_apply_node_func(_upd)
    old = g.ndata['h']
    g.apply_nodes()
    assert np.allclose((old * 2).asnumpy(), g.ndata['h'].asnumpy())
    u = mx.nd.array([0, 3, 4, 6], dtype=np.int64)
    g.apply_nodes(lambda nodes : {'h' : nodes.data['h'] * 0.}, u)
    h = g.ndata['h'][u].asnumpy()
    assert np.allclose(h, np.zeros(shape=(4, D), dtype=h.dtype))

def test_apply_edges():
    def _upd(edges):
        return {'w' : edges.data['w'] * 2}
    g = generate_graph()
    g.register_apply_edge_func(_upd)
    old = g.edata['w']
    g.apply_edges()
    assert np.allclose((old * 2).asnumpy(), g.edata['w'].asnumpy())
    u = mx.nd.array([0, 0, 0, 4, 5, 6], dtype=np.int64)
    v = mx.nd.array([1, 2, 3, 9, 9, 9], dtype=np.int64)
    g.apply_edges(lambda edges : {'w' : edges.data['w'] * 0.}, (u, v))
    eid = g.edge_ids(u, v)
    w = g.edata['w'][eid].asnumpy()
    assert np.allclose(w, np.zeros(shape=(6, D), dtype=w.dtype))

def check_update_routines(readonly):
    g = generate_graph(readonly=readonly)
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)

    # send_and_recv
    reduce_msg_shapes.clear()
    u = mx.nd.array([0, 0, 0, 4, 5, 6], dtype='int64')
    v = mx.nd.array([1, 2, 3, 9, 9, 9], dtype='int64')
    g.send_and_recv((u, v))
    assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    reduce_msg_shapes.clear()

    # pull
    v = mx.nd.array([1, 2, 3, 9], dtype='int64')
    reduce_msg_shapes.clear()
    g.pull(v)
    assert(reduce_msg_shapes == {(1, 8, D), (3, 1, D)})
    reduce_msg_shapes.clear()

    # push
    v = mx.nd.array([0, 1, 2, 3], dtype='int64')
    reduce_msg_shapes.clear()
    g.push(v)
    assert(reduce_msg_shapes == {(1, 3, D), (8, 1, D)})
    reduce_msg_shapes.clear()

    # update_all
    reduce_msg_shapes.clear()
    g.update_all()
    assert(reduce_msg_shapes == {(1, 8, D), (9, 1, D)})
    reduce_msg_shapes.clear()

def test_update_routines():
    check_update_routines(True)
    check_update_routines(False)

def check_reduce_0deg(readonly):
    if readonly:
        row_idx = []
        col_idx = []
        for i in range(1, 5):
            row_idx.append(i)
            col_idx.append(0)
        ones = np.ones(shape=(len(row_idx)))
        csr = spsp.csr_matrix((ones, (row_idx, col_idx)), shape=(5, 5))
        g = DGLGraph(csr, readonly=True)
    else:
        g = DGLGraph()
        g.add_nodes(5)
        g.add_edge(1, 0)
        g.add_edge(2, 0)
        g.add_edge(3, 0)
        g.add_edge(4, 0)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.data['h'] + nodes.mailbox['m'].sum(1)}
    def _init2(shape, dtype, ctx, ids):
        return 2 + mx.nd.zeros(shape, dtype=dtype, ctx=ctx)
    g.set_n_initializer(_init2, 'h')
    old_repr = mx.nd.random.normal(shape=(5, 5))
    g.set_n_repr({'h': old_repr})
    g.update_all(_message, _reduce)
    new_repr = g.ndata['h']

    assert np.allclose(new_repr[1:].asnumpy(), 2+np.zeros((4, 5)))
    assert np.allclose(new_repr[0].asnumpy(), old_repr.sum(0).asnumpy())

def test_reduce_0deg():
    check_reduce_0deg(True)
    check_reduce_0deg(False)

def test_recv_0deg_newfld():
    # test recv with 0deg nodes; the reducer also creates a new field
    g = DGLGraph()
    g.add_nodes(2)
    g.add_edge(0, 1)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h1' : nodes.data['h'] + mx.nd.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h1' : nodes.data['h1'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + mx.nd.zeros(shape=shape, dtype=dtype, ctx=ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    # test#1: recv both 0deg and non-0deg nodes
    old = mx.nd.random.normal(shape=(2, 5))
    g.set_n_initializer(_init2, 'h1')
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv([0, 1])
    new = g.ndata.pop('h1')
    # 0deg check: initialized with the func and got applied
    assert np.allclose(new[0].asnumpy(), np.full((5,), 4))
    # non-0deg check
    assert np.allclose(new[1].asnumpy(), mx.nd.sum(old, 0).asnumpy() * 2)

    # test#2: recv only 0deg node
    old = mx.nd.random.normal(shape=(2, 5))
    g.ndata['h'] = old
    g.ndata['h1'] = mx.nd.full((2, 5), -1)  # this is necessary
    g.send((0, 1))
    g.recv(0)
    new = g.ndata.pop('h1')
    # 0deg check: fallback to apply
    assert np.allclose(new[0].asnumpy(), np.full((5,), -2))
    # non-0deg check: not changed
    assert np.allclose(new[1].asnumpy(), np.full((5,), -1))

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
        return {'h' : nodes.data['h'] + mx.nd.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + mx.nd.zeros(shape, dtype=dtype, ctx=ctx)
    g.set_n_initializer(_init2, 'h')
    old_repr = mx.nd.random.normal(shape=(5, 5))
    g.ndata['h'] = old_repr
    g.update_all(_message, _reduce, _apply)
    new_repr = g.ndata['h']
    # the first row of the new_repr should be the sum of all the node
    # features; while the 0-deg nodes should be initialized by the
    # initializer and applied with UDF.
    assert np.allclose(new_repr[1:].asnumpy(), 2*(2+np.zeros((4,5))))
    assert np.allclose(new_repr[0].asnumpy(), 2 * mx.nd.sum(old_repr, 0).asnumpy())

    # test#2: graph with no edge
    g = DGLGraph()
    g.add_nodes(5)
    g.set_n_initializer(_init2, 'h')
    g.ndata['h'] = old_repr
    g.update_all(_message, _reduce, _apply)
    new_repr = g.ndata['h']
    # should fallback to apply
    assert np.allclose(new_repr.asnumpy(), 2*old_repr.asnumpy())


def check_pull_0deg(readonly):
    if readonly:
        row_idx = []
        col_idx = []
        row_idx.append(0)
        col_idx.append(1)
        ones = np.ones(shape=(len(row_idx)))
        csr = spsp.csr_matrix((ones, (row_idx, col_idx)), shape=(2, 2))
        g = DGLGraph(csr, readonly=True)
    else:
        g = DGLGraph()
        g.add_nodes(2)
        g.add_edge(0, 1)
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.mailbox['m'].sum(1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + mx.nd.zeros(shape, dtype=dtype, ctx=ctx)
    g.set_n_initializer(_init2, 'h')
    old_repr = mx.nd.random.normal(shape=(2, 5))

    # test#1: pull only 0-deg node
    g.ndata['h'] = old_repr
    g.pull(0, _message, _reduce, _apply)
    new_repr = g.ndata['h']
    # 0deg check: equal to apply_nodes
    assert np.allclose(new_repr[0].asnumpy(), old_repr[0].asnumpy() * 2)
    # non-0deg check: untouched
    assert np.allclose(new_repr[1].asnumpy(), old_repr[1].asnumpy())

    # test#2: pull only non-deg node
    g.ndata['h'] = old_repr
    g.pull(1, _message, _reduce, _apply)
    new_repr = g.ndata['h']
    # 0deg check: untouched
    assert np.allclose(new_repr[0].asnumpy(), old_repr[0].asnumpy())
    # non-0deg check: recved node0 and got applied
    assert np.allclose(new_repr[1].asnumpy(), old_repr[0].asnumpy() * 2)

    # test#3: pull only both nodes
    g.ndata['h'] = old_repr
    g.pull([0, 1], _message, _reduce, _apply)
    new_repr = g.ndata['h']
    # 0deg check: init and applied
    t = mx.nd.zeros(shape=(2,5)) + 4
    assert np.allclose(new_repr[0].asnumpy(), t.asnumpy())
    # non-0deg check: recv node0 and applied
    assert np.allclose(new_repr[1].asnumpy(), old_repr[0].asnumpy() * 2)

def test_pull_0deg():
    check_pull_0deg(True)
    check_pull_0deg(False)

def test_repr():
    G = dgl.DGLGraph()
    G.add_nodes(10)
    G.add_edge(0, 1)
    repr_string = G.__repr__()
    G.ndata['x'] = mx.nd.zeros((10, 5))
    G.add_edges([0, 1], 2)
    G.edata['y'] = mx.nd.zeros((3, 4))
    repr_string = G.__repr__()

if __name__ == '__main__':
    test_batch_setter_getter()
    test_batch_setter_autograd()
    test_batch_send()
    test_batch_recv()
    test_apply_nodes()
    test_apply_edges()
    test_update_routines()
    test_reduce_0deg()
    test_recv_0deg_newfld()
    test_update_all_0deg()
    test_pull_0deg()
    test_repr()
