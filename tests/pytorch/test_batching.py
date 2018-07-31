import torch as th
from dgl.graph import DGLGraph

D = 5
reduce_msg_shapes = set()

def message_func(src, edge):
    assert len(src['h'].shape) == 2
    assert src['h'].shape[1] == D
    return {'m' : src['h']}

def reduce_func(node, msgs):
    msgs = msgs['m']
    reduce_msg_shapes.add(tuple(msgs.shape))
    assert len(msgs.shape) == 3
    assert msgs.shape[2] == D
    return th.sum(msgs, 1)

def update_func(node, accum):
    assert node['h'].shape == accum.shape
    return {'h' : node['h'] + accum}

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    col = th.randn(10, D)
    g.set_n_repr({'h' : col})
    return g

def test_batch_setter_getter():
    def _pfc(x):
        return list(x.numpy()[:,0])
    g = generate_graph()
    # set all nodes
    g.set_n_repr({'h' : th.zeros((10, D))})
    assert _pfc(g.get_n_repr()['h']) == [0.] * 10
    # set partial nodes
    u = th.tensor([1, 3, 5])
    g.set_n_repr({'h' : th.ones((3, D))}, u)
    assert _pfc(g.get_n_repr()['h']) == [0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]
    # get partial nodes
    u = th.tensor([1, 2, 3])
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
    g.set_e_repr({'l' : th.zeros((17, D))})
    assert _pfc(g.get_e_repr()['l']) == [0.] * 17
    # set partial nodes (many-many)
    # TODO(minjie): following case will fail at the moment as CachedGraph
    # does not maintain edge addition order.
    u = th.tensor([0, 0, 2, 5, 9])
    v = th.tensor([1, 3, 9, 9, 0])
    g.set_e_repr({'l' : th.ones((5, D))}, u, v)
    truth = [0.] * 17
    truth[0] = truth[4] = truth[3] = truth[9] = truth[16] = 1.
    assert _pfc(g.get_e_repr()['l']) == truth

def test_batch_send():
    g = generate_graph()
    def _fmsg(src, edge):
        assert src['h'].shape == (5, D)
        return {'m' : src['h']}
    g.register_message_func(_fmsg, batchable=True)
    # many-many sendto
    u = th.tensor([0, 0, 0, 0, 0])
    v = th.tensor([1, 2, 3, 4, 5])
    g.sendto(u, v)
    # one-many sendto
    u = th.tensor([0])
    v = th.tensor([1, 2, 3, 4, 5])
    g.sendto(u, v)
    # many-one sendto
    u = th.tensor([1, 2, 3, 4, 5])
    v = th.tensor([9])
    g.sendto(u, v)

def test_batch_recv():
    g = generate_graph()
    g.register_message_func(message_func, batchable=True)
    g.register_reduce_func(reduce_func, batchable=True)
    g.register_update_func(update_func, batchable=True)
    u = th.tensor([0, 0, 0, 4, 5, 6])
    v = th.tensor([1, 2, 3, 9, 9, 9])
    reduce_msg_shapes.clear()
    g.sendto(u, v)
    g.recv(th.unique(v))
    assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    reduce_msg_shapes.clear()

def test_update_routines():
    g = generate_graph()
    g.register_message_func(message_func, batchable=True)
    g.register_reduce_func(reduce_func, batchable=True)
    g.register_update_func(update_func, batchable=True)

    # update_by_edge
    reduce_msg_shapes.clear()
    u = th.tensor([0, 0, 0, 4, 5, 6])
    v = th.tensor([1, 2, 3, 9, 9, 9])
    g.update_by_edge(u, v)
    assert(reduce_msg_shapes == {(1, 3, D), (3, 1, D)})
    reduce_msg_shapes.clear()

    # update_to
    v = th.tensor([1, 2, 3, 9])
    reduce_msg_shapes.clear()
    g.update_to(v)
    assert(reduce_msg_shapes == {(1, 8, D), (3, 1, D)})
    reduce_msg_shapes.clear()

    # update_from
    v = th.tensor([0, 1, 2, 3])
    reduce_msg_shapes.clear()
    g.update_from(v)
    assert(reduce_msg_shapes == {(1, 3, D), (8, 1, D)})
    reduce_msg_shapes.clear()

    # update_all
    reduce_msg_shapes.clear()
    g.update_all()
    assert(reduce_msg_shapes == {(1, 8, D), (9, 1, D)})
    reduce_msg_shapes.clear()

if __name__ == '__main__':
    test_batch_setter_getter()
    test_batch_send()
    test_batch_recv()
    test_update_routines()
