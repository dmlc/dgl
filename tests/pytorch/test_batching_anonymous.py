import torch as th
from dgl.graph import DGLGraph, __REPR__

D = 32
reduce_msg_shapes = set()

def message_func(hu, e_uv):
    assert len(hu.shape) == 2
    assert hu.shape[1] == D
    return hu

def reduce_func(hv, msgs):
    reduce_msg_shapes.add(tuple(msgs.shape))
    assert len(msgs.shape) == 3
    assert msgs.shape[2] == D
    return th.sum(msgs, 1)

def update_func(hv, accum):
    assert hv.shape == accum.shape
    return hv + accum

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
    g.set_n_repr(col)
    return g

def test_batch_setter_getter():
    def _pfc(x):
        return list(x.numpy()[:,0])
    g = generate_graph()
    # set all nodes
    g.set_n_repr(th.zeros((10, D)))
    assert _pfc(g.get_n_repr()) == [0.] * 10
    # set partial nodes
    u = th.tensor([1, 3, 5])
    g.set_n_repr(th.ones((3, D)), u)
    assert _pfc(g.get_n_repr()) == [0., 1., 0., 1., 0., 1., 0., 0., 0., 0.]
    # get partial nodes
    u = th.tensor([1, 2, 3])
    assert _pfc(g.get_n_repr(u)) == [1., 0., 1.]

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
    g.set_e_repr(th.zeros((17, D)))
    assert _pfc(g.get_e_repr()) == [0.] * 17
    # set partial edges (many-many)
    u = th.tensor([0, 0, 2, 5, 9])
    v = th.tensor([1, 3, 9, 9, 0])
    g.set_e_repr(th.ones((5, D)), u, v)
    truth = [0.] * 17
    truth[0] = truth[4] = truth[3] = truth[9] = truth[16] = 1.
    assert _pfc(g.get_e_repr()) == truth
    # set partial edges (many-one)
    u = th.tensor([3, 4, 6])
    v = th.tensor([9])
    g.set_e_repr(th.ones((3, D)), u, v)
    truth[5] = truth[7] = truth[11] = 1.
    assert _pfc(g.get_e_repr()) == truth
    # set partial edges (one-many)
    u = th.tensor([0])
    v = th.tensor([4, 5, 6])
    g.set_e_repr(th.ones((3, D)), u, v)
    truth[6] = truth[8] = truth[10] = 1.
    assert _pfc(g.get_e_repr()) == truth
    # get partial edges (many-many)
    u = th.tensor([0, 6, 0])
    v = th.tensor([6, 9, 7])
    assert _pfc(g.get_e_repr(u, v)) == [1., 1., 0.]
    # get partial edges (many-one)
    u = th.tensor([5, 6, 7])
    v = th.tensor([9])
    assert _pfc(g.get_e_repr(u, v)) == [1., 1., 0.]
    # get partial edges (one-many)
    u = th.tensor([0])
    v = th.tensor([3, 4, 5])
    assert _pfc(g.get_e_repr(u, v)) == [1., 1., 1.]

def test_batch_send():
    g = generate_graph()
    def _fmsg(hu, edge):
        assert hu.shape == (5, D)
        return hu
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
    test_batch_send()
    test_batch_recv()
    test_update_routines()
