import torch as th
from dgl.graph import DGLGraph

D = 32
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
    # TODO: use internal interface to set data.
    col = th.randn(10, D)
    g._node_frame['h'] = col
    return g

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

'''
def test_multi_sendrecv():
    g, bg = generate_graph()

    check(g, bg)

    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_update_func(update_func)

    bg.register_message_func(message_func)
    bg.register_reduce_func(reduce_func)
    bg.register_update_func(update_func)

    # one-many
    g.sendto(0, [1, 2, 3])
    g.recvfrom([1, 2, 3], [[0], [0], [0]])

    bg.sendto(0, [1, 2, 3])
    bg.recvfrom([1, 2, 3], [[0], [0], [0]])

    check(g, bg)

    # many-one
    g.sendto([6, 7, 8], 9)
    g.recvfrom(9, [6, 7, 8])

    bg.sendto([6, 7, 8], 9)
    bg.recvfrom(9, [6, 7, 8])

    check(g, bg)

    # many-many
    g.sendto([0, 0, 4, 5], [4, 5, 9, 9])
    g.recvfrom([4, 5, 9], [[0], [0], [4, 5]])

    bg.sendto([0, 0, 4, 5], [4, 5, 9, 9])
    bg.recvfrom([4, 5, 9], [[0], [0], [4, 5]])

    check(g, bg)

def test_update_routines():
    g, bg = generate_graph()

    check(g, bg)

    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func(reduce_func)

    bg.register_message_func(message_func)
    bg.register_update_func(update_func)
    bg.register_reduce_func(reduce_func)

    g.update_by_edge(0, 1)
    g.update_to(9)

    bg.update_by_edge(0, 1)
    bg.update_to(9)

    check(g, bg)

    g.update_from(0)
    g.update_all()

    bg.update_from(0)
    bg.update_all()

    check(g, bg)
'''

if __name__ == '__main__':
    test_batch_send()
    test_batch_recv()
    #test_multi_sendrecv()
    #test_update_routines()
