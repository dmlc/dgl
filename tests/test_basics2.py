from dgl import DGLGraph
from dgl.graph import __REPR__

def message_func(hu, e_uv):
    return hu

def message_not_called(hu, e_uv):
    assert False
    return hu

def reduce_not_called(msgs):
    assert False
    return 0

def update_no_msg(h, accum):
    assert accum is None
    return h + 1

def update_func(h, accum):
    assert accum is not None
    return h + accum

def check(g, h):
    nh = [str(g.get_n_repr(i)) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
        g.set_n_repr(i, i+1)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.set_e_repr(0, i, 1)
        g.add_edge(i, 9)
        g.set_e_repr(i, 9, 1)
    return g

def test_no_msg_update():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_not_called)
    g.register_reduce_func(reduce_not_called)
    g.register_update_func(update_no_msg)
    for i in range(10):
        g.recv(i)
    check(g, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

def test_double_recv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_reduce_func('sum')
    g.register_update_func(update_func)
    g.sendto(1, 9)
    g.sendto(2, 9)
    g.recv(9)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    try:
        # The second recv should have a None message
        g.recv(9)
    except:
        return
    assert False

def test_recv_no_pred():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_not_called)
    g.register_reduce_func(reduce_not_called)
    g.register_update_func(update_no_msg)
    g.recv(0)

def test_skipped_reduce():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_not_called)
    g.register_update_func(update_func)
    g.sendto(0, 1)
    g.recv(1)
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 10])

if __name__ == '__main__':
    test_no_msg_update()
    test_double_recv()
    test_recv_no_pred()
    test_skipped_reduce()
