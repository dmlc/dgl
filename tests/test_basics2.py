from dgl import DGLGraph
from dgl.graph import __REPR__

def message_func(hu, e_uv):
    return hu

def message_not_called(hu, e_uv):
    assert False
    return hu

def reduce_not_called(h, msgs):
    assert False
    return 0

def reduce_func(h, msgs):
    return h + sum(msgs)

def check(g, h):
    nh = [str(g.nodes[i][__REPR__]) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i, __REPR__=i+1) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    return g

def test_no_msg_recv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_not_called)
    g.register_reduce_func(reduce_not_called)
    g.register_apply_node_func(lambda h : h + 1)
    for i in range(10):
        g.recv(i)
    check(g, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

def test_double_recv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.send(1, 9)
    g.send(2, 9)
    g.recv(9)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 15])
    g.register_reduce_func(reduce_not_called)
    g.recv(9)

def test_pull_0deg():
    g = DGLGraph()
    g.add_node(0, h=2)
    g.add_node(1, h=1)
    g.add_edge(0, 1)
    def _message(src, edge):
        assert False
        return src
    def _reduce(node, msgs):
        assert False
        return node
    def _update(node):
        return {'h': node['h'] * 2}
    g.pull(0, _message, _reduce, _update)
    assert g.nodes[0]['h'] == 4

if __name__ == '__main__':
    test_no_msg_recv()
    test_double_recv()
    test_pull_0deg()
