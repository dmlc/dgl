from dgl import DGLGraph
from dgl.graph import __REPR__

def message_func(hu, e_uv):
    return hu + e_uv

def update_func(h, accum):
    return h + accum

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i, __REPR__=i+1) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i, __REPR__=1)
        g.add_edge(i, 9, __REPR__=1)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    return g

def check(g, h):
    nh = [str(g.nodes[i][__REPR__]) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def test_sendrecv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func('sum')
    g.send(0, 1)
    g.recv(1)
    check(g, [1, 4, 3, 4, 5, 6, 7, 8, 9, 10])
    g.send(5, 9)
    g.send(6, 9)
    g.recv(9)
    check(g, [1, 4, 3, 4, 5, 6, 7, 8, 9, 25])

def message_func_hybrid(src, edge):
    return src[__REPR__] + edge

def update_func_hybrid(node, accum):
    return node[__REPR__] + accum

def test_hybridrepr():
    g = generate_graph()
    for i in range(10):
        g.nodes[i]['id'] = -i
    g.register_message_func(message_func_hybrid)
    g.register_update_func(update_func_hybrid)
    g.register_reduce_func('sum')
    g.send(0, 1)
    g.recv(1)
    check(g, [1, 4, 3, 4, 5, 6, 7, 8, 9, 10])
    g.send(5, 9)
    g.send(6, 9)
    g.recv(9)
    check(g, [1, 4, 3, 4, 5, 6, 7, 8, 9, 25])

if __name__ == '__main__':
    test_sendrecv()
    test_hybridrepr()
