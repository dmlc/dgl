import torch as th
from dgl.graph import DGLGraph
from dgl.bgraph import DGLBGraph
import dgl.backend as F

D = 1024
w = th.randn(D, D)

def message_func(src, dst, edge):
    return {'h' : src['h']}

def reduce_func(msgs):
    return {'h' : sum(msg['h'] for msg in msgs)}

def update_func(node, msgs):
    return {'h' : node['h'] + th.mm(msgs['h'], w)}

def generate_graph():
    g = DGLGraph()
    bg = DGLBGraph()
    for i in range(10):
        h = th.rand(1, D)
        g.add_node(i, h=h) # 10 nodes.
        bg.add_node(i, h=h)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        h = th.rand(1, D)
        g.add_edge(0, i, h=h)
        bg.add_edge(0, i, h=h)

        h = th.rand(1, D)
        g.add_edge(i, 9, h=h)
        bg.add_edge(i, 9, h=h)
    # add a back flow from 9 to 0
    h = th.rand(1, D)
    g.add_edge(9, 0, h=h)
    bg.add_edge(9, 0, h=h)

    return g, bg

def check(g, bg, tolerance=1e-5):
    delta = max(th.max(th.abs(g.nodes[n]['h'] - bg.nodes[n]['h'])) for n in g.nodes)
    assert delta < tolerance, str(delta)

    delta = max(th.max(th.abs(g.edges[u, v]['h'] - bg.edges[u, v]['h'])) \
                for u, v in g.edges)
    assert delta < tolerance, str(delta)

def test_sendrecv():
    g, bg = generate_graph()
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_update_func(update_func)
    bg.register_message_func(message_func)
    bg.register_reduce_func(reduce_func)
    bg.register_update_func(update_func)

    g.sendto(0, 1)
    g.recvfrom(1, [0])

    g.sendto(0, 1)
    g.recvfrom(1, [0])
    bg.sendto(0, 1)
    bg.recvfrom(1, [0])
    check(g, bg)

    g.sendto(5, 9)
    g.sendto(6, 9)
    g.recvfrom(9, [5, 6])
    bg.sendto(5, 9)
    bg.sendto(6, 9)
    bg.recvfrom(9, [5, 6])
    check(g, bg)

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

    g.register_message_func(message_func)
    g.register_update_func(update_func)
    bg.register_message_func(message_func)
    bg.register_update_func(update_func)

    g.update_by_edge(0, 1)
    bg.update_by_edge(0, 1)
    check(g, bg)

    g.update_to(9)
    bg.update_to(9)
    check(g, bg)

    g.update_from(0)
    bg.update_from(0)
    check(g, bg)

    g.update_all()
    bg.update_all()
    check(g, bg)

if __name__ == '__main__':
    test_sendrecv()
    test_multi_sendrecv()
    test_update_routines()
