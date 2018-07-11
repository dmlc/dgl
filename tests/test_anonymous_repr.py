from dgl import DGLGraph

def message_func(hu, hv, e_uv):
    return hu

def update_func(h, accum):
    return h + accum

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
        g.set_n_repr(i, i+1)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    return g

def check(g, h):
    nh = [str(g.get_n_repr(i)) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def test_sendrecv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func('sum')
    g.sendto(0, 1)
    g.recvfrom(1, [0])
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    g.sendto(5, 9)
    g.sendto(6, 9)
    g.recvfrom(9, [5, 6])
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 23])

if __name__ == '__main__':
    test_sendrecv()
