from dgl.graph import DGLGraph

def message_func(src, dst, edge):
    return src['h']

def update_func(node, accum):
    return {'h' : node['h'] + accum}

def generate_graph():
    g = DGLGraph()
    for i in range(10):
        g.add_node(i, h=i+1) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    return g

def check(g, h):
    nh = [str(g.nodes[i]['h']) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def test_sendrecv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func('sum')
    g.sendto(0, 1)
    g.recv(1)
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    g.sendto(5, 9)
    g.sendto(6, 9)
    g.recv(9)
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 23])

def test_multi_sendrecv():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func('sum')
    # one-many
    g.sendto(0, [1, 2, 3])
    g.recv([1, 2, 3])
    check(g, [1, 3, 4, 5, 5, 6, 7, 8, 9, 10])
    # many-one
    g.sendto([6, 7, 8], 9)
    g.recv(9)
    check(g, [1, 3, 4, 5, 5, 6, 7, 8, 9, 34])
    # many-many
    g.sendto([0, 0, 4, 5], [4, 5, 9, 9])
    g.recv([4, 5, 9])
    check(g, [1, 3, 4, 5, 6, 7, 7, 8, 9, 45])

def test_update_routines():
    g = generate_graph()
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    g.register_message_func(message_func)
    g.register_update_func(update_func)
    g.register_reduce_func('sum')
    g.update_by_edge(0, 1)
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 10])
    g.update_to(9)
    check(g, [1, 3, 3, 4, 5, 6, 7, 8, 9, 55])
    g.update_from(0)
    check(g, [1, 4, 4, 5, 6, 7, 8, 9, 10, 55])
    g.update_all()
    check(g, [56, 5, 5, 6, 7, 8, 9, 10, 11, 108])

if __name__ == '__main__':
    test_sendrecv()
    test_multi_sendrecv()
    test_update_routines()
