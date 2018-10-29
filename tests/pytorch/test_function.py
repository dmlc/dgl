import torch as th
import dgl
import dgl.function as fn

def generate_graph():
    g = dgl.DGLGraph()
    g.add_nodes(10) # 10 nodes.
    h = th.arange(1, 11, dtype=th.float)
    g.set_n_repr({'h': h})
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    h = th.tensor([1., 2., 1., 3., 1., 4., 1., 5., 1., 6.,\
            1., 7., 1., 8., 1., 9., 10.])
    g.set_e_repr({'h' : h})
    return g

def generate_graph1():
    """graph with anonymous repr"""
    g = dgl.DGLGraph()
    g.add_nodes(10) # 10 nodes.
    h = th.arange(1, 11, dtype=th.float)
    h = th.arange(1, 11, dtype=th.float)
    g.set_n_repr(h)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    h = th.tensor([1., 2., 1., 3., 1., 4., 1., 5., 1., 6.,\
            1., 7., 1., 8., 1., 9., 10.])
    g.set_e_repr(h)
    return g

def reducer_both(node, msgs):
    return {'h' : th.sum(msgs['m'], 1)}

def test_copy_src():
    # copy_src with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_copy_edge():
    # copy_edge with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_src_mul_edge():
    # src_mul_edge with all fields
    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

def test_dynamic_addition():
    N = 3
    D = 1

    g = dgl.DGLGraph()

    # Test node addition
    g.add_nodes(N)
    g.set_n_repr({'h1': th.randn(N, D),
                  'h2': th.randn(N, D)})
    g.add_nodes(3)
    n_repr = g.get_n_repr()
    assert n_repr['h1'].shape[0] == n_repr['h2'].shape[0] == N + 3

    # Test edge addition
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.set_e_repr({'h1': th.randn(2, D),
                  'h2': th.randn(2, D)})
    e_repr = g.get_e_repr()
    assert e_repr['h1'].shape[0] == e_repr['h2'].shape[0] == 2

    g.add_edges([0, 2], [2, 0])
    e_repr = g.get_e_repr()
    g.set_e_repr({'h1': th.randn(4, D)})
    assert e_repr['h1'].shape[0] == e_repr['h2'].shape[0] == 4

    g.add_edge(1, 2)
    g.set_e_repr_by_id({'h1': th.randn(1, D)}, eid=4)
    e_repr = g.get_e_repr()
    assert e_repr['h1'].shape[0] == e_repr['h2'].shape[0] == 5

if __name__ == '__main__':
    test_copy_src()
    test_copy_edge()
    test_src_mul_edge()
    test_dynamic_addition()
