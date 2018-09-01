import torch as th
import dgl
import dgl.function as fn
from dgl.graph import __REPR__

def generate_graph():
    g = dgl.DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
    h = th.arange(1, 11)
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
    for i in range(10):
        g.add_node(i) # 10 nodes.
    h = th.arange(1, 11)
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

def reducer_msg(node, msgs):
    return th.sum(msgs['m'], 1)

def reducer_out(node, msgs):
    return {'h' : th.sum(msgs, 1)}

def reducer_both(node, msgs):
    return {'h' : th.sum(msgs['m'], 1)}

def reducer_none(node, msgs):
    return th.sum(msgs, 1)

def test_copy_src():
    # copy_src with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h', out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy_src with only src field; the out field should use anonymous repr
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h'), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy_src with no src field; should use anonymous repr
    g = generate_graph1()
    g.register_message_func(fn.copy_src(out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy src with no fields;
    g = generate_graph1()
    g.register_message_func(fn.copy_src(), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_copy_edge():
    # copy_edge with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h', out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy_edge with only edge field; the out field should use anonymous repr
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h'), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy_edge with no edge field; should use anonymous repr
    g = generate_graph1()
    g.register_message_func(fn.copy_edge(out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

    # copy edge with no fields;
    g = generate_graph1()
    g.register_message_func(fn.copy_edge(), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_src_mul_edge():
    # src_mul_edge with all fields
    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h', out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h'), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(out='m'), batchable=True)
    g.register_reduce_func(reducer_both, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(), batchable=True)
    g.register_reduce_func(reducer_out, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr()['h'],
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(), batchable=True)
    g.register_reduce_func(reducer_none, batchable=True)
    g.update_all()
    assert th.allclose(g.get_n_repr(),
            th.tensor([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

if __name__ == '__main__':
    test_copy_src()
    test_copy_edge()
    test_src_mul_edge()
