import dgl
import dgl.function as fn
from dgl.graph import __REPR__

def generate_graph():
    g = dgl.DGLGraph()
    for i in range(10):
        g.add_node(i, h=i+1) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i, h=1)
        g.add_edge(i, 9, h=i+1)
    # add a back flow from 9 to 0
    g.add_edge(9, 0, h=10)
    return g

def check(g, h, fld):
    nh = [str(g.nodes[i][fld]) for i in range(10)]
    h = [str(x) for x in h]
    assert nh == h, "nh=[%s], h=[%s]" % (' '.join(nh), ' '.join(h))

def generate_graph1():
    """graph with anonymous repr"""
    g = dgl.DGLGraph()
    for i in range(10):
        g.add_node(i, __REPR__=i+1) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i, __REPR__=1)
        g.add_edge(i, 9, __REPR__=i+1)
    # add a back flow from 9 to 0
    g.add_edge(9, 0, __REPR__=10)
    return g

def test_copy_src():
    # copy_src with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h', out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy_src with only src field; the out field should use anonymous repr
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h'), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy_src with no src field; should use anonymous repr
    g = generate_graph1()
    g.register_message_func(fn.copy_src(out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy src with no fields;
    g = generate_graph1()
    g.register_message_func(fn.copy_src(), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

def test_copy_edge():
    # copy_edge with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h', out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy_edge with only edge field; the out field should use anonymous repr
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h'), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy_edge with no edge field; should use anonymous repr
    g = generate_graph1()
    g.register_message_func(fn.copy_edge(out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

    # copy edge with no fields;
    g = generate_graph1()
    g.register_message_func(fn.copy_edge(), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [10, 1, 1, 1, 1, 1, 1, 1, 1, 44], 'h')

def test_src_mul_edge():
    # src_mul_edge with all fields
    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h', out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [100, 1, 1, 1, 1, 1, 1, 1, 1, 284], 'h')

    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h'), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'h')
    g.update_all()
    check(g, [100, 1, 1, 1, 1, 1, 1, 1, 1, 284], 'h')

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(out='m'), batchable=False)
    g.register_reduce_func(fn.sum(msgs='m', out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [100, 1, 1, 1, 1, 1, 1, 1, 1, 284], 'h')

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(), batchable=False)
    g.register_reduce_func(fn.sum(out='h'), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [100, 1, 1, 1, 1, 1, 1, 1, 1, 284], 'h')

    g = generate_graph1()
    g.register_message_func(fn.src_mul_edge(), batchable=False)
    g.register_reduce_func(fn.sum(), batchable=False)
    check(g, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], __REPR__)
    g.update_all()
    check(g, [100, 1, 1, 1, 1, 1, 1, 1, 1, 284], __REPR__)

if __name__ == '__main__':
    test_copy_src()
    test_copy_edge()
    test_src_mul_edge()
