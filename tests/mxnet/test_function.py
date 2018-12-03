import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
import dgl
import dgl.function as fn

def generate_graph():
    g = dgl.DGLGraph()
    g.add_nodes(10) # 10 nodes.
    h = mx.nd.arange(1, 11, dtype=np.float32)
    g.ndata['h'] = h
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    h = mx.nd.array([1., 2., 1., 3., 1., 4., 1., 5., 1., 6.,\
            1., 7., 1., 8., 1., 9., 10.])
    g.edata['h'] = h
    return g

def reducer_both(nodes):
    return {'h' : mx.nd.sum(nodes.mailbox['m'], 1)}

def test_copy_src():
    # copy_src with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_src(src='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert np.allclose(g.ndata['h'].asnumpy(),
            np.array([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_copy_edge():
    # copy_edge with both fields
    g = generate_graph()
    g.register_message_func(fn.copy_edge(edge='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert np.allclose(g.ndata['h'].asnumpy(),
            np.array([10., 1., 1., 1., 1., 1., 1., 1., 1., 44.]))

def test_src_mul_edge():
    # src_mul_edge with all fields
    g = generate_graph()
    g.register_message_func(fn.src_mul_edge(src='h', edge='h', out='m'))
    g.register_reduce_func(reducer_both)
    g.update_all()
    assert np.allclose(g.ndata['h'].asnumpy(),
            np.array([100., 1., 1., 1., 1., 1., 1., 1., 1., 284.]))

if __name__ == '__main__':
    test_copy_src()
    test_copy_edge()
    test_src_mul_edge()
