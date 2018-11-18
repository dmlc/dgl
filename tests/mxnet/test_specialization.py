import mxnet as mx
from mxnet import autograd
import scipy as sp
import numpy as np
import dgl
import dgl.function as fn

D = 5

mx.random.seed(1)
np.random.seed(1)

def generate_test():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    return g

def generate_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = dgl.DGLGraph(arr, readonly=True)
    num_nodes = g.number_of_nodes()
    g.set_n_repr({'f1' : mx.nd.random.normal(shape=(num_nodes,)),
        'f2' : mx.nd.random.normal(shape=(num_nodes, D))})
    weights = mx.nd.random.normal(shape=(g.number_of_edges(),))
    g.set_e_repr({'e1': weights, 'e2': mx.nd.expand_dims(weights, axis=1)})
    return g

def generate_graph2(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g1 = dgl.DGLGraph(arr, readonly=True)
    g2 = dgl.DGLGraph(arr, readonly=True)
    #g1 = generate_test()
    #g2 = generate_test()
    num_nodes = g1.number_of_nodes()
    g1.set_n_repr({'f1' : mx.nd.random.normal(shape=(num_nodes,)),
        'f2' : mx.nd.random.normal(shape=(num_nodes, D))})
    weights = mx.nd.random.normal(shape=(g1.number_of_edges(),))
    g1.set_e_repr({'e1': weights, 'e2': mx.nd.expand_dims(weights, axis=1)})

    g2.set_n_repr({'f1' : g1.ndata['f1'].copy(), 'f2' : g1.ndata['f2'].copy()})
    g2.set_e_repr({'e1': g1.edata['e1'].copy(), 'e2': g1.edata['e2'].copy()})

    return g1, g2

def test_update_all():
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : mx.nd.sum(nodes.mailbox['m'], axis=1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}
        g = generate_graph(100)
        # update all
        v1 = g.ndata[fld]
        g.update_all(fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        # update all with edge weights
        v1 = g.ndata[fld]
        g.update_all(fn.src_mul_edge(src=fld, edge='e1', out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(fn.src_mul_edge(src=fld, edge='e2', out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v3 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        assert np.allclose(v3.asnumpy(), v4.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_send_and_recv():
    u = mx.nd.array([0, 0, 0, 3, 4, 9], dtype=np.int64)
    v = mx.nd.array([1, 2, 3, 9, 9, 0], dtype=np.int64)
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : mx.nd.sum(nodes.mailbox['m'], axis=1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}

        g = generate_graph(100)
        # send and recv
        v1 = g.ndata[fld]
        g.send_and_recv((u, v), fn.copy_src(src=fld, out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv((u, v), message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        # send and recv with edge weights
        v1 = g.ndata[fld]
        g.send_and_recv((u, v), fn.src_mul_edge(src=fld, edge='e1', out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv((u, v), fn.src_mul_edge(src=fld, edge='e2', out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v3 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv((u, v), message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        assert np.allclose(v3.asnumpy(), v4.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_update_all_multi_fn():
    def message_func(edges):
        return {'m2': edges.src['f2']}

    def message_func_edge(edges):
        return {'m2': edges.src['f2'] * edges.data['e2']}

    def reduce_func(nodes):
        return {'v2': mx.nd.sum(nodes.mailbox['m2'], axis=1)}

    g = generate_graph(100)
    g.set_n_repr({'v1' : mx.nd.zeros(shape=(g.number_of_nodes(),)),
        'v2' : mx.nd.zeros(shape=(g.number_of_nodes(),))})
    fld = 'f2'
    # update all, mix of builtin and UDF
    g.update_all([fn.copy_src(src=fld, out='m1'), message_func],
                 [fn.sum(msg='m1', out='v1'), reduce_func],
                 None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

    # run builtin with single message and reduce
    g.update_all(fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out='v1'), None)
    v1 = g.ndata['v1']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

    # 1 message, 2 reduces
    g.update_all(fn.copy_src(src=fld, out='m'), [fn.sum(msg='m', out='v2'), fn.sum(msg='m', out='v3')], None)
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)
    assert np.allclose(v1.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

    # update all with edge weights, 2 message, 3 reduces
    g.update_all([fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                 [fn.sum(msg='m1', out='v1'), fn.sum(msg='m2', out='v2'), fn.sum(msg='m1', out='v3')],
                 None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)
    assert np.allclose(v1.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

    # run UDF with single message and reduce
    g.update_all(message_func_edge, reduce_func, None)
    v2 = g.ndata['v2']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

def test_send_and_recv_multi_fn():
    u = mx.nd.array([0, 0, 0, 3, 4, 9], dtype=np.int64)
    v = mx.nd.array([1, 2, 3, 9, 9, 0], dtype=np.int64)

    def message_func(edges):
        return {'m2': edges.src['f2']}

    def message_func_edge(edges):
        return {'m2': edges.src['f2'] * edges.data['e2']}

    def reduce_func(nodes):
        return {'v2' : mx.nd.sum(nodes.mailbox['m2'], axis=1)}

    g = generate_graph(100)
    g.set_n_repr({'v1' : mx.nd.zeros(shape=(g.number_of_nodes(), D)),
        'v2' : mx.nd.zeros(shape=(g.number_of_nodes(), D)),
        'v3' : mx.nd.zeros(shape=(g.number_of_nodes(), D))})
    fld = 'f2'

    # send and recv, mix of builtin and UDF
    g.send_and_recv((u, v),
                    [fn.copy_src(src=fld, out='m1'), message_func],
                    [fn.sum(msg='m1', out='v1'), reduce_func],
                    None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

    # run builtin with single message and reduce
    g.send_and_recv((u, v), fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out='v1'),
                    None)
    v1 = g.ndata['v1']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

    # 1 message, 2 reduces
    g.send_and_recv((u, v),
            fn.copy_src(src=fld, out='m'),
            [fn.sum(msg='m', out='v2'), fn.sum(msg='m', out='v3')],
            None)
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)
    assert np.allclose(v1.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

    # send and recv with edge weights, 2 message, 3 reduces
    g.send_and_recv((u, v),
                    [fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                    [fn.sum(msg='m1', out='v1'), fn.sum(msg='m2', out='v2'), fn.sum(msg='m1', out='v3')],
                    None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)
    assert np.allclose(v1.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

    # run UDF with single message and reduce
    g.send_and_recv((u, v), message_func_edge,
            reduce_func, None)
    v2 = g.ndata['v2']
    assert np.allclose(v1.asnumpy(), v2.asnumpy(), rtol=1e-05, atol=1e-05)

if __name__ == '__main__':
    test_update_all()
    test_send_and_recv()
    test_update_all_multi_fn()
    test_send_and_recv_multi_fn()
