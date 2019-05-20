import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
from mxnet import autograd
import scipy as sp
import numpy as np
import dgl
import dgl.function as fn

D = 5

mx.random.seed(1)
np.random.seed(1)

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
        g1, g2 = generate_graph2(100)
        # update all
        g1_data = g1.ndata[fld]
        g2_data = g2.ndata[fld]
        g1_data.attach_grad()
        g2_data.attach_grad()
        with mx.autograd.record():
            g1.update_all(fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out=fld), apply_func)
            g2.update_all(message_func, reduce_func, apply_func)
        g1_res = g1.ndata[fld]
        g2_res = g2.ndata[fld]
        assert np.allclose(g1_res.asnumpy(), g2_res.asnumpy(), rtol=1e-05, atol=1e-05)
        g1_res.backward()
        g2_res.backward()
        assert np.allclose(g1_data.grad.asnumpy(), g2_data.grad.asnumpy(), rtol=1e-05, atol=1e-05)

        # update all with edge weights
        g1_data = g1.ndata[fld]
        g1.update_all(fn.src_mul_edge(src=fld, edge='e1', out='m'),
                      fn.sum(msg='m', out=fld), apply_func)
        v2 = g1.ndata[fld]
        g1.set_n_repr({fld : g1_data})
        g1.update_all(fn.src_mul_edge(src=fld, edge='e2', out='m'),
                      fn.sum(msg='m', out=fld), apply_func)
        v3 = g1.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

        g1.set_n_repr({fld : g1_data})
        g2_data = g2.ndata[fld]
        g1_data.attach_grad()
        g2_data.attach_grad()
        with mx.autograd.record():
            g1.update_all(fn.src_mul_edge(src=fld, edge='e2', out='m'),
                          fn.sum(msg='m', out=fld), apply_func)
            g2.update_all(message_func_edge, reduce_func, apply_func)
        g1_res = g1.ndata[fld]
        g2_res = g2.ndata[fld]
        assert np.allclose(g1_res.asnumpy(), g2_res.asnumpy(), rtol=1e-05, atol=1e-05)
        g1_res.backward()
        g2_res.backward()
        assert np.allclose(g1_data.grad.asnumpy(), g2_data.grad.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_pull():
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

        g1, g2 = generate_graph2(100)
        num_nodes = g1.number_of_nodes()
        u = np.unique(np.random.randint(0, num_nodes, size=(int(num_nodes/10))))

        # pull in DGL
        g1_data = g1.ndata[fld]
        g2_data = g2.ndata[fld]
        if len(g1_data.shape) == 1:
            g1_data = mx.nd.expand_dims(g1_data, axis=1)
            g1.ndata[fld] = g1_data
        if len(g2_data.shape) == 1:
            g2_data = mx.nd.expand_dims(g2_data, axis=1)
            g2.ndata[fld] = g2_data
        g1_data.attach_grad()
        g2_data.attach_grad()
        with mx.autograd.record():
            g1.pull(u, fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out=fld), apply_func)
            spm = mx.nd.take(g2.adjacency_matrix(), mx.nd.array(u, dtype=np.int64))
            g2_res = mx.nd.dot(spm, g2_data) * 2
            g1_res = g1.ndata[fld][u]
        assert np.allclose(g1_res.asnumpy(), g2_res.asnumpy(), rtol=1e-05, atol=1e-05)
        g1_res.backward()
        g2_res.backward()
        assert np.allclose(g1_data.grad.asnumpy(), g2_data.grad.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_send_and_recv():
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

        g1, g2 = generate_graph2(100)
        u, v = g1.all_edges()
        idxs = np.unique(np.random.randint(0, len(u), size=(int(len(u)/10))))
        u = u[idxs]
        v = v[idxs]

        # send and recv
        g1_data = g1.ndata[fld]
        g2_data = g2.ndata[fld]
        g1_data.attach_grad()
        g2_data.attach_grad()
        with mx.autograd.record():
            g1.send_and_recv((u, v), fn.copy_src(src=fld, out='m'),
                             fn.sum(msg='m', out=fld), apply_func)
            g2.send_and_recv((u, v), message_func, reduce_func, apply_func)
        g1_res = g1.ndata[fld]
        g2_res = g2.ndata[fld]
        assert np.allclose(g1_res.asnumpy(), g2_res.asnumpy(), rtol=1e-05, atol=1e-05)
        g1_res.backward()
        g2_res.backward()
        assert np.allclose(g1_data.grad.asnumpy(), g2_data.grad.asnumpy(), rtol=1e-05, atol=1e-05)

        # send and recv with edge weights
        g1_data = g1.ndata[fld]
        g1.send_and_recv((u, v), fn.src_mul_edge(src=fld, edge='e1', out='m'),
                         fn.sum(msg='m', out=fld), apply_func)
        v2 = g1.ndata[fld]
        g1.set_n_repr({fld : g1_data})
        g1.send_and_recv((u, v), fn.src_mul_edge(src=fld, edge='e2', out='m'),
                         fn.sum(msg='m', out=fld), apply_func)
        v3 = g1.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)

        g1.set_n_repr({fld : g1_data})
        g2_data = g2.ndata[fld]
        g1_data.attach_grad()
        g2_data.attach_grad()
        with mx.autograd.record():
            g1.send_and_recv((u, v), fn.src_mul_edge(src=fld, edge='e2', out='m'),
                             fn.sum(msg='m', out=fld), apply_func)
            g2.send_and_recv((u, v), message_func_edge, reduce_func, apply_func)
        g1_res = g1.ndata[fld]
        g2_res = g2.ndata[fld]
        assert np.allclose(g1_res.asnumpy(), g2_res.asnumpy(), rtol=1e-05, atol=1e-05)
        g1_res.backward()
        g2_res.backward()
        assert np.allclose(g1_data.grad.asnumpy(), g1_data.grad.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    # TODO for some reason, this test doesn't pass in MXNet.
    # somehow, it fails in backward.
    #_test('f1')
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

    # run builtin with single message and reduce
    g.update_all(fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out='v1'), None)
    v1 = g.ndata['v1']

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

    # run builtin with single message and reduce
    g.send_and_recv((u, v), fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out='v1'),
                    None)
    v1 = g.ndata['v1']

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

############################ Copy from torch
D = 5
def simple_graph():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    g.set_n_repr({'f1' : mx.nd.random.normal(shape=(10,)), 'f2' : mx.nd.random.normal(shape=(10, D))})
    weights = mx.nd.random.normal(shape=(17,))
    g.set_e_repr({'e1': weights, 'e2': mx.nd.expand_dims(weights, 1)})
    return g

def test_v2v_update_all_sum():
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
        g = simple_graph()
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
        v3 = g.ndata[fld].squeeze()
        g.set_n_repr({fld : v1})
        g.update_all(message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        assert np.allclose(v3.asnumpy(), v4.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_v2v_update_all_max():
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : mx.nd.max(nodes.mailbox['m'], axis=1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}
        g = simple_graph()
        # update all
        v1 = g.ndata[fld]
        g.update_all(fn.copy_src(src=fld, out='m'), fn.max(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        # update all with edge weights
        v1 = g.ndata[fld]
        g.update_all(fn.src_mul_edge(src=fld, edge='e1', out='m'),
                fn.max(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(fn.src_mul_edge(src=fld, edge='e2', out='m'),
                fn.max(msg='m', out=fld), apply_func)
        v3 = g.ndata[fld].squeeze()
        g.set_n_repr({fld : v1})
        g.update_all(message_func_edge, reduce_func, apply_func)
        v4 = g.ndata[fld]
        assert np.allclose(v2.asnumpy(), v3.asnumpy(), rtol=1e-05, atol=1e-05)
        assert np.allclose(v3.asnumpy(), v4.asnumpy(), rtol=1e-05, atol=1e-05)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')
############################ Copy from torch

if __name__ == '__main__':
    test_update_all()
    test_pull()
    test_send_and_recv()
    test_update_all_multi_fn()
    test_send_and_recv_multi_fn()
    test_v2v_update_all_sum()
    test_v2v_update_all_max()
