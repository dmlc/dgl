import torch as th
import numpy as np
import dgl
import dgl.function as fn

D = 5

def generate_graph():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    g.set_n_repr({'f1' : th.randn(10,), 'f2' : th.randn(10, D)})
    weights = th.randn(17,)
    g.set_e_repr({'e1': weights, 'e2': th.unsqueeze(weights, 1)})
    return g

def test_update_all():
    def _test(fld):
        def message_func(hu, edge):
            return hu[fld]

        def message_func_edge(hu, edge):
            if len(hu[fld].shape) == 1:
                return hu[fld] * edge['e1']
            else:
                return hu[fld] * edge['e2']

        def reduce_func(hv, msgs):
            return {fld : th.sum(msgs, 1)}

        def apply_func(hu):
            return {fld : 2 * hu[fld]}
        g = generate_graph()
        # update all
        v1 = g.get_n_repr()[fld]
        g.update_all(fn.copy_src(src=fld), fn.sum(out=fld), apply_func)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func, reduce_func, apply_func)
        v3 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        # update all with edge weights
        v1 = g.get_n_repr()[fld]
        g.update_all(fn.src_mul_edge(src=fld, edge='e1'),
                fn.sum(out=fld), apply_func)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(fn.src_mul_edge(src=fld, edge='e2'),
                fn.sum(out=fld), apply_func)
        v3 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func_edge, reduce_func, apply_func)
        v4 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        assert th.allclose(v3, v4)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_send_and_recv():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])
    def _test(fld):
        def message_func(hu, edge):
            return hu[fld]

        def message_func_edge(hu, edge):
            if len(hu[fld].shape) == 1:
                return hu[fld] * edge['e1']
            else:
                return hu[fld] * edge['e2']

        def reduce_func(hv, msgs):
            return {fld : th.sum(msgs, 1)}

        def apply_func(hu):
            return {fld : 2 * hu[fld]}
        g = generate_graph()
        # send and recv
        v1 = g.get_n_repr()[fld]
        g.send_and_recv(u, v, fn.copy_src(src=fld),
                fn.sum(out=fld), apply_func)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, message_func,
                reduce_func, apply_func)
        v3 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        # send and recv with edge weights
        v1 = g.get_n_repr()[fld]
        g.send_and_recv(u, v, fn.src_mul_edge(src=fld, edge='e1'),
                fn.sum(out=fld), apply_func)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, fn.src_mul_edge(src=fld, edge='e2'),
                fn.sum(out=fld), apply_func)
        v3 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, message_func_edge,
                reduce_func, apply_func)
        v4 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        assert th.allclose(v3, v4)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_update_all_multi_fn():
    def message_func(hu, edge):
        return {'m2': hu['f2']}

    def message_func_edge(hu, edge):
        return {'m2': hu['f2'] * edge['e2']}

    def reduce_func(hv, msgs):
        return {'v2': th.sum(msgs['m2'], 1)}

    g = generate_graph()
    g.set_n_repr({'v1' : th.zeros((10,)), 'v2' : th.zeros((10,))})
    fld = 'f2'
    # update all, mix of builtin and UDF
    g.update_all([fn.copy_src(src=fld, out='m1'), message_func],
                 [fn.sum(msgs='m1', out='v1'), reduce_func],
                 None)
    v1 = g.get_n_repr()['v1']
    v2 = g.get_n_repr()['v2']
    assert th.allclose(v1, v2)

    # run builtin with single message and reduce
    g.update_all(fn.copy_src(src=fld), fn.sum(out='v1'), None)
    v1 = g.get_n_repr()['v1']
    assert th.allclose(v1, v2)

    # 1 message, 2 reduces, using anonymous repr
    g.update_all(fn.copy_src(src=fld), [fn.sum(out='v2'), fn.sum(out='v3')], None)
    v2 = g.get_n_repr()['v2']
    v3 = g.get_n_repr()['v3']
    assert th.allclose(v1, v2)
    assert th.allclose(v1, v3)

    # update all with edge weights, 2 message, 3 reduces
    g.update_all([fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                 [fn.sum(msgs='m1', out='v1'), fn.sum(msgs='m2', out='v2'), fn.sum(msgs='m1', out='v3')],
                 None)
    v1 = g.get_n_repr()['v1']
    v2 = g.get_n_repr()['v2']
    v3 = g.get_n_repr()['v3']
    assert th.allclose(v1, v2)
    assert th.allclose(v1, v3)

    # run UDF with single message and reduce
    g.update_all(message_func_edge, reduce_func, None)
    v2 = g.get_n_repr()['v2']
    assert th.allclose(v1, v2)

def test_send_and_recv_multi_fn():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])

    def message_func(hu, edge):
        return {'m2': hu['f2']}

    def message_func_edge(hu, edge):
        return {'m2': hu['f2'] * edge['e2']}

    def reduce_func(hv, msgs):
        return {'v2' : th.sum(msgs['m2'], 1)}

    g = generate_graph()
    g.set_n_repr({'v1' : th.zeros((10, D)), 'v2' : th.zeros((10, D)),
        'v3' : th.zeros((10, D))})
    fld = 'f2'

    # send and recv, mix of builtin and UDF
    g.send_and_recv(u, v,
                    [fn.copy_src(src=fld, out='m1'), message_func],
                    [fn.sum(msgs='m1', out='v1'), reduce_func],
                    None)
    v1 = g.get_n_repr()['v1']
    v2 = g.get_n_repr()['v2']
    assert th.allclose(v1, v2)

    # run builtin with single message and reduce
    g.send_and_recv(u, v, fn.copy_src(src=fld), fn.sum(out='v1'),
                    None)
    v1 = g.get_n_repr()['v1']
    assert th.allclose(v1, v2)

    # 1 message, 2 reduces, using anonymous repr
    g.send_and_recv(u, v, fn.copy_src(src=fld), [fn.sum(out='v2'), fn.sum(out='v3')], None)
    v2 = g.get_n_repr()['v2']
    v3 = g.get_n_repr()['v3']
    assert th.allclose(v1, v2)
    assert th.allclose(v1, v3)

    # send and recv with edge weights, 2 message, 3 reduces
    g.send_and_recv(u, v,
                    [fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                    [fn.sum(msgs='m1', out='v1'), fn.sum(msgs='m2', out='v2'), fn.sum(msgs='m1', out='v3')],
                    None)
    v1 = g.get_n_repr()['v1']
    v2 = g.get_n_repr()['v2']
    v3 = g.get_n_repr()['v3']
    assert th.allclose(v1, v2)
    assert th.allclose(v1, v3)

    # run UDF with single message and reduce
    g.send_and_recv(u, v, message_func_edge,
            reduce_func, None)
    v2 = g.get_n_repr()['v2']
    assert th.allclose(v1, v2)

if __name__ == '__main__':
    test_update_all()
    test_send_and_recv()
    test_update_all_multi_fn()
    test_send_and_recv_multi_fn()
