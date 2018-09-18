import torch as th
import numpy as np
import dgl
import dgl.function as fn

D = 5

def generate_graph():
    g = dgl.DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
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
        g.update_all(fn.copy_src(src=fld), fn.sum(out=fld), apply_func, batchable=True)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func, reduce_func, apply_func, batchable=True)
        v3 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        # update all with edge weights
        v1 = g.get_n_repr()[fld]
        g.update_all(fn.src_mul_edge(src=fld, edge='e1'),
                fn.sum(out=fld), apply_func, batchable=True)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(fn.src_mul_edge(src=fld, edge='e2'),
                fn.sum(out=fld), apply_func, batchable=True)
        v3 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func_edge, reduce_func, apply_func, batchable=True)
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
                fn.sum(out=fld), apply_func, batchable=True)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, message_func,
                reduce_func, apply_func, batchable=True)
        v3 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        # send and recv with edge weights
        v1 = g.get_n_repr()[fld]
        g.send_and_recv(u, v, fn.src_mul_edge(src=fld, edge='e1'),
                fn.sum(out=fld), apply_func, batchable=True)
        v2 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, fn.src_mul_edge(src=fld, edge='e2'),
                fn.sum(out=fld), apply_func, batchable=True)
        v3 = g.get_n_repr()[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv(u, v, message_func_edge,
                reduce_func, apply_func, batchable=True)
        v4 = g.get_n_repr()[fld]
        assert th.allclose(v2, v3)
        assert th.allclose(v3, v4)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

if __name__ == '__main__':
    test_update_all()
    test_send_and_recv()
