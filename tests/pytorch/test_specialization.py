import torch as th
import dgl
import dgl.function as fn
import utils as U

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

def test_v2v_update_all():
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : th.sum(nodes.mailbox['m'], 1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}
        g = generate_graph()
        # update all
        v1 = g.ndata[fld]
        g.update_all(fn.copy_src(src=fld, out='m'), fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.update_all(message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert U.allclose(v2, v3)
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
        assert U.allclose(v2, v3)
        assert U.allclose(v3, v4)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_v2v_snr():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : th.sum(nodes.mailbox['m'], 1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}
        g = generate_graph()
        # send and recv
        v1 = g.ndata[fld]
        g.send_and_recv((u, v), fn.copy_src(src=fld, out='m'),
                fn.sum(msg='m', out=fld), apply_func)
        v2 = g.ndata[fld]
        g.set_n_repr({fld : v1})
        g.send_and_recv((u, v), message_func, reduce_func, apply_func)
        v3 = g.ndata[fld]
        assert U.allclose(v2, v3)
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
        assert U.allclose(v2, v3)
        assert U.allclose(v3, v4)
    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_v2v_update_all_multi_fn():
    def message_func(edges):
        return {'m2': edges.src['f2']}

    def message_func_edge(edges):
        return {'m2': edges.src['f2'] * edges.data['e2']}

    def reduce_func(nodes):
        return {'v1': th.sum(nodes.mailbox['m2'], 1)}

    g = generate_graph()
    g.set_n_repr({'v1' : th.zeros((10,)), 'v2' : th.zeros((10,))})
    fld = 'f2'

    g.update_all(message_func, reduce_func)
    v1 = g.ndata['v1']

    # 1 message, 2 reduces
    g.update_all(fn.copy_src(src=fld, out='m'), [fn.sum(msg='m', out='v2'), fn.sum(msg='m', out='v3')])
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert U.allclose(v1, v2)
    assert U.allclose(v1, v3)

    # update all with edge weights, 2 message, 3 reduces
    g.update_all([fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                 [fn.sum(msg='m1', out='v1'), fn.sum(msg='m2', out='v2'), fn.sum(msg='m1', out='v3')],
                 None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert U.allclose(v1, v2)
    assert U.allclose(v1, v3)

    # run UDF with single message and reduce
    g.update_all(message_func_edge, reduce_func, None)
    v2 = g.ndata['v2']
    assert U.allclose(v1, v2)

def test_v2v_snr_multi_fn():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])

    def message_func(edges):
        return {'m2': edges.src['f2']}

    def message_func_edge(edges):
        return {'m2': edges.src['f2'] * edges.data['e2']}

    def reduce_func(nodes):
        return {'v1' : th.sum(nodes.mailbox['m2'], 1)}

    g = generate_graph()
    g.set_n_repr({'v1' : th.zeros((10, D)), 'v2' : th.zeros((10, D)),
        'v3' : th.zeros((10, D))})
    fld = 'f2'

    g.send_and_recv((u, v), message_func, reduce_func)
    v1 = g.ndata['v1']

    # 1 message, 2 reduces
    g.send_and_recv((u, v),
            fn.copy_src(src=fld, out='m'),
            [fn.sum(msg='m', out='v2'), fn.sum(msg='m', out='v3')],
            None)
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert U.allclose(v1, v2)
    assert U.allclose(v1, v3)

    # send and recv with edge weights, 2 message, 3 reduces
    g.send_and_recv((u, v),
                    [fn.src_mul_edge(src=fld, edge='e1', out='m1'), fn.src_mul_edge(src=fld, edge='e2', out='m2')],
                    [fn.sum(msg='m1', out='v1'), fn.sum(msg='m2', out='v2'), fn.sum(msg='m1', out='v3')],
                    None)
    v1 = g.ndata['v1']
    v2 = g.ndata['v2']
    v3 = g.ndata['v3']
    assert U.allclose(v1, v2)
    assert U.allclose(v1, v3)

    # run UDF with single message and reduce
    g.send_and_recv((u, v), message_func_edge,
            reduce_func, None)
    v2 = g.ndata['v2']
    assert U.allclose(v1, v2)

def test_e2v_reduce():
    def _test(fld):
        def message_func(edges):
            return {'m' : edges.src[fld]}

        def message_func_edge(edges):
            if len(edges.src[fld].shape) == 1:
                return {'m' : edges.src[fld] * edges.data['e1']}
            else:
                return {'m' : edges.src[fld] * edges.data['e2']}

        def reduce_func(nodes):
            return {fld : th.sum(nodes.mailbox['m'], 1)}

        def apply_func(nodes):
            return {fld : 2 * nodes.data[fld]}

        g = generate_graph()
        # update all
        v1 = g.get_n_repr()[fld]
        # no specialization
        g.update_all(message_func, reduce_func, apply_func)
        v2 = g.get_n_repr()[fld]

        g.set_n_repr({fld : v1})
        g.update_all(message_func, fn.sum(msg='m', out=fld), apply_func)
        v3 = g.get_n_repr()[fld]

        assert th.allclose(v2, v3)

    # test 1d node features
    _test('f1')
    # test 2d node features
    _test('f2')

def test_multi_fn_fallback():
    # create a graph with zero in degree nodes
    g = dgl.DGLGraph()
    g.add_nodes(10)
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    g.ndata['h'] = th.randn(10, D)
    g.edata['w1'] = th.randn(16,)
    g.edata['w2'] = th.randn(16, D)
    def _mfunc_hxw1(edges):
        return {'m1' : edges.src['h'] * th.unsqueeze(edges.data['w1'], 1)}
    def _mfunc_hxw2(edges):
        return {'m2' : edges.src['h'] * edges.data['w2']}
    def _rfunc_m1(nodes):
        return {'o1' : th.sum(nodes.mailbox['m1'], 1)}
    def _rfunc_m2(nodes):
        return {'o2' : th.sum(nodes.mailbox['m2'], 1)}
    def _rfunc_m1max(nodes):
        return {'o3' : th.max(nodes.mailbox['m1'], 1)[0]}
    def _afunc(nodes):
        ret = {}
        for k, v in nodes.data.items():
            if k.startswith('o'):
                ret[k] = 2 * v
        return ret
    # compute ground truth
    g.ndata['o1'] = th.zeros(10, D)  # init accum
    g.update_all(_mfunc_hxw1, _rfunc_m1, _afunc)
    o1 = g.ndata.pop('o1')
    g.ndata['o2'] = th.zeros(10, D)  # init accum
    g.update_all(_mfunc_hxw2, _rfunc_m2, _afunc)
    o2 = g.ndata.pop('o2')
    g.ndata['o3'] = th.zeros(10, D)  # init accum
    g.update_all(_mfunc_hxw1, _rfunc_m1max, _afunc)
    o3 = g.ndata.pop('o3')
    # v2v spmv
    g.update_all(fn.src_mul_edge(src='h', edge='w1', out='m1'),
                 fn.sum(msg='m1', out='o1'),
                 _afunc)
    assert U.allclose(o1, g.ndata.pop('o1'))
    # v2v fallback to e2v
    g.update_all(fn.src_mul_edge(src='h', edge='w2', out='m2'),
                 fn.sum(msg='m2', out='o2'),
                 _afunc)
    assert U.allclose(o2, g.ndata.pop('o2'))
    # v2v fallback to degree bucketing
    g.ndata['o3'] = th.zeros(10, D)  # init accum (this is necessary unfortunately)
    g.update_all(fn.src_mul_edge(src='h', edge='w1', out='m1'),
                 fn.max(msg='m1', out='o3'),
                 _afunc)
    assert U.allclose(o3, g.ndata.pop('o3'))
    # multi builtins, both v2v spmv
    g.update_all([fn.src_mul_edge(src='h', edge='w1', out='m1'), fn.src_mul_edge(src='h', edge='w1', out='m2')],
                 [fn.sum(msg='m1', out='o1'), fn.sum(msg='m2', out='o2')],
                 _afunc)
    assert U.allclose(o1, g.ndata.pop('o1'))
    assert U.allclose(o1, g.ndata.pop('o2'))
    # multi builtins, one v2v spmv, one fallback to e2v
    g.update_all([fn.src_mul_edge(src='h', edge='w1', out='m1'), fn.src_mul_edge(src='h', edge='w2', out='m2')],
                 [fn.sum(msg='m1', out='o1'), fn.sum(msg='m2', out='o2')],
                 _afunc)
    assert U.allclose(o1, g.ndata.pop('o1'))
    assert U.allclose(o2, g.ndata.pop('o2'))
    # multi builtins, one v2v spmv, one fallback to e2v, one fallback to degree-bucketing
    g.ndata['o3'] = th.zeros(10, D)  # init accum (this is necessary unfortunately)
    g.update_all([fn.src_mul_edge(src='h', edge='w1', out='m1'),
                  fn.src_mul_edge(src='h', edge='w2', out='m2'),
                  fn.src_mul_edge(src='h', edge='w1', out='m3')],
                 [fn.sum(msg='m1', out='o1'),
                  fn.sum(msg='m2', out='o2'),
                  fn.max(msg='m3', out='o3')],
                 _afunc)
    assert U.allclose(o1, g.ndata.pop('o1'))
    assert U.allclose(o2, g.ndata.pop('o2'))
    assert U.allclose(o3, g.ndata.pop('o3'))

if __name__ == '__main__':
    test_e2v_reduce()
    test_v2v_update_all()
    test_v2v_snr()
    test_v2v_update_all_multi_fn()
    test_v2v_snr_multi_fn()
    test_multi_fn_fallback()
