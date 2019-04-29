import dgl
import torch
import torch_scatter
import dgl.function as fn


def allclose(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-4)


def scatter_max(*args, **kwargs):
    return torch_scatter.scatter_max(*args, **kwargs)[0]


D = 5
scatter_add = torch_scatter.scatter_add
builtin = {'sum': fn.sum, 'max': fn.max}
udf_reduce = {'sum': scatter_add, 'max': scatter_max}
fill_value = {'sum': 0, 'max': float("-inf")}


def generate_graph():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    g.add_edge(9, 0)
    g.ndata['n'] = torch.randn((10, D)).cuda().requires_grad_()
    g.ndata['f'] = torch.randn((10, D)).cuda().requires_grad_()
    g.edata['e'] = torch.randn((17, D)).cuda().requires_grad_()
    return g


def test_src_op_edge_reduce():
    def _test(red, test_backward=False):
        g = generate_graph()
        # test forward
        g.update_all(fn.src_mul_edge(src='n', edge='e', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        n1 = g.ndata['n'].detach().requires_grad_()
        e1 = g.edata['e'].detach().requires_grad_()
        u, v, eid = g.all_edges('all')
        u, v, eid = u.cuda(), v.cuda(), eid.cuda()
        msg = n1[u] * e1[eid]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(n1.grad, g.ndata['n'].grad))
            assert(allclose(e1.grad, g.edata['e'].grad))

    _test('sum', True)
    _test('max')


def test_src_op_dst_reduce():
    def _test(red, test_backward=False):
        g = generate_graph()
        # test forward
        g.update_all(fn.src_mul_dst(src='n', dst='f', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        n1 = g.ndata['n'].detach().requires_grad_()
        f1 = g.ndata['f'].detach().requires_grad_()
        u, v = g.all_edges('uv')
        u, v = u.cuda(), v.cuda()
        msg = n1[u] * f1[v]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(n1.grad, g.ndata['n'].grad))
            assert(allclose(f1.grad, g.ndata['f'].grad))

    _test('sum')
    _test('max')


def test_copy_src_reduce():
    def _test(red, test_backward=False):
        g = generate_graph()
        # test forward
        g.update_all(fn.copy_src(src='n', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        n1 = g.ndata['n'].detach().requires_grad_()
        u, v = g.all_edges('uv')
        u, v = u.cuda(), v.cuda()
        msg = n1[u]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(n1.grad, g.ndata['n'].grad))

    _test('sum', True)
    _test('max')


def test_copy_edge_reduce():
    def _test(red, test_backward=False):
        g = generate_graph()
        # test forward
        g.update_all(fn.copy_edge(edge='e', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        e1 = g.edata['e'].detach().requires_grad_()
        _, v, eid = g.all_edges('all')
        v, eid = v.cuda(), eid.cuda()
        msg = e1[eid]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(e1.grad, g.edata['e'].grad))

    _test('sum')
    _test('max')


if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_src_op_edge_reduce()
    test_src_op_dst_reduce()
