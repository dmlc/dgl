import dgl
import torch
import torch_scatter
import dgl.function as fn


def allclose(a, b):
    return torch.allclose(a.float(), b.float(), rtol=1e-4, atol=1e-4)


def scatter_max(*args, **kwargs):
    return torch_scatter.scatter_max(*args, **kwargs)[0]


D1 = 5
D2 = 10
D3 = 2
scatter_add = torch_scatter.scatter_add
builtin = {'sum': fn.sum, 'max': fn.max}
udf_reduce = {'sum': scatter_add, 'max': scatter_max}
fill_value = {'sum': 0, 'max': float("-inf")}


def generate_graph(broadcast='none'):
    """Create graph with src, edge, dst feature. broadcast can be 'src',
    'edge', 'dst', 'none'
    """
    g = dgl.DGLGraph()
    g.add_nodes(10)
    for i in range(1, 9):
        g.add_edges(0, i)
        g.add_edges(i, 9)
    g.add_edge(9, 0)
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if broadcast == 'edge':
        n = torch.randn((nv, D1, D2, D3))
        e = torch.randn((ne, D2, 1))
        f = torch.randn((nv, D1, D2, D3))
    elif broadcast == 'src':
        n = torch.randn((nv, D2, 1))
        e = torch.randn((ne, D1, D2, D3))
        f = torch.randn((nv, D1, D2, D3))
    elif broadcast == 'dst':
        n = torch.randn((nv, D1, D2, D3))
        e = torch.randn((ne, D1, D2, D3))
        f = torch.randn((nv, D2, 1))
    else:
        n = torch.randn((nv, D1))
        e = torch.randn((ne, D1))
        f = torch.randn((nv, D1))

    g.ndata['n'] = n.cuda().requires_grad_()
    g.ndata['f'] = f.cuda().requires_grad_()
    g.edata['e'] = e.cuda().requires_grad_()
    return g

def _tweak_shape(n, e, f, broadcast):
    if broadcast == 'src':
        n = n.unsqueeze(1)
    elif broadcast == 'dst':
        f = f.unsqueeze(1)
    elif broadcast == 'edge':
        e = e.unsqueeze(1)
    return n, e, f

def test_src_op_edge_reduce():
    def _test(red, test_backward=False, broadcast='none'):
        g = generate_graph(broadcast)
        n = g.ndata['n'].detach().requires_grad_()
        e = g.edata['e'].detach().requires_grad_()
        f = g.ndata['f'].detach().requires_grad_()

        # test forward
        g.update_all(fn.src_mul_edge(src='n', edge='e', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']

        u, v, eid = g.all_edges('all')
        u, v, eid = u.cuda(), v.cuda(), eid.cuda()
        nn, ee, _ = _tweak_shape(n, e, f, broadcast)
        msg = nn[u] * ee[eid]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(n.grad.view(n.shape), g.ndata['n'].grad))
            assert(allclose(e.grad.view(e.shape), g.edata['e'].grad))

    _test('sum', True)
    _test('sum', True, 'src')
    _test('sum', True, 'dst')
    _test('sum', True, 'edge')
    _test('max', True)
    _test('max', True, 'src')
    _test('max', True, 'dst')
    _test('max', True, 'edge')


def test_src_op_dst_reduce():
    def _test(red, test_backward=False, broadcast='none'):
        g = generate_graph(broadcast)
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

    _test('sum', True)
    _test('sum', True, 'src')
    _test('sum', True, 'dst')
    _test('sum', True, 'edge')
    _test('max', True)
    _test('max', True, 'src')
    _test('max', True, 'dst')
    _test('max', True, 'edge')


def test_copy_src_reduce():
    def _test(red, test_backward=False):
        g = generate_graph('none')
        n = g.ndata['n'].detach().requires_grad_()
        # test forward
        g.update_all(fn.copy_src(src='n', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        u, v = g.all_edges('uv')
        u, v = u.cuda(), v.cuda()
        msg = n[u]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(allclose(n.grad, g.ndata['n'].grad))

    _test('sum', True)
    _test('max', True)


def test_copy_edge_reduce():
    def _test(red, test_backward=False, broadcast='none'):
        g = generate_graph(broadcast)
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

    _test('sum', True)
    _test('max', True)


if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_src_op_edge_reduce()
    #test_src_op_dst_reduce()
