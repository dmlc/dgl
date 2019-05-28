import dgl
import dgl.function as fn
import networkx as nx
import backend as F

def udf_src_x_edge(broadcast):
    def fn(edges):
        n = edges.src['n']
        e = edges.data['e']
        if broadcast == 'src':
            n = F.unsqueeze(n, 1)
        elif broadcast == 'edge':
            e = F.unsqueeze(e, 1)
        return {'m' : n * e}
    return fn

def udf_copy_src(edges):
    return {'m' : edges.src['n']}

def udf_copy_edge(edges):
    return {'m' : edges.data['e']}

def udf_sum(nodes):
    return {'r1' : nodes.mailbox['m'].sum(1)}

def udf_max(nodes):
    return {'r1' : F.max(nodes.mailbox['m'], 1)}

D1 = 5
D2 = 10
D3 = 4
builtin = {'sum': fn.sum, 'max': fn.max}
udf_reduce = {'sum': udf_sum, 'max': udf_max}
fill_value = {'sum': 0, 'max': float("-inf")}

def generate_graph(broadcast='none'):
    """Create graph with src, edge, dst feature. broadcast can be 'src',
    'edge', 'dst', 'none'
    """
    g = dgl.DGLGraph(nx.erdos_renyi_graph(100, 0.5))
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if broadcast == 'edge':
        n = F.randn((nv, D1, D2, D3))
        e = F.randn((ne, D2, 1))
        f = F.randn((nv, D1, D2, D3))
    elif broadcast == 'src':
        n = F.randn((nv, D2, 1))
        e = F.randn((ne, D1, D2, D3))
        f = F.randn((nv, D1, D2, D3))
    elif broadcast == 'dst':
        n = F.randn((nv, D1, D2, D3))
        e = F.randn((ne, D1, D2, D3))
        f = F.randn((nv, D2, 1))
    else:
        n = F.randn((nv, D1, D2, D3))
        e = F.randn((ne, D1, D2, D3))
        f = F.randn((nv, D1, D2, D3))

    g.ndata['n'] = F.attach_grad(n)
    g.ndata['f'] = F.attach_grad(f)
    g.edata['e'] = F.attach_grad(e)
    return g

def test_src_op_edge_reduce():
    def _test(red, broadcast='none'):
        g = generate_graph(broadcast)

        with F.record_grad():
            g.update_all(fn.src_mul_edge(src='n', edge='e', out='m'),
                        builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            n_grad1 = F.grad(g.ndata['n'])
            e_grad1 = F.grad(g.edata['e'])

        # reset grad
        F.attach_grad(g.ndata['n'])
        F.attach_grad(g.edata['e'])

        with F.record_grad():
            g.update_all(udf_src_x_edge(broadcast), udf_reduce[red])
            r2 = g.ndata['r1']
            F.backward(r2.sum())
            n_grad2 = F.grad(g.ndata['n'])
            e_grad2 = F.grad(g.edata['e'])

        assert F.allclose(r1, r2)
        assert(F.allclose(n_grad1, n_grad2))
        assert(F.allclose(e_grad1, e_grad2))

    _test('sum')
    _test('sum', 'src')
    _test('sum', 'edge')
    _test('max')
    _test('max', 'src')
    _test('max', 'edge')


def _test_src_op_dst_reduce():
    def _test(red, test_backward=False, broadcast='none'):
        g = generate_graph(broadcast)
        # test forward
        g.update_all(fn.src_mul_dst(src='n', dst='f', out='m'),
                     builtin[red](msg='m', out='r1'))
        r1 = g.ndata['r1']
        n1 = g.ndata['n'].detach().requires_grad_()
        f1 = g.ndata['f'].detach().requires_grad_()
        u, v = g.all_edges('uv')
        u, v = F.tensor(u), F.tensor(v)
        msg = n1[u] * f1[v]
        r2 = udf_reduce[red](msg, v, dim=0, fill_value=fill_value[red])
        assert F.allclose(r1, r2)

        # test backward
        if test_backward:
            r1.sum().backward()
            r2.sum().backward()
            assert(F.allclose(n1.grad, g.ndata['n'].grad))
            assert(F.allclose(f1.grad, g.ndata['f'].grad))

    _test('sum', True)
    _test('sum', True, 'src')
    _test('sum', True, 'dst')
    _test('max', True)
    _test('max', True, 'src')
    _test('max', True, 'dst')

def test_copy_src_reduce():
    def _test(red):
        g = generate_graph('none')

        with F.record_grad():
            g.update_all(fn.copy_src(src='n', out='m'),
                         builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            n_grad1 = F.grad(g.ndata['n'])

        # reset grad
        F.attach_grad(g.ndata['n'])

        with F.record_grad():
            g.update_all(udf_copy_src, udf_reduce[red])
            r2 = g.ndata['r1']
            F.backward(r2.sum())
            n_grad2 = F.grad(g.ndata['n'])

        assert F.allclose(r1, r2)
        assert(F.allclose(n_grad1, n_grad2))

    _test('sum')
    _test('max')


def test_copy_edge_reduce():
    def _test(red):
        g = generate_graph('none')

        with F.record_grad():
            g.update_all(fn.copy_edge(edge='e', out='m'),
                        builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            e_grad1 = F.grad(g.edata['e'])

        # reset grad
        F.attach_grad(g.edata['e'])

        with F.record_grad():
            g.update_all(udf_copy_edge, udf_reduce[red])
            r2 = g.ndata['r1']
            F.backward(r2.sum())
            e_grad2 = F.grad(g.edata['e'])

        assert F.allclose(r1, r2)
        assert(F.allclose(e_grad1, e_grad2))

    _test('sum')
    _test('max')


def _edge_softmax_ground_truth(a, dst, nv):
    shape = a.shape
    a = a.view(shape[0], -1)
    max_norm = scatter_max(a, dst, dim=0, fill_value=float("-inf"))
    a = a - max_norm.index_select(0, dst)
    a = a.exp()
    norm = scatter_add(a, dst, dim=0)
    a = a.view(shape)
    norm = norm.view((nv, ) + shape[1:])
    return a, norm


def _test_edge_softmax():
    g = generate_graph('src')  # generate high dim edge feature
    softmax = EdgeSoftmax()
    e = g.edata['e']
    e1 = e.detach().requires_grad_()
    a, norm = softmax(e, g)
    _, dst = g.edges()
    nv = g.number_of_nodes()
    a1, norm1 = _edge_softmax_ground_truth(e1, dst.cuda(), nv)
    assert(F.allclose(a, a1))
    assert(F.allclose(norm, norm1))
    loss = a.sum() + norm.sum()
    loss1 = a1.sum() + norm1.sum()
    loss.backward()
    loss1.backward()
    assert(F.allclose(e.grad, e1.grad))


if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_src_op_edge_reduce()
    # FIXME: expose backward of src_op_dst_reduce and enable this unit test
    # test_src_op_dst_reduce()
    #test_edge_softmax()
