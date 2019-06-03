import dgl
import dgl.function as fn
import networkx as nx
import backend as F
from itertools import product

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
    def _test(red, broadcast="none"):
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


def test_all_binary_builtins():
    def _test(lhs, rhs, binary_op, reducer):
        g = dgl.DGLGraph()
        g.add_nodes(10)
        for i in range(1, 9):
            g.add_edge(0, i)
            g.add_edge(i, 9)
        g.add_edge(9, 0)
        nv = g.number_of_nodes()
        ne = g.number_of_edges()
        hv = F.randn((nv, D1))
        he = F.randn((ne, D1))
        g.ndata['h'] = F.attach_grad(hv)
        g.edata['h'] = F.attach_grad(he)

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)
        builtin_red = getattr(fn, reducer)

        with F.record_grad():
            g.update_all(builtin_msg('h', 'h', 'm'), builtin_red('m', 'r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            n_grad1 = F.grad(g.ndata['h'])
            e_grad1 = F.grad(g.edata['h'])

        # reset grad
        F.attach_grad(g.ndata['h'])
        F.attach_grad(g.edata['h'])

        def target_switch(edges, target):
            if target == "u":
                return edges.src
            elif target == "v":
                return edges.dst
            elif target == "e":
                return edges.data
            else:
                assert(0), "Unknown target {}".format(target)

        def mfunc(edges):
            op = getattr(F, binary_op)
            lhs_data = target_switch(edges, lhs)
            rhs_data = target_switch(edges, rhs)
            return {"m": op(lhs_data['h'], rhs_data['h'])}

        def rfunc(nodes):
            op = getattr(F, reducer)
            return {"r2":op(nodes.mailbox['m'], 1)}

        with F.record_grad():
            g.update_all(mfunc, rfunc)
            r2 = g.ndata['r2']
            F.backward(r2.sum())
            n_grad2 = F.grad(g.ndata['h'])
            e_grad2 = F.grad(g.edata['h'])

        assert F.allclose(r1, r2)
        if n_grad2 is not None:
            assert(F.allclose(n_grad1, n_grad2))
        if e_grad2 is not None:
            assert(F.allclose(e_grad1, e_grad2))

    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs == rhs:
            continue
        for reducer in ["sum", "max", "min", "prod"]:
            for binary_op in ["add", "sub", "mul", "div"]:
                _test(lhs, rhs, binary_op, reducer)


if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_src_op_edge_reduce()
    test_all_binary_builtins()
