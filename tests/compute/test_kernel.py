import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import backend as F
from itertools import product

np.random.seed(42)

def udf_copy_src(edges):
    return {'m': edges.src['u']}


def udf_copy_edge(edges):
    return {'m': edges.data['e']}


def udf_sum(nodes):
    return {'r2': nodes.mailbox['m'].sum(1)}


def udf_max(nodes):
    return {'r2': F.max(nodes.mailbox['m'], 1)}


D1 = 5
D2 = 3
D3 = 4
builtin = {'sum': fn.sum, 'max': fn.max}
udf_reduce = {'sum': udf_sum, 'max': udf_max}
fill_value = {'sum': 0, 'max': float("-inf")}


def generate_feature(g, broadcast='none'):
    """Create graph with src, edge, dst feature. broadcast can be 'u',
    'e', 'v', 'none'
    """
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if broadcast == 'e':
        u = F.tensor(np.random.randn(nv, D1, D2, D3) + 1)
        e = F.tensor(np.random.randn(ne, D2, 1) - 1)
        v = F.tensor(np.random.randn(nv, D1, D2, D3))
    elif broadcast == 'u':
        u = F.tensor(np.random.randn(nv, D2, 1) + 1)
        e = F.tensor(np.random.randn(ne, D1, D2, D3) - 1)
        v = F.tensor(np.random.randn(nv, D1, D2, D3))
    elif broadcast == 'v':
        u = F.tensor(np.random.randn(nv, D1, D2, D3) + 1)
        e = F.tensor(np.random.randn(ne, D1, D2, D3) - 1)
        v = F.tensor(np.random.randn(nv, D2, 1))
    else:
        u = F.tensor(np.random.randn(nv, D1, D2, D3) + 1)
        e = F.tensor(np.random.randn(ne, D1, D2, D3) - 1)
        v = F.tensor(np.random.randn(nv, D1, D2, D3))
    return u, v, e


def test_copy_src_reduce():
    def _test(red):
        g = dgl.DGLGraph(nx.erdos_renyi_graph(100, 0.1))
        hu, hv, he = generate_feature(g, 'none')

        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            g.update_all(fn.copy_src(src='u', out='m'),
                         builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            n_grad1 = F.grad(g.ndata['u'])

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            g.update_all(udf_copy_src, udf_reduce[red])
            r2 = g.ndata['r2']
            F.backward(r2.sum())
            n_grad2 = F.grad(g.ndata['u'])

        assert F.allclose(r1, r2)
        assert(F.allclose(n_grad1, n_grad2))

    _test('sum')
    _test('max')


def test_copy_edge_reduce():
    def _test(red):
        g = dgl.DGLGraph(nx.erdos_renyi_graph(100, 0.1))
        hu, hv, he = generate_feature(g, 'none')
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            g.update_all(fn.copy_edge(edge='e', out='m'),
                         builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            e_grad1 = F.grad(g.edata['e'])

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            g.update_all(udf_copy_edge, udf_reduce[red])
            r2 = g.ndata['r2']
            F.backward(r2.sum())
            e_grad2 = F.grad(g.edata['e'])

        assert F.allclose(r1, r2)
        assert(F.allclose(e_grad1, e_grad2))

    _test('sum')
    _test('max')


def test_all_binary_builtins():
    def _test(g, lhs, rhs, binary_op, reducer, broadcast='none'):
        hu, hv, he = generate_feature(g, broadcast)
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)
        builtin_red = getattr(fn, reducer)

        def target_feature_switch(g, target):
            if target == "u":
                return g.ndata["u"]
            elif target == "v":
                return g.ndata["v"]
            else:
                return g.edata["e"]

        with F.record_grad():
            g.update_all(builtin_msg(lhs, rhs, 'm'), builtin_red('m', 'r1'))
            r1 = g.ndata['r1']
            F.backward(r1.sum())
            lhs_grad_1 = F.grad(target_feature_switch(g, lhs))
            rhs_grad_1 = F.grad(target_feature_switch(g, rhs))

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

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
            return {"m": op(lhs_data[lhs], rhs_data[rhs])}

        def rfunc(nodes):
            op = getattr(F, reducer)
            return {"r2": op(nodes.mailbox['m'], 1)}

        with F.record_grad():
            g.update_all(mfunc, rfunc)
            r2 = g.ndata['r2']
            F.backward(r2.sum(), F.tensor([1.]))
            lhs_grad_2 = F.grad(target_feature_switch(g, lhs))
            rhs_grad_2 = F.grad(target_feature_switch(g, rhs))

        if reducer == 'prod':
            rtol = 1e-2
            atol = 1e-2
        else:
            rtol = 1e-4
            atol = 1e-4

        def _print_error(a, b):
            print("ERROR: Test {}_{}_{}_{} {}".
                  format(lhs, binary_op, rhs, reducer, broadcast))
            print(a, b)
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y, rtol, atol):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2, rtol, atol):
            _print_error(r1, r2)
        assert F.allclose(r1, r2, rtol, atol)

        if not F.allclose(lhs_grad_1, lhs_grad_2, rtol, atol):
            print("left grad")
            _print_error(lhs_grad_1, lhs_grad_2)
        assert(F.allclose(lhs_grad_1, lhs_grad_2, rtol, atol))

        if not F.allclose(rhs_grad_1, rhs_grad_2, rtol, atol):
            print("right grad")
            _print_error(rhs_grad_1, rhs_grad_2)
        assert(F.allclose(rhs_grad_1, rhs_grad_2, rtol, atol))

    g = dgl.DGLGraph()
    g.add_nodes(20)
    for i in range(2, 18):
        g.add_edge(0, i)
        g.add_edge(1, i)
        g.add_edge(i, 18)
        g.add_edge(i, 19)
    g.add_edge(18, 0)
    g.add_edge(18, 1)
    g.add_edge(19, 0)
    g.add_edge(19, 1)
    target = ["u", "v", "e"]
    for lhs, rhs in product(target, target):
        if lhs == rhs:
            continue
        for binary_op in ["add", "sub", "mul", "div"]:
            for reducer in ["sum", "max", "min", "prod"]:
                for broadcast in ["none", lhs, rhs]:
                    _test(g, lhs, rhs, binary_op, reducer)

if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_all_binary_builtins()
