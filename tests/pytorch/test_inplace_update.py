import torch as th
import numpy as np
import scipy.sparse as sp
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
    g.ndata['f'] = th.randn(10, D)
    g.edata['e'] = th.randn(17, D)
    return g

def test_inplace_recv():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])

    def message_func(edges):
        return {'m' : edges.src['f'] + edges.dst['f']}

    def reduce_func(nodes):
        return {'f' : th.sum(nodes.mailbox['m'], 1)}

    def apply_func(nodes):
        return {'f' : 2 * nodes.data['f']}

    def _test(apply_func):
        g = generate_graph()
        f = g.ndata['f']

        # one out place run to get result
        g.send((u, v), message_func)
        g.recv([0,1,2,3,9], reduce_func, apply_func)
        result = g.get_n_repr()['f']

        # inplace deg bucket run
        v1 = f.clone()
        g.ndata['f'] = v1
        g.send((u, v), message_func)
        g.recv([0,1,2,3,9], reduce_func, apply_func, inplace=True)
        r1 = g.get_n_repr()['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace e2v
        v1 = f.clone()
        g.ndata['f'] = v1
        g.send((u, v), message_func)
        g.recv([0,1,2,3,9], fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

    # test send_and_recv with apply_func
    _test(apply_func)
    # test send_and_recv without apply_func
    _test(None)

def test_inplace_snr():
    u = th.tensor([0, 0, 0, 3, 4, 9])
    v = th.tensor([1, 2, 3, 9, 9, 0])

    def message_func(edges):
        return {'m' : edges.src['f']}

    def reduce_func(nodes):
        return {'f' : th.sum(nodes.mailbox['m'], 1)}

    def apply_func(nodes):
        return {'f' : 2 * nodes.data['f']}

    def _test(apply_func):
        g = generate_graph()
        f = g.ndata['f']

        # an out place run to get result
        g.send_and_recv((u, v), fn.copy_src(src='f', out='m'),
                fn.sum(msg='m', out='f'), apply_func)
        result = g.ndata['f']

        # inplace deg bucket
        v1 = f.clone()
        g.ndata['f'] = v1
        g.send_and_recv((u, v), message_func, reduce_func, apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace v2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.send_and_recv((u, v), fn.copy_src(src='f', out='m'),
                        fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace e2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.send_and_recv((u, v), message_func,
                        fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

    # test send_and_recv with apply_func
    _test(apply_func)
    # test send_and_recv without apply_func
    _test(None)

def test_inplace_push():
    nodes = th.tensor([0, 3, 4, 9])

    def message_func(edges):
        return {'m' : edges.src['f']}

    def reduce_func(nodes):
        return {'f' : th.sum(nodes.mailbox['m'], 1)}

    def apply_func(nodes):
        return {'f' : 2 * nodes.data['f']}

    def _test(apply_func):
        g = generate_graph()
        f = g.ndata['f']

        # an out place run to get result
        g.push(nodes,
               fn.copy_src(src='f', out='m'), fn.sum(msg='m', out='f'), apply_func)
        result = g.ndata['f']

        # inplace deg bucket
        v1 = f.clone()
        g.ndata['f'] = v1
        g.push(nodes, message_func, reduce_func, apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace v2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.push(nodes, fn.copy_src(src='f', out='m'),
               fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace e2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.push(nodes,
               message_func, fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

    # test send_and_recv with apply_func
    _test(apply_func)
    # test send_and_recv without apply_func
    _test(None)

def test_inplace_pull():
    nodes = th.tensor([1, 2, 3, 9])

    def message_func(edges):
        return {'m' : edges.src['f']}

    def reduce_func(nodes):
        return {'f' : th.sum(nodes.mailbox['m'], 1)}

    def apply_func(nodes):
        return {'f' : 2 * nodes.data['f']}

    def _test(apply_func):
        g = generate_graph()
        f = g.ndata['f']

        # an out place run to get result
        g.pull(nodes,
               fn.copy_src(src='f', out='m'), fn.sum(msg='m', out='f'), apply_func)
        result = g.ndata['f']

        # inplace deg bucket
        v1 = f.clone()
        g.ndata['f'] = v1
        g.pull(nodes, message_func, reduce_func, apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace v2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.pull(nodes, fn.copy_src(src='f', out='m'),
               fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

        # inplace e2v spmv
        v1 = f.clone()
        g.ndata['f'] = v1
        g.pull(nodes,
               message_func, fn.sum(msg='m', out='f'), apply_func, inplace=True)
        r1 = g.ndata['f']
        # check result
        assert U.allclose(r1, result)
        # check inplace
        assert U.allclose(v1, r1)

    # test send_and_recv with apply_func
    _test(apply_func)
    # test send_and_recv without apply_func
    _test(None)

def test_inplace_apply():
    def apply_node_func(nodes):
        return {'f': nodes.data['f'] * 2}

    def apply_edge_func(edges):
        return {'e': edges.data['e'] * 2}

    g = generate_graph()
    nodes = [1, 2, 3, 9]
    nf = g.ndata['f']
    # out place run
    g.apply_nodes(apply_node_func, nodes)
    new_nf = g.ndata['f']
    # in place run
    g.ndata['f'] = nf
    g.apply_nodes(apply_node_func, nodes, inplace=True)
    # check results correct and in place
    assert U.allclose(nf, new_nf)
    # test apply all nodes, should not be done in place
    g.ndata['f'] = nf
    g.apply_nodes(apply_node_func, inplace=True)
    assert U.allclose(nf, g.ndata['f']) == False

    edges = [3, 5, 7, 10]
    ef = g.edata['e']
    # out place run
    g.apply_edges(apply_edge_func, edges)
    new_ef = g.edata['e']
    # in place run
    g.edata['e'] = ef
    g.apply_edges(apply_edge_func, edges, inplace=True)
    g.edata['e'] = ef
    assert U.allclose(ef, new_ef)
    # test apply all edges, should not be done in place
    g.edata['e'] == ef
    g.apply_edges(apply_edge_func, inplace=True)
    assert U.allclose(ef, g.edata['e']) == False

if __name__ == '__main__':
    test_inplace_recv()
    test_inplace_snr()
    test_inplace_push()
    test_inplace_pull()
    test_inplace_apply()
