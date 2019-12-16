import numpy as np
import dgl
from dgl.graph import DGLGraph
from collections import defaultdict as ddict
import scipy.sparse as sp
import backend as F

D = 5

def message_func(edges):
    assert len(edges.src['h'].shape) == 2
    assert edges.src['h'].shape[1] == D
    return {'m' : edges.src['h']}

def reduce_func(nodes):
    msgs = nodes.mailbox['m']
    assert len(msgs.shape) == 3
    assert msgs.shape[2] == D
    return {'accum' : F.sum(msgs, 1)}

def apply_node_func(nodes):
    return {'h' : nodes.data['h'] + nodes.data['accum']}

def generate_graph(grad=False):
    g = DGLGraph()
    g.add_nodes(10) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    # 16 edges
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    ncol = F.randn((10, D))
    ecol = F.randn((16, D))
    if grad:
        ncol = F.attach_grad(ncol)
        ecol = F.attach_grad(ecol)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    g.ndata['h'] = ncol
    g.edata['w'] = ecol
    return g

def test_multi_send():
    g = generate_graph()
    def _fmsg(edges):
        assert edges.src['h'].shape == (5, D)
        return {'m' : edges.src['h']}
    g.register_message_func(_fmsg)
    # many-many send
    u = F.tensor([0, 0, 0, 0, 0])
    v = F.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # duplicate send
    u = F.tensor([0])
    v = F.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # send more
    u = F.tensor([1, 2, 3, 4, 5])
    v = F.tensor([9])
    g.send((u, v))

    # check if message indicator is as expected
    expected = F.copy_to(F.zeros((g.number_of_edges(),), dtype=F.int64), F.cpu())
    eid = g.edge_ids([0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5, 9, 9, 9, 9, 9])
    expected[eid] = 1
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

def test_multi_recv():
    # basic recv test
    g = generate_graph()
    h = g.ndata['h']
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)
    expected = F.copy_to(F.zeros((g.number_of_edges(),), dtype=F.int64), F.cpu())
    # two separate round of send and recv
    u = [4, 5, 6]
    v = [9]
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)
    g.recv(v)
    expected[eid] = 0
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

    u = [0]
    v = [1, 2, 3]
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)
    g.recv(v)
    expected[eid] = 0
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

    h1 = g.ndata['h']

    # one send, two recv
    g.ndata['h'] = h
    u = F.tensor([0, 0, 0, 4, 5, 6])
    v = F.tensor([1, 2, 3, 9, 9, 9])
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)
    u = [4, 5, 6]
    v = [9]
    g.recv(v)
    eid = g.edge_ids(u, v)
    expected[eid] = 0
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)
    u = [0]
    v = [1, 2, 3]
    g.recv(v)
    eid = g.edge_ids(u, v)
    expected[eid] = 0
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

    h2 = g.ndata['h']
    assert F.allclose(h1, h2)

def test_multi_recv_0deg():
    # test recv with 0deg nodes;
    g = DGLGraph()
    def _message(edges):
        return {'m' : edges.src['h']}
    def _reduce(nodes):
        return {'h' : nodes.data['h'] + nodes.mailbox['m'].sum(1)}
    def _apply(nodes):
        return {'h' : nodes.data['h'] * 2}
    def _init2(shape, dtype, ctx, ids):
        return 2 + F.zeros(shape, dtype=dtype, ctx=ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(_init2)
    g.add_nodes(2)
    g.add_edge(0, 1)
    # recv both 0deg and non-0deg nodes
    old = F.randn((2, 5))
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv([0, 1])
    new = g.ndata['h']
    # 0deg check: initialized with the func and got applied
    assert F.allclose(new[0], F.full((5,), 4, F.float32))
    # non-0deg check
    assert F.allclose(new[1], F.sum(old, 0) * 2)

    # recv again on zero degree node
    g.recv([0])
    assert F.allclose(g.nodes[0].data['h'], F.full((5,), 8, F.float32))

    # recv again on node with no incoming message
    g.recv([1])
    assert F.allclose(g.nodes[1].data['h'], F.sum(old, 0) * 4)

def test_send_twice_different_shape():
    g = generate_graph()
    def _message_1(edges):
        return {'h': edges.src['h']}
    def _message_2(edges):
        return {'h': F.cat((edges.src['h'], edges.data['w']), dim=1)}
    g.send(message_func=_message_1)
    g.send(message_func=_message_2)

def test_send_twice_different_msg():
    g = DGLGraph()
    g.set_n_initializer(dgl.init.zero_initializer)
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(2, 1)
    def _message_a(edges):
        return {'a': edges.src['a']}
    def _message_b(edges):
        return {'a': edges.src['a'] * 3}
    def _reduce(nodes):
        return {'a': F.max(nodes.mailbox['a'], 1)}

    old_repr = F.randn((3, 5))
    g.ndata['a'] = old_repr
    g.send((0, 1), _message_a)
    g.send((0, 1), _message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], old_repr[0] * 3)

    g.ndata['a'] = old_repr
    g.send((0, 1), _message_a)
    g.send((2, 1), _message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert F.allclose(new_repr[1], F.max(F.stack([old_repr[0], old_repr[2] * 3], 0), 0))

def test_send_twice_different_field():
    g = DGLGraph()
    g.set_n_initializer(dgl.init.zero_initializer)
    g.add_nodes(2)
    g.add_edge(0, 1)
    def _message_a(edges):
        return {'a': edges.src['a']}
    def _message_b(edges):
        return {'b': edges.src['b']}
    def _reduce(nodes):
        return {'a': F.sum(nodes.mailbox['a'], 1), 'b': F.sum(nodes.mailbox['b'], 1)}
    old_a = F.randn((2, 5))
    old_b = F.randn((2, 5))
    g.set_n_repr({'a': old_a, 'b': old_b})
    g.send((0, 1), _message_a)
    g.send((0, 1), _message_b)
    g.recv([1], _reduce)
    new_repr = g.get_n_repr()
    assert F.allclose(new_repr['a'][1], old_a[0])
    assert F.allclose(new_repr['b'][1], old_b[0])

def test_dynamic_addition():
    N = 3
    D = 1

    g = DGLGraph()
    def _init(shape, dtype, ctx, ids):
        return F.copy_to(F.astype(F.randn(shape), dtype), ctx)
    g.set_n_initializer(_init)
    g.set_e_initializer(_init)

    def _message(edges):
        return {'m' : edges.src['h1'] + edges.dst['h2'] + edges.data['h1'] +
                edges.data['h2']}
    def _reduce(nodes):
        return {'h' : F.sum(nodes.mailbox['m'], 1)}
    def _apply(nodes):
        return {'h' : nodes.data['h']}

    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # add nodes and edges
    g.add_nodes(N)
    g.ndata.update({'h1': F.randn((N, D)),
                    'h2': F.randn((N, D))})
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.edata.update({'h1': F.randn((2, D)),
                    'h2': F.randn((2, D))})
    g.send()
    expected = F.copy_to(F.ones((g.number_of_edges(),), dtype=F.int64), F.cpu())
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

    # add more edges
    g.add_edges([0, 2], [2, 0], {'h1': F.randn((2, D))})
    g.send(([0, 2], [2, 0]))
    g.recv(0)

    g.add_edge(1, 2)
    g.edges[4].data['h1'] = F.randn((1, D))
    g.send((1, 2))
    g.recv([1, 2])

    h = g.ndata.pop('h')

    # a complete round of send and recv
    g.send()
    g.recv()
    assert F.allclose(h, g.ndata['h'])

def test_recv_no_send():
    g = generate_graph()
    g.recv(1, reduce_func)
    # test recv after clear
    g.clear()
    g.add_nodes(3)
    g.add_edges([0, 1], [1, 2])
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata['h'] = F.randn((3, D))
    g.send((1, 2), message_func)
    expected = F.copy_to(F.zeros(2, dtype=F.int64), F.cpu())
    expected[1] = 1
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)
    g.recv(2, reduce_func)
    expected[1] = 0
    assert F.array_equal(g._get_msg_index().tousertensor(), expected)

def test_send_recv_after_conversion():
    # test send and recv after converting from a graph with edges

    g = generate_graph()

    # nx graph
    nxg = g.to_networkx(node_attrs=['h'])
    g1 = DGLGraph()
    # some random node and edges
    g1.add_nodes(4)
    g1.add_edges([1, 2], [2, 3])
    g1.set_n_initializer(dgl.init.zero_initializer)
    g1.from_networkx(nxg, node_attrs=['h'])

    # sparse matrix
    row, col= g.all_edges()
    data = range(len(row))
    n = g.number_of_nodes()
    a = sp.coo_matrix(
            (data, (F.zerocopy_to_numpy(row), F.zerocopy_to_numpy(col))),
            shape=(n, n))
    g2 = DGLGraph()
    # some random node and edges
    g2.add_nodes(5)
    g2.add_edges([1, 2, 4], [2, 3, 0])
    g2.set_n_initializer(dgl.init.zero_initializer)
    g2.from_scipy_sparse_matrix(a)
    g2.ndata['h'] = g.ndata['h']

    # on dgl graph
    g.send(message_func=message_func)
    g.recv([0, 1, 3, 5], reduce_func=reduce_func,
           apply_node_func=apply_node_func)
    g.recv([0, 2, 4, 8], reduce_func=reduce_func,
           apply_node_func=apply_node_func)

    # nx
    g1.send(message_func=message_func)
    g1.recv([0, 1, 3, 5], reduce_func=reduce_func,
            apply_node_func=apply_node_func)
    g1.recv([0, 2, 4, 8], reduce_func=reduce_func,
            apply_node_func=apply_node_func)

    # sparse matrix
    g2.send(message_func=message_func)
    g2.recv([0, 1, 3, 5], reduce_func=reduce_func,
            apply_node_func=apply_node_func)
    g2.recv([0, 2, 4, 8], reduce_func=reduce_func,
            apply_node_func=apply_node_func)

    assert F.allclose(g.ndata['h'], g1.ndata['h'])
    assert F.allclose(g.ndata['h'], g2.ndata['h'])


if __name__ == '__main__':
    test_multi_send()
    test_multi_recv()
    test_multi_recv_0deg()
    test_dynamic_addition()
    test_send_twice_different_shape()
    test_send_twice_different_msg()
    test_send_twice_different_field()
    test_recv_no_send()
    test_send_recv_after_conversion()
