import torch as th
from torch.autograd import Variable
import numpy as np
import dgl
from dgl.graph import DGLGraph
import utils as U
from collections import defaultdict as ddict

D = 5

def message_func(edges):
    assert len(edges.src['h'].shape) == 2
    assert edges.src['h'].shape[1] == D
    return {'m' : edges.src['h']}

def reduce_func(nodes):
    msgs = nodes.mailbox['m']
    assert len(msgs.shape) == 3
    assert msgs.shape[2] == D
    return {'accum' : th.sum(msgs, 1)}

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
    # add a back flow from 9 to 0
    ncol = Variable(th.randn(10, D), requires_grad=grad)
    ecol = Variable(th.randn(16, D), requires_grad=grad)
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
    u = th.tensor([0, 0, 0, 0, 0])
    v = th.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # duplicate send
    u = th.tensor([0])
    v = th.tensor([1, 2, 3, 4, 5])
    g.send((u, v))
    # send more
    u = th.tensor([1, 2, 3, 4, 5])
    v = th.tensor([9])
    g.send((u, v))

    # check if message indicator is as expected
    expected = th.zeros((g.number_of_edges(),), dtype=th.int64)
    eid = g.edge_ids([0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5, 9, 9, 9, 9, 9])
    expected[eid] = 1
    assert th.equal(g._msg_index.tousertensor(), expected)

def test_multi_recv():
    # basic recv test
    g = generate_graph()
    h = g.ndata['h']
    g.register_message_func(message_func)
    g.register_reduce_func(reduce_func)
    g.register_apply_node_func(apply_node_func)
    expected = th.zeros((g.number_of_edges(),), dtype=th.int64)
    # two separate round of send and recv
    u = [4, 5, 6]
    v = [9]
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert th.equal(g._msg_index.tousertensor(), expected)
    g.recv(v)
    expected[eid] = 0
    assert th.equal(g._msg_index.tousertensor(), expected)

    u = [0]
    v = [1, 2, 3]
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert th.equal(g._msg_index.tousertensor(), expected)
    g.recv(v)
    expected[eid] = 0
    assert th.equal(g._msg_index.tousertensor(), expected)

    h1 = g.ndata['h']

    # one send, two recv
    g.ndata['h'] = h
    u = th.tensor([0, 0, 0, 4, 5, 6])
    v = th.tensor([1, 2, 3, 9, 9, 9])
    g.send((u, v))
    eid = g.edge_ids(u, v)
    expected[eid] = 1
    assert th.equal(g._msg_index.tousertensor(), expected)
    u = [4, 5, 6]
    v = [9]
    g.recv(v)
    eid = g.edge_ids(u, v)
    expected[eid] = 0
    assert th.equal(g._msg_index.tousertensor(), expected)
    u = [0]
    v = [1, 2, 3]
    g.recv(v)
    eid = g.edge_ids(u, v)
    expected[eid] = 0
    assert th.equal(g._msg_index.tousertensor(), expected)

    h2 = g.ndata['h']
    assert U.allclose(h1, h2)

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
        return 2 + th.zeros(shape, dtype=dtype, device=ctx)
    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(_init2)
    g.add_nodes(2)
    g.add_edge(0, 1)
    # recv both 0deg and non-0deg nodes
    old = th.randn((2, 5))
    g.ndata['h'] = old
    g.send((0, 1))
    g.recv([0, 1])
    new = g.ndata['h']
    # 0deg check: initialized with the func and got applied
    assert U.allclose(new[0], th.full((5,), 4))
    # non-0deg check
    assert U.allclose(new[1], th.sum(old, 0) * 2)

    # recv again on zero degree node
    g.recv([0])
    assert U.allclose(g.nodes[0].data['h'], th.full((5,), 8))

    # recv again on node with no incoming message
    g.recv([1])
    assert U.allclose(g.nodes[1].data['h'], th.sum(old, 0) * 4)

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
        return {'a': nodes.mailbox['a'].max(1)[0]}

    old_repr = th.randn(3, 5)
    g.ndata['a'] = old_repr
    g.send((0, 1), _message_a)
    g.send((0, 1), _message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert U.allclose(new_repr[1], old_repr[0] * 3)

    g.ndata['a'] = old_repr
    g.send((0, 1), _message_a)
    g.send((2, 1), _message_b)
    g.recv(1, _reduce)
    new_repr = g.ndata['a']
    assert U.allclose(new_repr[1], th.stack([old_repr[0], old_repr[2] * 3], 0).max(0)[0])

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
        return {'a': nodes.mailbox['a'].sum(1), 'b': nodes.mailbox['b'].sum(1)}
    old_a = th.randn(2, 5)
    old_b = th.randn(2, 5)
    g.set_n_repr({'a': old_a, 'b': old_b})
    g.send((0, 1), _message_a)
    g.send((0, 1), _message_b)
    g.recv([1], _reduce)
    new_repr = g.get_n_repr()
    assert th.allclose(new_repr['a'][1], old_a[0])
    assert th.allclose(new_repr['b'][1], old_b[0])

def test_dynamic_addition():
    N = 3
    D = 1

    g = DGLGraph()
    def _init(shape, dtype, ctx, ids):
        return th.randn(shape, dtype=dtype, device=ctx)
    g.set_n_initializer(_init)
    g.set_e_initializer(_init)

    def _message(edges):
        return {'m' : edges.src['h1'] + edges.dst['h2'] + edges.data['h1'] +
                edges.data['h2']}
    def _reduce(nodes):
        return {'h' : nodes.mailbox['m'].sum(1)}
    def _apply(nodes):
        return {'h' : nodes.data['h']}

    g.register_message_func(_message)
    g.register_reduce_func(_reduce)
    g.register_apply_node_func(_apply)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    # add nodes and edges
    g.add_nodes(N)
    g.ndata.update({'h1': th.randn(N, D),
                    'h2': th.randn(N, D)})
    g.add_nodes(3)
    g.add_edge(0, 1)
    g.add_edge(1, 0)
    g.edata.update({'h1': th.randn(2, D),
                    'h2': th.randn(2, D)})
    g.send()
    expected = th.ones((g.number_of_edges(),), dtype=th.int64)
    assert th.equal(g._msg_index.tousertensor(), expected)

    # add more edges
    g.add_edges([0, 2], [2, 0], {'h1': th.randn(2, D)})
    g.send(([0, 2], [2, 0]))
    g.recv(0)

    g.add_edge(1, 2)
    g.edges[4].data['h1'] = th.randn(1, D)
    g.send((1, 2))
    g.recv([1, 2])

    h = g.ndata.pop('h')

    # a complete round of send and recv
    g.send()
    g.recv()
    assert U.allclose(h, g.ndata['h'])

def test_recv_no_send():
    g = generate_graph()
    g.recv(1, reduce_func)
    # test recv after clear
    g.clear()
    g.add_nodes(3)
    g.add_edges([0, 1], [1, 2])
    g.set_n_initializer(dgl.init.zero_initializer)
    g.ndata['h'] = th.randn(3, D)
    g.send((1, 2), message_func)
    expected = th.zeros((2,), dtype=th.int64)
    expected[1] = 1
    assert th.equal(g._msg_index.tousertensor(), expected)
    g.recv(2, reduce_func)
    expected[1] = 0
    assert th.equal(g._msg_index.tousertensor(), expected)


if __name__ == '__main__':
    test_multi_send()
    test_multi_recv()
    test_multi_recv_0deg()
    test_dynamic_addition()
    test_send_twice_different_msg()
    test_send_twice_different_field()
    test_recv_no_send()
