import networkx as nx
import dgl
import torch
import numpy as np

def tree1():
    """Generate a tree
         0
        / \
       1   2
      / \
     3   4
    Edges are from leaves to root.
    """
    g = dgl.DGLGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_edge(3, 1)
    g.add_edge(4, 1)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    g.set_n_repr(torch.Tensor([0, 1, 2, 3, 4]))
    g.set_e_repr(torch.randn(4, 10))
    return g

def tree2():
    """Generate a tree
         1
        / \
       4   3
      / \
     2   0
    Edges are from leaves to root.
    """
    g = dgl.DGLGraph()
    g.add_node(0)
    g.add_node(1)
    g.add_node(2)
    g.add_node(3)
    g.add_node(4)
    g.add_edge(2, 4)
    g.add_edge(0, 4)
    g.add_edge(4, 1)
    g.add_edge(3, 1)
    g.set_n_repr(torch.Tensor([0, 1, 2, 3, 4]))
    g.set_e_repr(torch.randn(4, 10))
    return g

def test_batch_unbatch():
    t1 = tree1()
    t2 = tree2()
    n1 = t1.get_n_repr()
    n2 = t2.get_n_repr()
    e1 = t1.get_e_repr()
    e2 = t2.get_e_repr()

    bg = dgl.batch([t1, t2])
    dgl.unbatch(bg)

    assert(n1.equal(t1.get_n_repr()))
    assert(n2.equal(t2.get_n_repr()))
    assert(e1.equal(t1.get_e_repr()))
    assert(e2.equal(t2.get_e_repr()))


def test_batch_sendrecv():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda src, edge: src, batchable=True)
    bg.register_reduce_func(lambda node, msgs: torch.sum(msgs, 1), batchable=True)
    bg.register_update_func(lambda node, accum: accum, batchable=True)
    e1 = [(3, 1), (4, 1)]
    e2 = [(2, 4), (0, 4)]

    u1, v1 = bg.query_new_edge(t1, *zip(*e1))
    u2, v2 = bg.query_new_edge(t2, *zip(*e2))
    u = np.concatenate((u1, u2)).tolist()
    v = np.concatenate((v1, v2)).tolist()

    bg.send(u, v)
    bg.recv(v)

    dgl.unbatch(bg)
    assert t1.get_n_repr()[1] == 7
    assert t2.get_n_repr()[4] == 2


def test_batch_propagate():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda src, edge: src, batchable=True)
    bg.register_reduce_func(lambda node, msgs: torch.sum(msgs, 1), batchable=True)
    bg.register_update_func(lambda node, accum: accum, batchable=True)
    # get leaves.

    order = []

    # step 1
    e1 = [(3, 1), (4, 1)]
    e2 = [(2, 4), (0, 4)]
    u1, v1 = bg.query_new_edge(t1, *zip(*e1))
    u2, v2 = bg.query_new_edge(t2, *zip(*e2))
    u = np.concatenate((u1, u2)).tolist()
    v = np.concatenate((v1, v2)).tolist()
    order.append((u, v))

    # step 2
    e1 = [(1, 0), (2, 0)]
    e2 = [(4, 1), (3, 1)]
    u1, v1 = bg.query_new_edge(t1, *zip(*e1))
    u2, v2 = bg.query_new_edge(t2, *zip(*e2))
    u = np.concatenate((u1, u2)).tolist()
    v = np.concatenate((v1, v2)).tolist()
    order.append((u, v))

    bg.propagate(iterator=order)
    dgl.unbatch(bg)

    assert t1.get_n_repr()[0] == 9
    assert t2.get_n_repr()[1] == 5

def test_batched_edge_ordering():
    g1 = dgl.DGLGraph()
    g1.add_nodes_from([0,1,2, 3, 4, 5])
    g1.add_edges_from([(4, 5), (4, 3), (2, 3), (2, 1), (0, 1)])
    g1.edge_list
    e1 = torch.randn(5, 10)
    g1.set_e_repr(e1)
    g2 = dgl.DGLGraph()
    g2.add_nodes_from([0, 1, 2, 3, 4, 5])
    g2.add_edges_from([(0, 1), (1, 2), (2, 3), (5, 4), (4, 3), (5, 0)])
    e2 = torch.randn(6, 10)
    g2.set_e_repr(e2)
    g = dgl.batch([g1, g2])
    r1 = g.get_e_repr()[g.get_edge_id(4, 5)]
    r2 = g1.get_e_repr()[g1.get_edge_id(4, 5)]
    assert torch.equal(r1, r2)

if __name__ == '__main__':
    test_batch_unbatch()
    test_batched_edge_ordering()
    test_batch_sendrecv()
    test_batch_propagate()
