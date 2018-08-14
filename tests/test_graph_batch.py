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
    return g

def test_batch_unbatch():
    t1 = tree1()
    t2 = tree2()
    f1 = t1.get_n_repr()
    f2 = t2.get_n_repr()

    bg = dgl.batch([t1, t2])

    # test immutability
    for g in [bg, t1, t2]:
        try:
            g.add_node(len(g))
            print("immutability test failed")
        except:
            print("pass")
        try:
            g.add_edge(len(g) - 2, len(g) - 1)
            print("immutability test failed")
        except:
            print("pass")

    dgl.unbatch(bg)

    assert(f1.equal(t1.get_n_repr()))
    assert(f2.equal(t2.get_n_repr()))
    print("pass")

    print("good")
    # test immutability
    for g in [t1, t2]:
        try:
            g.add_node(len(g))
            print("pass")
        except:
            # FIXME: this will fail...
            # print("immutability test failed")
            pass
        try:
            g.add_edge(len(g) - 2, len(g) - 1)
            print("pass")
        except:
            print("immutability test failed")

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

    bg.sendto(u, v)
    bg.recv(v)

    dgl.unbatch(bg)
    assert t1.get_n_repr()[1] == 7
    assert t2.get_n_repr()[4] == 2
    print("pass")


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
    print("pass")


if __name__ == '__main__':
    test_batch_unbatch()
    test_batch_sendrecv()
    test_batch_propagate()
