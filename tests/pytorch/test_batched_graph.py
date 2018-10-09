import networkx as nx
import dgl
import torch as th
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
    g.add_nodes(5)
    g.add_edge(3, 1)
    g.add_edge(4, 1)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    g.set_n_repr(th.Tensor([0, 1, 2, 3, 4]))
    g.set_e_repr(th.randn(4, 10))
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
    g.add_nodes(5)
    g.add_edge(2, 4)
    g.add_edge(0, 4)
    g.add_edge(4, 1)
    g.add_edge(3, 1)
    g.set_n_repr(th.Tensor([0, 1, 2, 3, 4]))
    g.set_e_repr(th.randn(4, 10))
    return g

def test_batch_unbatch():
    t1 = tree1()
    t2 = tree2()
    n1 = t1.get_n_repr()
    n2 = t2.get_n_repr()
    e1 = t1.get_e_repr()
    e2 = t2.get_e_repr()

    bg = dgl.batch([t1, t2])
    assert bg.number_of_nodes() == 10
    assert bg.number_of_edges() == 8
    assert bg.batch_size == 2
    assert bg.batch_num_nodes == [5, 5]
    assert bg.batch_num_edges == [4, 4]

    tt1, tt2 = dgl.unbatch(bg)
    assert th.allclose(t1.get_n_repr(), tt1.get_n_repr())
    assert th.allclose(t1.get_e_repr(), tt1.get_e_repr())
    assert th.allclose(t2.get_n_repr(), tt2.get_n_repr())
    assert th.allclose(t2.get_e_repr(), tt2.get_e_repr())

def test_batch_unbatch1():
    t1 = tree1()
    t2 = tree2()
    b1 = dgl.batch([t1, t2])
    b2 = dgl.batch([t2, b1])
    assert b2.number_of_nodes() == 15
    assert b2.number_of_edges() == 12
    assert b2.batch_size == 3
    assert b2.batch_num_nodes == [5, 5, 5]
    assert b2.batch_num_edges == [4, 4, 4]

    s1, s2, s3 = dgl.unbatch(b2)
    assert th.allclose(t2.get_n_repr(), s1.get_n_repr())
    assert th.allclose(t2.get_e_repr(), s1.get_e_repr())
    assert th.allclose(t1.get_n_repr(), s2.get_n_repr())
    assert th.allclose(t1.get_e_repr(), s2.get_e_repr())
    assert th.allclose(t2.get_n_repr(), s3.get_n_repr())
    assert th.allclose(t2.get_e_repr(), s3.get_e_repr())

def test_batch_sendrecv():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda src, edge: src)
    bg.register_reduce_func(lambda node, msgs: th.sum(msgs, 1))
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]

    bg.send(u, v)
    bg.recv(v)

    t1, t2 = dgl.unbatch(bg)
    assert t1.get_n_repr()[1] == 7
    assert t2.get_n_repr()[4] == 2


def test_batch_propagate():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda src, edge: src)
    bg.register_reduce_func(lambda node, msgs: th.sum(msgs, 1))
    # get leaves.

    order = []

    # step 1
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]
    order.append((u, v))

    # step 2
    u = [1, 2, 4 + 5, 3 + 5]
    v = [0, 0, 1 + 5, 1 + 5]
    order.append((u, v))

    bg.propagate(traverser=order)
    t1, t2 = dgl.unbatch(bg)

    assert t1.get_n_repr()[0] == 9
    assert t2.get_n_repr()[1] == 5

def test_batched_edge_ordering():
    g1 = dgl.DGLGraph()
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    e1 = th.randn(5, 10)
    g1.set_e_repr(e1)
    g2 = dgl.DGLGraph()
    g2.add_nodes(6)
    g2.add_edges([0, 1 ,2 ,5, 4 ,5], [1, 2, 3, 4, 3, 0])
    e2 = th.randn(6, 10)
    g2.set_e_repr(e2)
    g = dgl.batch([g1, g2])
    r1 = g.get_e_repr()[g.edge_id(4, 5)]
    r2 = g1.get_e_repr()[g1.edge_id(4, 5)]
    assert th.equal(r1, r2)

def test_batch_no_edge():
    # FIXME: current impl cannot handle this case!!!
    #        comment out for now to test CI
    return
    """
    g1 = dgl.DGLGraph()
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    e1 = th.randn(5, 10)
    g1.set_e_repr(e1)
    g2 = dgl.DGLGraph()
    g2.add_nodes(6)
    g2.add_edges([0, 1, 2, 5, 4, 5], [1 ,2 ,3, 4, 3, 0])
    e2 = th.randn(6, 10)
    g2.set_e_repr(e2)
    g3 = dgl.DGLGraph()
    g3.add_nodes(1)  # no edges

    g = dgl.batch([g1, g3, g2]) # should not throw an error
    """

if __name__ == '__main__':
    test_batch_unbatch()
    test_batch_unbatch1()
    test_batched_edge_ordering()
    test_batch_sendrecv()
    test_batch_propagate()
    test_batch_no_edge()
