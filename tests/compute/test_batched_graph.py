import dgl
from dgl import DGLGraph
import backend as F

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
    g.ndata['h'] = F.tensor([0, 1, 2, 3, 4])
    g.edata['h'] = F.randn((4, 10))
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
    g.ndata['h'] = F.tensor([0, 1, 2, 3, 4])
    g.edata['h'] = F.randn((4, 10))
    return g

def test_batch_unbatch():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    assert bg.number_of_nodes() == 10
    assert bg.number_of_edges() == 8
    assert bg.batch_size == 2
    assert bg.batch_num_nodes == [5, 5]
    assert bg.batch_num_edges == [4, 4]

    tt1, tt2 = dgl.unbatch(bg)
    assert F.allclose(t1.ndata['h'], tt1.ndata['h'])
    assert F.allclose(t1.edata['h'], tt1.edata['h'])
    assert F.allclose(t2.ndata['h'], tt2.ndata['h'])
    assert F.allclose(t2.edata['h'], tt2.edata['h'])

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
    assert F.allclose(t2.ndata['h'], s1.ndata['h'])
    assert F.allclose(t2.edata['h'], s1.edata['h'])
    assert F.allclose(t1.ndata['h'], s2.ndata['h'])
    assert F.allclose(t1.edata['h'], s2.edata['h'])
    assert F.allclose(t2.ndata['h'], s3.ndata['h'])
    assert F.allclose(t2.edata['h'], s3.edata['h'])

    # Test batching readonly graphs
    t1.readonly()
    t2.readonly()
    t1_was_readonly = t1.is_readonly
    t2_was_readonly = t2.is_readonly
    bg = dgl.batch([t1, t2])

    assert t1.is_readonly == t1_was_readonly
    assert t2.is_readonly == t2_was_readonly
    assert bg.number_of_nodes() == 10
    assert bg.number_of_edges() == 8
    assert bg.batch_size == 2
    assert bg.batch_num_nodes == [5, 5]
    assert bg.batch_num_edges == [4, 4]

    rs1, rs2 = dgl.unbatch(bg)
    assert F.allclose(rs1.edges()[0], t1.edges()[0])
    assert F.allclose(rs1.edges()[1], t1.edges()[1])
    assert F.allclose(rs2.edges()[0], t2.edges()[0])
    assert F.allclose(rs2.edges()[1], t2.edges()[1])
    assert F.allclose(rs1.nodes(), t1.nodes())
    assert F.allclose(rs2.nodes(), t2.nodes())
    assert F.allclose(t1.ndata['h'], rs1.ndata['h'])
    assert F.allclose(t1.edata['h'], rs1.edata['h'])
    assert F.allclose(t2.ndata['h'], rs2.ndata['h'])
    assert F.allclose(t2.edata['h'], rs2.edata['h'])

def test_batch_unbatch2():
    # test setting/getting features after batch
    a = dgl.DGLGraph()
    a.add_nodes(4)
    a.add_edges(0, [1, 2, 3])
    b = dgl.DGLGraph()
    b.add_nodes(3)
    b.add_edges(0, [1, 2])
    c = dgl.batch([a, b])
    c.ndata['h'] = F.ones((7, 1))
    c.edata['w'] = F.ones((5, 1))
    assert F.allclose(c.ndata['h'], F.ones((7, 1)))
    assert F.allclose(c.edata['w'], F.ones((5, 1)))

def test_batch_send_then_recv():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda edges: {'m' : edges.src['h']})
    bg.register_reduce_func(lambda nodes: {'h' : F.sum(nodes.mailbox['m'], 1)})
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]

    bg.send((u, v))
    bg.recv([1, 9]) # assuming recv takes in unique nodes

    t1, t2 = dgl.unbatch(bg)
    assert t1.ndata['h'][1] == 7
    assert t2.ndata['h'][4] == 2

def test_batch_send_and_recv():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda edges: {'m' : edges.src['h']})
    bg.register_reduce_func(lambda nodes: {'h' : F.sum(nodes.mailbox['m'], 1)})
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]

    bg.send_and_recv((u, v))

    t1, t2 = dgl.unbatch(bg)
    assert t1.ndata['h'][1] == 7
    assert t2.ndata['h'][4] == 2

def test_batch_propagate():
    t1 = tree1()
    t2 = tree2()

    bg = dgl.batch([t1, t2])
    bg.register_message_func(lambda edges: {'m' : edges.src['h']})
    bg.register_reduce_func(lambda nodes: {'h' : F.sum(nodes.mailbox['m'], 1)})
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

    bg.prop_edges(order)
    t1, t2 = dgl.unbatch(bg)

    assert t1.ndata['h'][0] == 9
    assert t2.ndata['h'][1] == 5

def test_batched_edge_ordering():
    g1 = dgl.DGLGraph()
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    e1 = F.randn((5, 10))
    g1.edata['h'] = e1
    g2 = dgl.DGLGraph()
    g2.add_nodes(6)
    g2.add_edges([0, 1 ,2 ,5, 4 ,5], [1, 2, 3, 4, 3, 0])
    e2 = F.randn((6, 10))
    g2.edata['h'] = e2
    g = dgl.batch([g1, g2])
    r1 = g.edata['h'][g.edge_id(4, 5)]
    r2 = g1.edata['h'][g1.edge_id(4, 5)]
    assert F.array_equal(r1, r2)

def test_batch_no_edge():
    g1 = dgl.DGLGraph()
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    g2 = dgl.DGLGraph()
    g2.add_nodes(6)
    g2.add_edges([0, 1, 2, 5, 4, 5], [1 ,2 ,3, 4, 3, 0])
    g3 = dgl.DGLGraph()
    g3.add_nodes(1)  # no edges
    g = dgl.batch([g1, g3, g2]) # should not throw an error

if __name__ == '__main__':
    test_batch_unbatch()
    test_batch_unbatch1()
    #test_batch_unbatch2()
    #test_batched_edge_ordering()
    #test_batch_send_then_recv()
    #test_batch_send_and_recv()
    #test_batch_propagate()
    #test_batch_no_edge()
