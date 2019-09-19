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
    g.gdata['label'] = F.tensor([0, 1, 0])
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
    g.gdata['label'] = F.tensor([0, 0, 1])
    return g


# def test_batch_unbatch():
t1 = tree1()
t2 = tree2()

bg = dgl.batch([t1, t2])
    # return bg


# test_batch_unbatch()
