import torch as th
from torch.autograd import Variable
import numpy as np
from dgl.graph import DGLGraph

D = 5

def check_eq(a, b):
    return a.shape == b.shape and np.allclose(a.numpy(), b.numpy())

def generate_graph(grad=False):
    g = DGLGraph()
    for i in range(10):
        g.add_node(i) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    ncol = Variable(th.randn(10, D), requires_grad=grad)
    ecol = Variable(th.randn(17, D), requires_grad=grad)
    g.set_n_repr({'h' : ncol})
    g.set_e_repr({'l' : ecol})
    return g

def test_basics():
    g = generate_graph()
    h = g.get_n_repr()['h']
    l = g.get_e_repr()['l']
    nid = [0, 2, 3, 6, 7, 9]
    eid = [2, 3, 4, 5, 10, 11, 12, 13, 16]
    sg = g.subgraph(nid)
    # the subgraph is empty initially
    assert len(sg.get_n_repr()) == 0
    assert len(sg.get_e_repr()) == 0
    # the data is copied after explict copy from
    sg.copy_from(g)
    assert len(sg.get_n_repr()) == 1
    assert len(sg.get_e_repr()) == 1
    sh = sg.get_n_repr()['h']
    assert check_eq(h[nid], sh)
    '''
    s, d, eid
    0, 1, 0
    1, 9, 1
    0, 2, 2
    2, 9, 3
    0, 3, 4
    3, 9, 5
    0, 4, 6
    4, 9, 7
    0, 5, 8
    5, 9, 9
    0, 6, 10
    6, 9, 11
    0, 7, 12
    7, 9, 13
    0, 8, 14
    8, 9, 15
    9, 0, 16
    '''
    assert check_eq(l[eid], sg.get_e_repr()['l'])
    # update the node/edge features on the subgraph should NOT
    # reflect to the parent graph.
    sg.set_n_repr({'h' : th.zeros((6, D))})
    assert check_eq(h, g.get_n_repr()['h'])

if __name__ == '__main__':
    test_basics()
