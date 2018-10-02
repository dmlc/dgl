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
    sg.copy_from_parent()
    assert len(sg.get_n_repr()) == 1
    assert len(sg.get_e_repr()) == 1
    sh = sg.get_n_repr()['h']
    assert check_eq(h[nid], sh)
    '''
    s, d, eid
    0, 1, 0
    1, 9, 1
    0, 2, 2    1
    2, 9, 3    1
    0, 3, 4    1
    3, 9, 5    1
    0, 4, 6
    4, 9, 7
    0, 5, 8
    5, 9, 9       3
    0, 6, 10   1
    6, 9, 11   1  3
    0, 7, 12   1
    7, 9, 13   1  3
    0, 8, 14
    8, 9, 15      3
    9, 0, 16   1
    '''
    assert check_eq(l[eid], sg.get_e_repr()['l'])
    # update the node/edge features on the subgraph should NOT
    # reflect to the parent graph.
    sg.set_n_repr({'h' : th.zeros((6, D))})
    assert check_eq(h, g.get_n_repr()['h'])

def test_merge():
    g = generate_graph()
    g.set_n_repr({'h' : th.zeros((10, D))})
    g.set_e_repr({'l' : th.zeros((17, D))})
    # subgraphs
    sg1 = g.subgraph([0, 2, 3, 6, 7, 9])
    sg1.set_n_repr({'h' : th.ones((6, D))})
    sg1.set_e_repr({'l' : th.ones((9, D))})

    sg2 = g.subgraph([0, 2, 3, 4])
    sg2.set_n_repr({'h' : th.ones((4, D)) * 2})

    sg3 = g.subgraph([5, 6, 7, 8, 9])
    sg3.set_e_repr({'l' : th.ones((4, D)) * 3})

    g.merge([sg1, sg2, sg3])

    h = g.get_n_repr()['h'][:,0]
    l = g.get_e_repr()['l'][:,0]
    assert check_eq(h, th.tensor([3., 0., 3., 3., 2., 0., 1., 1., 0., 1.]))
    assert check_eq(l,
            th.tensor([0., 0., 1., 1., 1., 1., 0., 0., 0., 3., 1., 4., 1., 4., 0., 3., 1.]))

if __name__ == '__main__':
    test_basics()
    test_merge()
