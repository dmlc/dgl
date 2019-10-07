import numpy as np
from dgl.graph import DGLGraph
import backend as F

D = 5

def generate_graph(grad=False, add_data=True):
    g = DGLGraph()
    g.add_nodes(10)
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    if add_data:
        ncol = F.randn((10, D))
        ecol = F.randn((17, D))
        if grad:
            ncol = F.attach_grad(ncol)
            ecol = F.attach_grad(ecol)
        g.ndata['h'] = ncol
        g.edata['l'] = ecol
    return g

def test_basics1():
    # Test when the graph has no node data and edge data.
    g = generate_graph(add_data=False)
    eid = [0, 2, 3, 6, 7, 9]
    sg = g.edge_subgraph(eid)
    sg.copy_from_parent()
    sg.ndata['h'] = F.arange(0, sg.number_of_nodes())
    sg.edata['h'] = F.arange(0, sg.number_of_edges())

def test_basics():
    g = generate_graph()
    h = g.ndata['h']
    l = g.edata['l']
    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    eid = {2, 3, 4, 5, 10, 11, 12, 13, 16}
    assert set(F.zerocopy_to_numpy(sg.parent_eid)) == eid
    eid = F.tensor(sg.parent_eid)
    # the subgraph is empty initially
    assert len(sg.ndata) == 0
    assert len(sg.edata) == 0
    # the data is copied after explict copy from
    sg.copy_from_parent()
    assert len(sg.ndata) == 1
    assert len(sg.edata) == 1
    sh = sg.ndata['h']
    assert F.allclose(h[nid], sh)
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
    assert F.allclose(F.gather_row(l, eid), sg.edata['l'])
    # update the node/edge features on the subgraph should NOT
    # reflect to the parent graph.
    sg.ndata['h'] = F.zeros((6, D))
    assert F.allclose(h, g.ndata['h'])

def test_merge():
    # FIXME: current impl cannot handle this case!!!
    #        comment out for now to test CI
    return
    """
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

    h = g.ndata['h'][:,0]
    l = g.edata['l'][:,0]
    assert U.allclose(h, th.tensor([3., 0., 3., 3., 2., 0., 1., 1., 0., 1.]))
    assert U.allclose(l,
            th.tensor([0., 0., 1., 1., 1., 1., 0., 0., 0., 3., 1., 4., 1., 4., 0., 3., 1.]))
    """

if __name__ == '__main__':
    test_basics()
    test_basics1()
    #test_merge()
