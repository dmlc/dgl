import numpy as np
import dgl
import backend as F
import unittest

D = 5

def generate_graph(grad=False, add_data=True):
    g = dgl.DGLGraph().to(F.ctx())
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

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU not implemented")
def test_edge_subgraph():
    # Test when the graph has no node data and edge data.
    g = generate_graph(add_data=False)
    eid = [0, 2, 3, 6, 7, 9]
    sg = g.edge_subgraph(eid)
    sg.ndata['h'] = F.arange(0, sg.number_of_nodes())
    sg.edata['h'] = F.arange(0, sg.number_of_edges())

def test_subgraph():
    g = generate_graph()
    h = g.ndata['h']
    l = g.edata['l']
    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    eid = {2, 3, 4, 5, 10, 11, 12, 13, 16}
    assert set(F.asnumpy(sg.edata[dgl.EID])) == eid
    eid = sg.edata[dgl.EID]
    # the subgraph is empty initially except for NID/EID field
    assert len(sg.ndata) == 2
    assert len(sg.edata) == 2
    sh = sg.ndata['h']
    assert F.allclose(F.gather_row(h, F.tensor(nid)), sh)
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

def _test_map_to_subgraph():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(F.arange(0, 9), F.arange(1, 10))
    h = g.subgraph([0, 1, 2, 5, 8])
    v = h.map_to_subgraph_nid([0, 8, 2])
    assert np.array_equal(F.asnumpy(v), np.array([0, 4, 2]))
