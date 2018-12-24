import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
from dgl.graph import DGLGraph
import dgl

D = 5

def generate_graph():
    g = DGLGraph()
    g.add_nodes(10) # 10 nodes.
    # create a graph where 0 is the source and 9 is the sink
    for i in range(1, 9):
        g.add_edge(0, i)
        g.add_edge(i, 9)
    # add a back flow from 9 to 0
    g.add_edge(9, 0)
    g.ndata['h1'] = mx.nd.arange(10 * D).reshape((10, D))
    g.ndata['h2'] = mx.nd.arange(10 * D).reshape((10, D))
    g.edata['w'] = mx.nd.arange(17 * D).reshape((17, D))
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g

def test_node_cache():
    g = generate_graph()
    g._cache_node_data([1, 0, 5, 9, 7, 8], mx.cpu())
    subg_ids = [0, 1, 2, 4, 5, 9]
    subg = g.subgraph(subg_ids)
    subg.copy_from_parent()
    assert subg.ndata.keys() == g.ndata.keys()
    for key in subg.ndata:
        assert key in g.ndata
        data = g.ndata[key][subg_ids]
        assert np.all(data.asnumpy() == subg.ndata[key].asnumpy())

    subg_ids = [[0, 1, 2, 4, 5, 9], [0, 2, 3, 7], [3, 4, 5]]
    for i in range(2, 4):
        subgs = g.subgraphs(subg_ids[:i])
        for subg, subg_id in zip(subgs, subg_ids):
            subg.copy_from_parent()
            assert subg.ndata.keys() == g.ndata.keys()
            for key in subg.ndata:
                assert key in g.ndata
                data = g.ndata[key][subg_id]
                assert np.all(data.asnumpy() == subg.ndata[key].asnumpy())

if __name__ == '__main__':
    test_node_cache()
