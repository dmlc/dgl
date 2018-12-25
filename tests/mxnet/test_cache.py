import os
os.environ['DGLBACKEND'] = 'mxnet'
import mxnet as mx
import numpy as np
from dgl.graph import DGLGraph
from dgl import utils
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

def subgraph(g, nodes, vertex_cache):
    induced_nodes = utils.toindex(nodes)
    sgi = g._graph.node_subgraph(induced_nodes)
    cache = vertex_cache.cache_lookup([sgi.induced_nodes])[0]
    return dgl.DGLSubGraph(g, sgi.induced_nodes, sgi.induced_edges, sgi,
                           vertex_cache=cache)

def subgraphs(g, nodes, vertex_cache):
    induced_nodes = [utils.toindex(n) for n in nodes]
    sgis = g._graph.node_subgraphs(induced_nodes)
    sg_nodes = [i.induced_nodes for i in sgis]
    caches = vertex_cache.cache_lookup(sg_nodes)
    return [dgl.DGLSubGraph(g, sgi.induced_nodes, sgi.induced_edges,
                            sgi, vertex_cache=cache) for sgi, cache in zip(sgis, caches)]

def test_node_cache():
    g = generate_graph()
    vids = mx.nd.array([1, 0, 5, 9, 7, 8], dtype=np.int64)
    vertex_cache = dgl.frame_cache.FrameRowCache(g._node_frame, utils.toindex(vids), mx.cpu())

    subg_ids = [0, 1, 2, 4, 5, 9]
    subg = subgraph(g, subg_ids, vertex_cache)
    subg.copy_from_parent()
    assert subg.ndata.keys() == g.ndata.keys()
    for key in subg.ndata:
        assert key in g.ndata
        data = g.ndata[key][subg_ids]
        assert np.all(data.asnumpy() == subg.ndata[key].asnumpy())

    subg_ids = [[0, 1, 2, 4, 5, 9], [0, 2, 3, 7], [3, 4, 5]]
    for i in range(2, 4):
        subgs = subgraphs(g, subg_ids[:i], vertex_cache)
        for subg, subg_id in zip(subgs, subg_ids):
            subg.copy_from_parent()
            assert subg.ndata.keys() == g.ndata.keys()
            for key in subg.ndata:
                assert key in g.ndata
                data = g.ndata[key][subg_id]
                assert np.all(data.asnumpy() == subg.ndata[key].asnumpy())

if __name__ == '__main__':
    test_node_cache()
