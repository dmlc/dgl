import dgl
import sys
import numpy as np
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from dgl.graph_index import create_graph_index
from dgl.distributed import partition_graph, load_partition
import backend as F
import unittest
import pickle

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

def test_partition():
    g = create_random_graph(10000)
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10))
    num_parts = 4
    num_hops = 2
    partition_graph(g, 'test', num_parts, num_hops, 'metis', '/tmp')
    for i in range(num_parts):
        part_g, node_feats, edge_feats, meta = load_partition('/tmp/test.json', i)
        num_nodes, num_edges, node_map = meta

        # Check the metadata
        assert num_nodes == g.number_of_nodes()
        assert num_edges == g.number_of_edges()

        # Check the node map.
        node_map = node_map
        local_nodes = np.nonzero(node_map == i)[0]
        part_ids = node_map[F.asnumpy(part_g.ndata[dgl.NID])]
        local_nodes1 = F.asnumpy(part_g.ndata[dgl.NID])[part_ids == i]
        assert np.all(local_nodes == local_nodes1)

        for name in ['labels', 'feats']:
            assert name in node_feats
            assert node_feats[name].shape[0] == len(local_nodes)
            assert len(local_nodes) == len(node_feats[name])
            assert np.all(F.asnumpy(g.ndata[name][local_nodes]) == F.asnumpy(node_feats[name]))
        assert len(edge_feats) == 0


if __name__ == '__main__':
    test_partition()
