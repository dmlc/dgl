import dgl
import sys
import os
import numpy as np
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from dgl.graph_index import create_graph_index
from dgl.distributed import partition_graph, load_partition
import backend as F
import unittest
import pickle
import random

def create_random_graph(n):
    random.seed(100)
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

def check_partition(reshuffle):
    g = create_random_graph(10000)
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10))
    num_parts = 4
    num_hops = 2

    partition_graph(g, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                    part_method='metis', reshuffle=reshuffle)
    part_sizes = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb = load_partition('/tmp/partition/test.json', i)

        # Check the metadata
        assert gpb._num_nodes() == g.number_of_nodes()
        assert gpb._num_edges() == g.number_of_edges()

        assert gpb.num_partitions() == num_parts
        gpb_meta = gpb.metadata()
        assert len(gpb_meta) == num_parts
        assert len(gpb.partid2nids(i)) == gpb_meta[i]['num_nodes']
        assert len(gpb.partid2eids(i)) == gpb_meta[i]['num_edges']
        part_sizes.append((gpb_meta[i]['num_nodes'], gpb_meta[i]['num_edges']))

        local_nid = gpb.nid2localnid(F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node']), i)
        assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
        local_eid = gpb.eid2localeid(F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge']), i)
        assert np.all(F.asnumpy(local_eid) == np.arange(0, len(local_eid)))

        # Check the node map.
        local_nodes = F.asnumpy(F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node']))
        local_nodes1 = F.asnumpy(gpb.partid2nids(i))
        assert np.all(np.sort(local_nodes) == np.sort(local_nodes1))

        # Check the edge map.
        local_edges = F.asnumpy(F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge']))
        local_edges1 = F.asnumpy(gpb.partid2eids(i))
        assert np.all(np.sort(local_edges) == np.sort(local_edges1))

        for name in ['labels', 'feats']:
            assert name in node_feats
            assert node_feats[name].shape[0] == len(local_nodes)
            assert len(local_nodes) == len(node_feats[name])
            assert np.all(F.asnumpy(g.ndata[name])[local_nodes] == F.asnumpy(node_feats[name]))
        assert len(edge_feats) == 0

    if reshuffle:
        node_map = []
        edge_map = []
        for i, (num_nodes, num_edges) in enumerate(part_sizes):
            node_map.append(np.ones(num_nodes) * i)
            edge_map.append(np.ones(num_edges) * i)
        node_map = np.concatenate(node_map)
        edge_map = np.concatenate(edge_map)
        assert np.all(F.asnumpy(gpb.nid2partid(F.arange(0, len(node_map)))) == node_map)
        assert np.all(F.asnumpy(gpb.eid2partid(F.arange(0, len(edge_map)))) == edge_map)

def test_partition():
    check_partition(True)
    check_partition(False)


if __name__ == '__main__':
    os.mkdir('/tmp/partition')
    test_partition()
