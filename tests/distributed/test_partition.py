import dgl
import sys
import os
import numpy as np
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from dgl.heterograph_index import create_unitgraph_from_coo
from dgl.distributed import partition_graph, load_partition
from dgl import function as fn
import backend as F
import unittest
import pickle
import random

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo', random_state=100) != 0).astype(np.int64)
    return dgl.from_scipy(arr)

def check_partition(g, part_method, reshuffle):
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10), F.float32)
    g.edata['feats'] = F.tensor(np.random.randn(g.number_of_edges(), 10), F.float32)
    g.update_all(fn.copy_src('feats', 'msg'), fn.sum('msg', 'h'))
    g.update_all(fn.copy_edge('feats', 'msg'), fn.sum('msg', 'eh'))
    num_parts = 4
    num_hops = 2

    partition_graph(g, 'test', num_parts, '/tmp/partition', num_hops=num_hops,
                    part_method=part_method, reshuffle=reshuffle)
    part_sizes = []
    for i in range(num_parts):
        part_g, node_feats, edge_feats, gpb, _ = load_partition('/tmp/partition/test.json', i)

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
        assert F.dtype(local_nid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_nid) == np.arange(0, len(local_nid)))
        local_eid = gpb.eid2localeid(F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge']), i)
        assert F.dtype(local_eid) in (F.int64, F.int32)
        assert np.all(F.asnumpy(local_eid) == np.arange(0, len(local_eid)))

        # Check the node map.
        local_nodes = F.boolean_mask(part_g.ndata[dgl.NID], part_g.ndata['inner_node'])
        llocal_nodes = F.nonzero_1d(part_g.ndata['inner_node'])
        local_nodes1 = gpb.partid2nids(i)
        assert F.dtype(local_nodes1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_nodes)) == np.sort(F.asnumpy(local_nodes1)))

        # Check the edge map.
        local_edges = F.boolean_mask(part_g.edata[dgl.EID], part_g.edata['inner_edge'])
        local_edges1 = gpb.partid2eids(i)
        assert F.dtype(local_edges1) in (F.int32, F.int64)
        assert np.all(np.sort(F.asnumpy(local_edges)) == np.sort(F.asnumpy(local_edges1)))

        if reshuffle:
            part_g.ndata['feats'] = F.gather_row(g.ndata['feats'], part_g.ndata['orig_id'])
            part_g.edata['feats'] = F.gather_row(g.edata['feats'], part_g.edata['orig_id'])
            # when we read node data from the original global graph, we should use orig_id.
            local_nodes = F.boolean_mask(part_g.ndata['orig_id'], part_g.ndata['inner_node'])
            local_edges = F.boolean_mask(part_g.edata['orig_id'], part_g.edata['inner_edge'])
        else:
            part_g.ndata['feats'] = F.gather_row(g.ndata['feats'], part_g.ndata[dgl.NID])
            part_g.edata['feats'] = F.gather_row(g.edata['feats'], part_g.edata[dgl.NID])
        part_g.update_all(fn.copy_src('feats', 'msg'), fn.sum('msg', 'h'))
        part_g.update_all(fn.copy_edge('feats', 'msg'), fn.sum('msg', 'eh'))
        assert F.allclose(F.gather_row(g.ndata['h'], local_nodes),
                          F.gather_row(part_g.ndata['h'], llocal_nodes))
        assert F.allclose(F.gather_row(g.ndata['eh'], local_nodes),
                          F.gather_row(part_g.ndata['eh'], llocal_nodes))

        for name in ['labels', 'feats']:
            assert name in node_feats
            assert node_feats[name].shape[0] == len(local_nodes)
            assert np.all(F.asnumpy(g.ndata[name])[F.asnumpy(local_nodes)] == F.asnumpy(node_feats[name]))
        for name in ['feats']:
            assert name in edge_feats
            assert edge_feats[name].shape[0] == len(local_edges)
            assert np.all(F.asnumpy(g.edata[name])[F.asnumpy(local_edges)] == F.asnumpy(edge_feats[name]))

    if reshuffle:
        node_map = []
        edge_map = []
        for i, (num_nodes, num_edges) in enumerate(part_sizes):
            node_map.append(np.ones(num_nodes) * i)
            edge_map.append(np.ones(num_edges) * i)
        node_map = np.concatenate(node_map)
        edge_map = np.concatenate(edge_map)
        nid2pid = gpb.nid2partid(F.arange(0, len(node_map)))
        assert F.dtype(nid2pid) in (F.int32, F.int64)
        assert np.all(F.asnumpy(nid2pid) == node_map)
        eid2pid = gpb.eid2partid(F.arange(0, len(edge_map)))
        assert F.dtype(eid2pid) in (F.int32, F.int64)
        assert np.all(F.asnumpy(eid2pid) == edge_map)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_partition():
    g = create_random_graph(10000)
    check_partition(g, 'metis', True)
    check_partition(g, 'metis', False)
    check_partition(g, 'random', True)
    check_partition(g, 'random', False)

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_hetero_partition():
    g = create_random_graph(10000)
    check_partition(g, 'metis', True)
    check_partition(g, 'metis', False)
    check_partition(g, 'random', True)
    check_partition(g, 'random', False)


if __name__ == '__main__':
    os.makedirs('/tmp/partition', exist_ok=True)
    test_partition()
    test_hetero_partition()
