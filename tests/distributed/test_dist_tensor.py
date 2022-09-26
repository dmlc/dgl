import dgl
import os
import backend as F
import unittest
import pytest
import operator
from utils import reset_envs, generate_ip_config, create_random_graph

dist_g = None

def rand_mask(shape, dtype):
    return F.randn(shape) > 0

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
def setup_module():
    global dist_g

    reset_envs()
    os.environ['DGL_DIST_MODE'] = 'standalone'

    dist_g = create_random_graph(10000)
    # Partition the graph
    num_parts = 1
    graph_name = 'dist_graph_test_3'
    dist_g.ndata['features'] = F.unsqueeze(F.arange(0, dist_g.number_of_nodes()), 1)
    dist_g.edata['features'] = F.unsqueeze(F.arange(0, dist_g.number_of_edges()), 1)
    dgl.distributed.partition_graph(dist_g, graph_name, num_parts, '/tmp/dist_graph')

    dgl.distributed.initialize("kv_ip_config.txt")
    dist_g = dgl.distributed.DistGraph(
            graph_name, part_config='/tmp/dist_graph/{}.json'.format(graph_name))
    dist_g.edata['mask1'] = dgl.distributed.DistTensor(
            (dist_g.num_edges(),), F.bool, init_func=rand_mask)
    dist_g.edata['mask2'] = dgl.distributed.DistTensor(
            (dist_g.num_edges(),), F.bool, init_func=rand_mask)

def check_binary_op(key1, key2, key3, op):
    for i in range(0, dist_g.num_edges(), 1000):
        i_end = min(i + 1000, dist_g.num_edges())
        assert F.array_equal(
                dist_g.edata[key3][i:i_end],
                op(dist_g.edata[key1][i:i_end], dist_g.edata[key2][i:i_end]))

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
def test_op():
    dist_g.edata['mask3'] = dist_g.edata['mask1'] | dist_g.edata['mask2']
    check_binary_op('mask1', 'mask2', 'mask3', operator.or_)

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TF doesn't support some of operations in DistGraph")
@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support")
def teardown_module():
    dgl.distributed.exit_client() # this is needed since there's two test here in one process

if __name__ == '__main__':
    setup_module()
    test_op()
    teardown_module()
