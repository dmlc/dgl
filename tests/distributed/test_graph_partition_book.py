import dgl
import sys
import numpy as np
from scipy import sparse as spsp
from numpy.testing import assert_array_equal
from dgl.graph_index import create_graph_index
from dgl.distributed import partition_graph, load_partition, GraphPartitionBook
import backend as F
import unittest
import pickle

def create_ip_config():
    ip_config = open("ip_config.txt", "w")
    ip_config.write('192.168.9.12 30050 0\n')
    ip_config.write('192.168.9.13 30050 1\n')
    ip_config.write('192.168.9.14 30050 2\n')
    ip_config.write('192.168.9.15 30050 3\n')
    ip_config.close()

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

@unittest.skipIf(F._default_context_str == 'gpu', reason="METIS doesn't support GPU")
def test_graph_partition_book():
    g = create_random_graph(10000)
    g.ndata['labels'] = F.arange(0, g.number_of_nodes())
    g.ndata['feats'] = F.tensor(np.random.randn(g.number_of_nodes(), 10))
    num_parts = 4
    num_hops = 2

    create_ip_config()
    partition_graph(g, 'test', num_parts, '/tmp', num_hops=num_hops, part_method='metis')

    for i in range(num_parts):
        part_g, node_feats, edge_feats, meta = load_partition('/tmp/test.json', i)
        num_nodes, num_edges, node_map, edge_map, num_partitions = meta
        gpb = GraphPartitionBook(part_id=i,
                                 num_parts=num_partitions,
                                 node_map=node_map,
                                 edge_map=edge_map,
                                 part_graph=part_g)
        assert gpb.num_partitions() == num_parts
        gpb_meta = gpb.metadata()
        assert len(gpb_meta) == num_parts
        assert np.all(F.asnumpy(gpb.nid2partid(F.arange(0, len(node_map)))) == node_map)
        assert np.all(F.asnumpy(gpb.eid2partid(F.arange(0, len(edge_map)))) == edge_map)
        assert len(gpb.partid2nids(i)) == gpb_meta[i]['num_nodes']
        assert len(gpb.partid2eids(i)) == gpb_meta[i]['num_edges']
        local_nid = gpb.nid2localnid(part_g.ndata[dgl.NID], i)
        assert np.all(F.asnumpy(local_nid) == F.asnumpy(F.arange(0, len(local_nid))))
        local_eid = gpb.eid2localeid(part_g.edata[dgl.EID], i)
        assert np.all(F.asnumpy(local_eid) == F.asnumpy(F.arange(0, len(local_eid))))


if __name__ == '__main__':
    test_graph_partition_book()