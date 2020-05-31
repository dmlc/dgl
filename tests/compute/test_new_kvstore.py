import os
import time

import dgl
import backend as F
import unittest, pytest

from numpy.testing import assert_array_equal

# Create an one-part Graph
node_map = F.tensor([0,0,0,0,0,0], F.int64)
edge_map = F.tensor([0,0,0,0,0,0], F.int64)
global_nid = F.tensor([0,1,2,3,4,5], F.int64)
global_eid = F.tensor([0,1,2,3,4,5], F.int64)

gpb = dgl.distributed.GraphPartitionBook(part_id=0,
                                         num_parts=1,
                                         node_map=node_map,
                                         edge_map=edge_map,
                                         global_nid=global_nid,
                                         global_eid=global_eid)

node_policy = dgl.distributed.PartitionPolicy(policy_str='node', 
                                              part_id=0, 
                                              partition_book=gpb)

edge_policy = dgl.distributed.PartitionPolicy(policy_str='edge', 
                                              part_id=0, 
                                              partition_book=gpb)

def test_partition_policy():
    assert node_policy.policy_str == 'node'
    assert edge_policy.policy_str == 'edge'
    assert node_policy.part_id == 0
    assert edge_policy.part_id == 0
    local_nid = node_policy.to_local(F.tensor([0,1,2,3,4,5]))
    local_eid = edge_policy.to_local(F.tensor([0,1,2,3,4,5]))
    assert_array_equal(F.asnumpy(local_nid), F.asnumpy(F.tensor([0,1,2,3,4,5], F.int64)))
    assert_array_equal(F.asnumpy(local_eid), F.asnumpy(F.tensor([0,1,2,3,4,5], F.int64)))
    nid_partid = node_policy.to_partid(F.tensor([0,1,2,3,4,5], F.int64))
    eid_partid = edge_policy.to_partid(F.tensor([0,1,2,3,4,5], F.int64))
    assert_array_equal(F.asnumpy(nid_partid), F.asnumpy(F.tensor([0,0,0,0,0,0], F.int64)))
    assert_array_equal(F.asnumpy(eid_partid), F.asnumpy(F.tensor([0,0,0,0,0,0], F.int64)))
    assert node_policy.get_data_size() == len(node_map)
    assert edge_policy.get_data_size() == len(edge_map)

def start_server():
    kvserver = dgl.distributed.KVServer(server_id=0, 
                                        ip_config='ip_config.txt', 
                                        num_clients=1)
    server_state = dgl.distributed.ServerState(kv_store=kvserver)
    dgl.distributed.start_server(server_id=0, 
                                 ip_config='ip_config.txt', 
                                 num_clients=1, 
                                 server_state=server_state)

def start_client():
    dgl.distributed.connect_to_server(ip_config='ip_config.txt')
    # clean up
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()

@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_kv_store():
    ip_config = open("ip_config.txt", "w")
    ip_config.write('127.0.0.1 30050 1\n')
    ip_config.close()
    pid = os.fork()
    if pid == 0:
        start_server()
    else:
        time.sleep(1)
        start_client()

if __name__ == '__main__':
    test_partition_policy()
    test_kv_store()