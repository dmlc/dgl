import os
import time
import numpy as np
import socket
from scipy import sparse as spsp
import dgl
import backend as F
import unittest, pytest
from dgl.graph_index import create_graph_index
import multiprocessing as mp
from numpy.testing import assert_array_equal

if os.name != 'nt':
    import fcntl
    import struct

def get_local_usable_addr():
    """Get local usable IP and port

    Returns
    -------
    str
        IP address, e.g., '192.168.8.12:50051'
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        sock.connect(('10.255.255.255', 1))
        ip_addr = sock.getsockname()[0]
    except ValueError:
        ip_addr = '127.0.0.1'
    finally:
        sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    sock.close()

    return ip_addr + ' ' + str(port)

def create_random_graph(n):
    arr = (spsp.random(n, n, density=0.001, format='coo') != 0).astype(np.int64)
    ig = create_graph_index(arr, readonly=True)
    return dgl.DGLGraph(ig)

# Create an one-part Graph
node_map = F.tensor([0,0,0,0,0,0], F.int64)
edge_map = F.tensor([0,0,0,0,0,0,0], F.int64)
global_nid = F.tensor([0,1,2,3,4,5], F.int64)
global_eid = F.tensor([0,1,2,3,4,5,6], F.int64)

g = dgl.DGLGraph()
g.add_nodes(6)
g.add_edge(0, 1) # 0
g.add_edge(0, 2) # 1
g.add_edge(0, 3) # 2
g.add_edge(2, 3) # 3
g.add_edge(1, 1) # 4
g.add_edge(0, 4) # 5
g.add_edge(2, 5) # 6

g.ndata[dgl.NID] = global_nid
g.edata[dgl.EID] = global_eid

gpb = dgl.distributed.GraphPartitionBook(part_id=0,
                                         num_parts=1,
                                         node_map=node_map,
                                         edge_map=edge_map,
                                         part_graph=g)

node_policy = dgl.distributed.PartitionPolicy(policy_str='node',
                                              part_id=0,
                                              partition_book=gpb)

edge_policy = dgl.distributed.PartitionPolicy(policy_str='edge',
                                              part_id=0,
                                              partition_book=gpb)

data_0 = F.tensor([[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.],[1.,1.]], F.float32)
data_0_1 = F.tensor([1.,2.,3.,4.,5.,6.], F.float32)
data_0_2 = F.tensor([1,2,3,4,5,6], F.int32)
data_0_3 = F.tensor([1,2,3,4,5,6], F.int64)
data_1 = F.tensor([[2.,2.],[2.,2.],[2.,2.],[2.,2.],[2.,2.],[2.,2.],[2.,2.]], F.float32)
data_2 = F.tensor([[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.],[0.,0.]], F.float32)

def init_zero_func(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

def udf_push(target, name, id_tensor, data_tensor):
    target[name][id_tensor] = data_tensor * data_tensor 

@unittest.skipIf(os.name == 'nt' or os.getenv('DGLBACKEND') == 'tensorflow', reason='Do not support windows and TF yet')
def test_partition_policy():
    assert node_policy.policy_str == 'node'
    assert edge_policy.policy_str == 'edge'
    assert node_policy.part_id == 0
    assert edge_policy.part_id == 0
    local_nid = node_policy.to_local(F.tensor([0,1,2,3,4,5]))
    local_eid = edge_policy.to_local(F.tensor([0,1,2,3,4,5,6]))
    assert_array_equal(F.asnumpy(local_nid), F.asnumpy(F.tensor([0,1,2,3,4,5], F.int64)))
    assert_array_equal(F.asnumpy(local_eid), F.asnumpy(F.tensor([0,1,2,3,4,5,6], F.int64)))
    nid_partid = node_policy.to_partid(F.tensor([0,1,2,3,4,5], F.int64))
    eid_partid = edge_policy.to_partid(F.tensor([0,1,2,3,4,5,6], F.int64))
    assert_array_equal(F.asnumpy(nid_partid), F.asnumpy(F.tensor([0,0,0,0,0,0], F.int64)))
    assert_array_equal(F.asnumpy(eid_partid), F.asnumpy(F.tensor([0,0,0,0,0,0,0], F.int64)))
    assert node_policy.get_data_size() == len(node_map)
    assert edge_policy.get_data_size() == len(edge_map)

def start_server():
	# Init kvserver
    kvserver = dgl.distributed.KVServer(server_id=0,
                                        ip_config='kv_ip_config.txt',
                                        num_clients=1)
    kvserver.add_part_policy(node_policy)
    kvserver.add_part_policy(edge_policy)
    kvserver.init_data('data_0', 'node', data_0)
    kvserver.init_data('data_0_1', 'node', data_0_1)
    kvserver.init_data('data_0_2', 'node', data_0_2)
    kvserver.init_data('data_0_3', 'node', data_0_3)
    # start server
    server_state = dgl.distributed.ServerState(kv_store=kvserver, local_g=None, partition_book=None)
    dgl.distributed.start_server(server_id=0,
                                 ip_config='kv_ip_config.txt',
                                 num_clients=1,
                                 server_state=server_state)

def start_client():
    # Note: connect to server first !
    dgl.distributed.connect_to_server(ip_config='kv_ip_config.txt')
    # Init kvclient
    kvclient = dgl.distributed.KVClient(ip_config='kv_ip_config.txt')
    kvclient.init_data(name='data_1', 
                       shape=F.shape(data_1), 
                       dtype=F.dtype(data_1), 
                       policy_str='edge', 
                       partition_book=gpb, 
                       init_func=init_zero_func)
    kvclient.init_data(name='data_2', 
                       shape=F.shape(data_2), 
                       dtype=F.dtype(data_2), 
                       policy_str='node', 
                       partition_book=gpb, 
                       init_func=init_zero_func)

    kvclient.map_shared_data(partition_book=gpb)
    
    # Test data_name_list
    name_list = kvclient.data_name_list()
    print(name_list)
    assert 'data_0' in name_list
    assert 'data_0_1' in name_list
    assert 'data_0_2' in name_list
    assert 'data_0_3' in name_list
    assert 'data_1' in name_list
    assert 'data_2' in name_list
    # Test get_meta_data
    meta = kvclient.get_data_meta('data_0')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0)
    assert shape == F.shape(data_0)
    assert policy.policy_str == 'node'

    meta = kvclient.get_data_meta('data_0_1')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_1)
    assert shape == F.shape(data_0_1)
    assert policy.policy_str == 'node'

    meta = kvclient.get_data_meta('data_0_2')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_2)
    assert shape == F.shape(data_0_2)
    assert policy.policy_str == 'node'

    meta = kvclient.get_data_meta('data_0_3')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_3)
    assert shape == F.shape(data_0_3)
    assert policy.policy_str == 'node'

    meta = kvclient.get_data_meta('data_1')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_1)
    assert shape == F.shape(data_1)
    assert policy.policy_str == 'edge'

    meta = kvclient.get_data_meta('data_2')
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_2)
    assert shape == F.shape(data_2)
    assert policy.policy_str == 'node'

    # Test push and pull
    id_tensor = F.tensor([0,2,4], F.int64)
    data_tensor = F.tensor([[6.,6.],[6.,6.],[6.,6.]], F.float32)
    kvclient.push(name='data_0',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    kvclient.push(name='data_1',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    kvclient.push(name='data_2',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    res = kvclient.pull(name='data_0', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name='data_1', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name='data_2', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    # Register new push handler
    kvclient.register_push_handler('data_0', udf_push)
    kvclient.register_push_handler('data_1', udf_push)
    kvclient.register_push_handler('data_2', udf_push)
    # Test push and pull
    kvclient.push(name='data_0',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    kvclient.push(name='data_1',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    kvclient.push(name='data_2',
                  id_tensor=id_tensor,
                  data_tensor=data_tensor)
    data_tensor = data_tensor * data_tensor
    res = kvclient.pull(name='data_0', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name='data_1', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name='data_2', id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    # clean up
    dgl.distributed.shutdown_servers()
    dgl.distributed.finalize_client()

@unittest.skipIf(os.name == 'nt' or os.getenv('DGLBACKEND') == 'tensorflow', reason='Do not support windows and TF yet')
def test_kv_store():
    ip_config = open("kv_ip_config.txt", "w")
    ip_addr = get_local_usable_addr()
    ip_config.write('%s 1\n' % ip_addr)
    ip_config.close()
    ctx = mp.get_context('spawn')
    pserver = ctx.Process(target=start_server)
    pclient = ctx.Process(target=start_client)
    pserver.start()
    time.sleep(1)
    pclient.start()
    pserver.join()
    pclient.join()

if __name__ == '__main__':
    test_partition_policy()
    test_kv_store()
