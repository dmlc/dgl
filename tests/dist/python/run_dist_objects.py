import dgl
import torch
import os
import numpy as np
import dgl.backend as F
from dgl.distributed import load_partition_book
import time

mode = os.environ.get('DIST_DGL_TEST_MODE', "")
graph_name = os.environ.get('DIST_DGL_TEST_GRAPH_NAME', 'random_test_graph')
num_part = int(os.environ.get('DIST_DGL_TEST_NUM_PART'))
num_servers_per_machine = int(os.environ.get('DIST_DGL_TEST_NUM_SERVER'))
num_client_per_machine = int(os.environ.get('DIST_DGL_TEST_NUM_CLIENT'))
shared_workspace = os.environ.get('DIST_DGL_TEST_WORKSPACE')
graph_path = os.environ.get('DIST_DGL_TEST_GRAPH_PATH')
part_id = int(os.environ.get('DIST_DGL_TEST_PART_ID'))
net_type = os.environ.get('DIST_DGL_TEST_NET_TYPE')
ip_config = os.environ.get('DIST_DGL_TEST_IP_CONFIG', 'ip_config.txt')

os.environ['DGL_DIST_MODE'] = 'distributed'

def zeros_init(shape, dtype):
    return F.zeros(shape, dtype=dtype, ctx=F.cpu())

def run_server(graph_name, server_id, server_count, num_clients, shared_mem, keep_alive=False):
    # server_count = num_servers_per_machine
    g = dgl.distributed.DistGraphServer(server_id, ip_config,
                        server_count, num_clients,
                        graph_path + '/{}.json'.format(graph_name),
                        disable_shared_mem=not shared_mem,
                        graph_format=['csc', 'coo'], keep_alive=keep_alive,
                        net_type=net_type)
    print('start server', server_id)
    g.start()

def dist_tensor_test_sanity(data_shape, rank, name=None):
    dist_ten = dgl.distributed.DistTensor(data_shape,
                                          F.int32,
                                          init_func=zeros_init,
                                          name=name)
    # arbitrary value
    stride = 3
    local_rank = rank % num_client_per_machine
    pos = (part_id // 2) * num_client_per_machine + local_rank
    if part_id % 2 == 0:
        dist_ten[pos*stride:(pos+1)*stride] = F.ones((stride, 2), dtype=F.int32, ctx=F.cpu()) * (pos+1)

    dgl.distributed.client_barrier()
    assert F.allclose(dist_ten[pos*stride:(pos+1)*stride],
                    F.ones((stride, 2), dtype=F.int32, ctx=F.cpu()) * (pos+1))


def dist_tensor_test_destroy_recreate(data_shape, name):
    dist_ten = dgl.distributed.DistTensor(data_shape, F.float32, name, init_func=zeros_init)
    del dist_ten

    dgl.distributed.client_barrier()

    new_shape = (data_shape[0], 4)
    dist_ten = dgl.distributed.DistTensor(new_shape, F.float32, name, init_func=zeros_init)

def dist_tensor_test_persistent(data_shape):
    dist_ten_name = 'persistent_dist_tensor'
    dist_ten = dgl.distributed.DistTensor(data_shape, F.float32, dist_ten_name, init_func=zeros_init,
                                          persistent=True)
    del dist_ten
    try:
        dist_ten = dgl.distributed.DistTensor(data_shape, F.float32, dist_ten_name)
        raise Exception('')
    except:
        pass


def test_dist_tensor(g, rank):
    first_type = g.ntypes[0]
    data_shape = (g.number_of_nodes(first_type), 2)
    dist_tensor_test_sanity(data_shape, rank)
    dist_tensor_test_sanity(data_shape, rank, name="DistTensorSanity")
    dist_tensor_test_destroy_recreate(data_shape, name="DistTensorRecreate")
    dist_tensor_test_persistent(data_shape)


if mode == "server":
    shared_mem = bool(int(os.environ.get('DIST_DGL_TEST_SHARED_MEM')))
    server_id = int(os.environ.get('DIST_DGL_TEST_SERVER_ID'))
    run_server(graph_name, server_id, server_count=num_servers_per_machine,
               num_clients=num_part*num_client_per_machine, shared_mem=shared_mem, keep_alive=False)
elif mode == "client":
    os.environ['DGL_NUM_SERVER'] = str(num_servers_per_machine)
    dgl.distributed.initialize(ip_config, net_type=net_type)
    global_rank = dgl.distributed.get_rank()

    gpb, graph_name, _, _ = load_partition_book(graph_path + '/{}.json'.format(graph_name), part_id, None)
    g = dgl.distributed.DistGraph(graph_name, gpb=gpb)
    target = os.environ.get('DIST_DGL_TEST_OBJECT_TYPE', 'DistTensor') 
    if target == "DistTensor":
        test_dist_tensor(g, global_rank)
    elif target == "DistEmbedding":
        # TODO: implement DistEmbedding
        pass
    else:
        print(target + " is not a valid DIST_DGL_TEST_OBJECT_TYPE")
else:
    print("DIST_DGL_TEST_MODE has to be either server or client")
    exit(1)

