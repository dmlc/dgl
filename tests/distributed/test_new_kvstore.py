import multiprocessing as mp
import os
import time
import unittest

import backend as F

import dgl
from numpy.testing import assert_array_equal
from utils import generate_ip_config, reset_envs


# Create an one-part Graph
node_map = {"_N": F.tensor([[0, 6]], F.int64)}
edge_map = {("_N", "_E", "_N"): F.tensor([[0, 7]], F.int64)}
global_nid = F.tensor([0, 1, 2, 3, 4, 5], F.int64)
global_eid = F.tensor([0, 1, 2, 3, 4, 5, 6], F.int64)

g = dgl.graph([])
g.add_nodes(6)
g.add_edges(0, 1)  # 0
g.add_edges(0, 2)  # 1
g.add_edges(0, 3)  # 2
g.add_edges(2, 3)  # 3
g.add_edges(1, 1)  # 4
g.add_edges(0, 4)  # 5
g.add_edges(2, 5)  # 6

g.ndata[dgl.NID] = global_nid
g.edata[dgl.EID] = global_eid

gpb = dgl.distributed.graph_partition_book.RangePartitionBook(
    part_id=0,
    num_parts=1,
    node_map=node_map,
    edge_map=edge_map,
    ntypes={ntype: i for i, ntype in enumerate(g.ntypes)},
    etypes={etype: i for i, etype in enumerate(g.canonical_etypes)},
)

node_policy = dgl.distributed.PartitionPolicy(
    policy_str="node~_N", partition_book=gpb
)

edge_policy = dgl.distributed.PartitionPolicy(
    policy_str="edge~_N:_E:_N", partition_book=gpb
)

data_0 = F.tensor(
    [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
    F.float32,
)
data_0_1 = F.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], F.float32)
data_0_2 = F.tensor([1, 2, 3, 4, 5, 6], F.int32)
data_0_3 = F.tensor([1, 2, 3, 4, 5, 6], F.int64)
data_1 = F.tensor(
    [
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
        [2.0, 2.0],
    ],
    F.float32,
)
data_2 = F.tensor(
    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    F.float32,
)


def init_zero_func(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())


def udf_push(target, name, id_tensor, data_tensor):
    target[name][id_tensor] = data_tensor * data_tensor


def add_push(target, name, id_tensor, data_tensor):
    target[name][id_tensor] += data_tensor


@unittest.skipIf(
    os.name == "nt" or os.getenv("DGLBACKEND") == "tensorflow",
    reason="Do not support windows and TF yet",
)
def test_partition_policy():
    assert node_policy.part_id == 0
    assert edge_policy.part_id == 0
    local_nid = node_policy.to_local(F.tensor([0, 1, 2, 3, 4, 5]))
    local_eid = edge_policy.to_local(F.tensor([0, 1, 2, 3, 4, 5, 6]))
    assert_array_equal(
        F.asnumpy(local_nid), F.asnumpy(F.tensor([0, 1, 2, 3, 4, 5], F.int64))
    )
    assert_array_equal(
        F.asnumpy(local_eid),
        F.asnumpy(F.tensor([0, 1, 2, 3, 4, 5, 6], F.int64)),
    )
    nid_partid = node_policy.to_partid(F.tensor([0, 1, 2, 3, 4, 5], F.int64))
    eid_partid = edge_policy.to_partid(F.tensor([0, 1, 2, 3, 4, 5, 6], F.int64))
    assert_array_equal(
        F.asnumpy(nid_partid), F.asnumpy(F.tensor([0, 0, 0, 0, 0, 0], F.int64))
    )
    assert_array_equal(
        F.asnumpy(eid_partid),
        F.asnumpy(F.tensor([0, 0, 0, 0, 0, 0, 0], F.int64)),
    )
    assert node_policy.get_part_size() == len(local_nid)
    assert edge_policy.get_part_size() == len(local_eid)


def start_server(server_id, num_clients, num_servers):
    # Init kvserver
    print("Sleep 5 seconds to test client re-connect.")
    time.sleep(5)
    kvserver = dgl.distributed.KVServer(
        server_id=server_id,
        ip_config="kv_ip_config.txt",
        num_servers=num_servers,
        num_clients=num_clients,
    )
    kvserver.add_part_policy(node_policy)
    kvserver.add_part_policy(edge_policy)
    if kvserver.is_backup_server():
        kvserver.init_data("data_0", "node~_N")
        kvserver.init_data("data_0_1", "node~_N")
        kvserver.init_data("data_0_2", "node~_N")
        kvserver.init_data("data_0_3", "node~_N")
    else:
        kvserver.init_data("data_0", "node~_N", data_0)
        kvserver.init_data("data_0_1", "node~_N", data_0_1)
        kvserver.init_data("data_0_2", "node~_N", data_0_2)
        kvserver.init_data("data_0_3", "node~_N", data_0_3)
    # start server
    server_state = dgl.distributed.ServerState(
        kv_store=kvserver, local_g=None, partition_book=None
    )
    dgl.distributed.start_server(
        server_id=server_id,
        ip_config="kv_ip_config.txt",
        num_servers=num_servers,
        num_clients=num_clients,
        server_state=server_state,
    )


def start_server_mul_role(server_id, num_clients, num_servers):
    # Init kvserver
    kvserver = dgl.distributed.KVServer(
        server_id=server_id,
        ip_config="kv_ip_mul_config.txt",
        num_servers=num_servers,
        num_clients=num_clients,
    )
    kvserver.add_part_policy(node_policy)
    if kvserver.is_backup_server():
        kvserver.init_data("data_0", "node~_N")
    else:
        kvserver.init_data("data_0", "node~_N", data_0)
    # start server
    server_state = dgl.distributed.ServerState(
        kv_store=kvserver, local_g=None, partition_book=None
    )
    dgl.distributed.start_server(
        server_id=server_id,
        ip_config="kv_ip_mul_config.txt",
        num_servers=num_servers,
        num_clients=num_clients,
        server_state=server_state,
    )


def start_client(num_clients, num_servers):
    os.environ["DGL_DIST_MODE"] = "distributed"
    # Note: connect to server first !
    dgl.distributed.initialize(ip_config="kv_ip_config.txt")
    # Init kvclient
    kvclient = dgl.distributed.KVClient(
        ip_config="kv_ip_config.txt", num_servers=num_servers
    )
    kvclient.map_shared_data(partition_book=gpb)
    assert dgl.distributed.get_num_client() == num_clients
    kvclient.init_data(
        name="data_1",
        shape=F.shape(data_1),
        dtype=F.dtype(data_1),
        part_policy=edge_policy,
        init_func=init_zero_func,
    )
    kvclient.init_data(
        name="data_2",
        shape=F.shape(data_2),
        dtype=F.dtype(data_2),
        part_policy=node_policy,
        init_func=init_zero_func,
    )

    # Test data_name_list
    name_list = kvclient.data_name_list()
    print(name_list)
    assert "data_0" in name_list
    assert "data_0_1" in name_list
    assert "data_0_2" in name_list
    assert "data_0_3" in name_list
    assert "data_1" in name_list
    assert "data_2" in name_list
    # Test get_meta_data
    meta = kvclient.get_data_meta("data_0")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0)
    assert shape == F.shape(data_0)
    assert policy.policy_str == "node~_N"

    meta = kvclient.get_data_meta("data_0_1")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_1)
    assert shape == F.shape(data_0_1)
    assert policy.policy_str == "node~_N"

    meta = kvclient.get_data_meta("data_0_2")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_2)
    assert shape == F.shape(data_0_2)
    assert policy.policy_str == "node~_N"

    meta = kvclient.get_data_meta("data_0_3")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_0_3)
    assert shape == F.shape(data_0_3)
    assert policy.policy_str == "node~_N"

    meta = kvclient.get_data_meta("data_1")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_1)
    assert shape == F.shape(data_1)
    assert policy.policy_str == "edge~_N:_E:_N"

    meta = kvclient.get_data_meta("data_2")
    dtype, shape, policy = meta
    assert dtype == F.dtype(data_2)
    assert shape == F.shape(data_2)
    assert policy.policy_str == "node~_N"

    # Test push and pull
    id_tensor = F.tensor([0, 2, 4], F.int64)
    data_tensor = F.tensor([[6.0, 6.0], [6.0, 6.0], [6.0, 6.0]], F.float32)
    kvclient.push(name="data_0", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.push(name="data_1", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.push(name="data_2", id_tensor=id_tensor, data_tensor=data_tensor)
    res = kvclient.pull(name="data_0", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name="data_1", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name="data_2", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    # Register new push handler
    kvclient.register_push_handler("data_0", udf_push)
    kvclient.register_push_handler("data_1", udf_push)
    kvclient.register_push_handler("data_2", udf_push)
    # Test push and pull
    kvclient.push(name="data_0", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.push(name="data_1", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.push(name="data_2", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.barrier()
    data_tensor = data_tensor * data_tensor
    res = kvclient.pull(name="data_0", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name="data_1", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))
    res = kvclient.pull(name="data_2", id_tensor=id_tensor)
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))

    # Test delete data
    kvclient.delete_data("data_0")
    kvclient.delete_data("data_1")
    kvclient.delete_data("data_2")

    # Register new push handler
    kvclient.init_data(
        name="data_3",
        shape=F.shape(data_2),
        dtype=F.dtype(data_2),
        part_policy=node_policy,
        init_func=init_zero_func,
    )
    kvclient.register_push_handler("data_3", add_push)
    data_tensor = F.tensor([[6.0, 6.0], [6.0, 6.0], [6.0, 6.0]], F.float32)
    kvclient.barrier()
    time.sleep(kvclient.client_id + 1)
    print("add...")
    kvclient.push(name="data_3", id_tensor=id_tensor, data_tensor=data_tensor)
    kvclient.barrier()
    res = kvclient.pull(name="data_3", id_tensor=id_tensor)
    data_tensor = data_tensor * num_clients
    assert_array_equal(F.asnumpy(res), F.asnumpy(data_tensor))


def start_client_mul_role(i):
    os.environ["DGL_DIST_MODE"] = "distributed"
    # Initialize creates kvstore !
    dgl.distributed.initialize(ip_config="kv_ip_mul_config.txt")
    if i == 0:  # block one trainer
        time.sleep(5)
    kvclient = dgl.distributed.kvstore.get_kvstore()
    kvclient.barrier()
    print("i: %d role: %s" % (i, kvclient.role))

    assert dgl.distributed.role.get_num_trainers() == 2
    assert dgl.distributed.role.get_trainer_rank() < 2
    print(
        "trainer rank: %d, global rank: %d"
        % (
            dgl.distributed.role.get_trainer_rank(),
            dgl.distributed.role.get_global_rank(),
        )
    )
    dgl.distributed.exit_client()


@unittest.skipIf(
    os.name == "nt" or os.getenv("DGLBACKEND") == "tensorflow",
    reason="Do not support windows and TF yet",
)
def test_kv_store():
    reset_envs()
    num_servers = 2
    num_clients = 2
    generate_ip_config("kv_ip_config.txt", 1, num_servers)
    ctx = mp.get_context("spawn")
    pserver_list = []
    pclient_list = []
    os.environ["DGL_NUM_SERVER"] = str(num_servers)
    for i in range(num_servers):
        pserver = ctx.Process(
            target=start_server, args=(i, num_clients, num_servers)
        )
        pserver.start()
        pserver_list.append(pserver)
    for i in range(num_clients):
        pclient = ctx.Process(
            target=start_client, args=(num_clients, num_servers)
        )
        pclient.start()
        pclient_list.append(pclient)
    for i in range(num_clients):
        pclient_list[i].join()
    for i in range(num_servers):
        pserver_list[i].join()


@unittest.skipIf(
    os.name == "nt" or os.getenv("DGLBACKEND") == "tensorflow",
    reason="Do not support windows and TF yet",
)
def test_kv_multi_role():
    reset_envs()
    num_servers = 2
    num_trainers = 2
    num_samplers = 2
    generate_ip_config("kv_ip_mul_config.txt", 1, num_servers)
    # There are two trainer processes and each trainer process has two sampler processes.
    num_clients = num_trainers * (1 + num_samplers)
    ctx = mp.get_context("spawn")
    pserver_list = []
    pclient_list = []
    os.environ["DGL_NUM_SAMPLER"] = str(num_samplers)
    os.environ["DGL_NUM_SERVER"] = str(num_servers)
    for i in range(num_servers):
        pserver = ctx.Process(
            target=start_server_mul_role, args=(i, num_clients, num_servers)
        )
        pserver.start()
        pserver_list.append(pserver)
    for i in range(num_trainers):
        pclient = ctx.Process(target=start_client_mul_role, args=(i,))
        pclient.start()
        pclient_list.append(pclient)
    for i in range(num_trainers):
        pclient_list[i].join()
    for i in range(num_servers):
        pserver_list[i].join()


if __name__ == "__main__":
    test_partition_policy()
    test_kv_store()
    test_kv_multi_role()
