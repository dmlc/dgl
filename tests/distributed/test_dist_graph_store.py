import os

os.environ["OMP_NUM_THREADS"] = "1"
import math
import multiprocessing as mp
import pickle
import socket
import sys
import time
import unittest
from multiprocessing import Condition, Manager, Process, Value

import backend as F
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy import sparse as spsp
from utils import create_random_graph, generate_ip_config, reset_envs

import dgl
from dgl.data.utils import load_graphs, save_graphs
from dgl.distributed import (
    DistGraph,
    DistGraphServer,
    edge_split,
    load_partition,
    load_partition_book,
    node_split,
    partition_graph,
)
from dgl.heterograph_index import create_unitgraph_from_coo

if os.name != "nt":
    import fcntl
    import struct


def run_server(
    graph_name,
    server_id,
    server_count,
    num_clients,
    shared_mem,
    keep_alive=False,
):
    g = DistGraphServer(
        server_id,
        "kv_ip_config.txt",
        server_count,
        num_clients,
        "/tmp/dist_graph/{}.json".format(graph_name),
        disable_shared_mem=not shared_mem,
        graph_format=["csc", "coo"],
        keep_alive=keep_alive,
    )
    print("start server", server_id)
    # verify dtype of underlying graph
    cg = g.client_g
    for k, dtype in dgl.distributed.dist_graph.FIELD_DICT.items():
        if k in cg.ndata:
            assert (
                F.dtype(cg.ndata[k]) == dtype
            ), "Data type of {} in ndata should be {}.".format(k, dtype)
        if k in cg.edata:
            assert (
                F.dtype(cg.edata[k]) == dtype
            ), "Data type of {} in edata should be {}.".format(k, dtype)
    g.start()


def emb_init(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())


def rand_init(shape, dtype):
    return F.tensor(np.random.normal(size=shape), F.float32)


def check_dist_graph_empty(g, num_clients, num_nodes, num_edges):
    # Test API
    assert g.number_of_nodes() == num_nodes
    assert g.number_of_edges() == num_edges

    # Test init node data
    new_shape = (g.number_of_nodes(), 2)
    g.ndata["test1"] = dgl.distributed.DistTensor(new_shape, F.int32)
    nids = F.arange(0, int(g.number_of_nodes() / 2))
    feats = g.ndata["test1"][nids]
    assert np.all(F.asnumpy(feats) == 0)

    # create a tensor and destroy a tensor and create it again.
    test3 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test3", init_func=rand_init
    )
    del test3
    test3 = dgl.distributed.DistTensor(
        (g.number_of_nodes(), 3), F.float32, "test3"
    )
    del test3

    # Test write data
    new_feats = F.ones((len(nids), 2), F.int32, F.cpu())
    g.ndata["test1"][nids] = new_feats
    feats = g.ndata["test1"][nids]
    assert np.all(F.asnumpy(feats) == 1)

    # Test metadata operations.
    assert g.node_attr_schemes()["test1"].dtype == F.int32

    print("end")


def run_client_empty(
    graph_name, part_id, server_count, num_clients, num_nodes, num_edges
):
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    dgl.distributed.initialize("kv_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id, None
    )
    g = DistGraph(graph_name, gpb=gpb)
    check_dist_graph_empty(g, num_clients, num_nodes, num_edges)


def check_server_client_empty(shared_mem, num_servers, num_clients):
    prepare_dist(num_servers)
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = "dist_graph_test_1"
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context("spawn")
    for serv_id in range(num_servers):
        p = ctx.Process(
            target=run_server,
            args=(graph_name, serv_id, num_servers, num_clients, shared_mem),
        )
        serv_ps.append(p)
        p.start()

    cli_ps = []
    for cli_id in range(num_clients):
        print("start client", cli_id)
        p = ctx.Process(
            target=run_client_empty,
            args=(
                graph_name,
                0,
                num_servers,
                num_clients,
                g.number_of_nodes(),
                g.number_of_edges(),
            ),
        )
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()

    for p in serv_ps:
        p.join()

    print("clients have terminated")


def run_client(
    graph_name,
    part_id,
    server_count,
    num_clients,
    num_nodes,
    num_edges,
    group_id,
):
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    os.environ["DGL_GROUP_ID"] = str(group_id)
    dgl.distributed.initialize("kv_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id, None
    )
    g = DistGraph(graph_name, gpb=gpb)
    check_dist_graph(g, num_clients, num_nodes, num_edges)


def run_emb_client(
    graph_name,
    part_id,
    server_count,
    num_clients,
    num_nodes,
    num_edges,
    group_id,
):
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    os.environ["DGL_GROUP_ID"] = str(group_id)
    dgl.distributed.initialize("kv_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id, None
    )
    g = DistGraph(graph_name, gpb=gpb)
    check_dist_emb(g, num_clients, num_nodes, num_edges)


def run_client_hierarchy(
    graph_name, part_id, server_count, node_mask, edge_mask, return_dict
):
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    dgl.distributed.initialize("kv_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id, None
    )
    g = DistGraph(graph_name, gpb=gpb)
    node_mask = F.tensor(node_mask)
    edge_mask = F.tensor(edge_mask)
    nodes = node_split(
        node_mask,
        g.get_partition_book(),
        node_trainer_ids=g.ndata["trainer_id"],
    )
    edges = edge_split(
        edge_mask,
        g.get_partition_book(),
        edge_trainer_ids=g.edata["trainer_id"],
    )
    rank = g.rank()
    return_dict[rank] = (nodes, edges)


def check_dist_emb(g, num_clients, num_nodes, num_edges):
    from dgl.distributed import DistEmbedding
    from dgl.distributed.optim import SparseAdagrad

    # Test sparse emb
    try:
        emb = DistEmbedding(g.number_of_nodes(), 1, "emb1", emb_init)
        nids = F.arange(0, int(g.number_of_nodes()))
        lr = 0.001
        optimizer = SparseAdagrad([emb], lr=lr)
        with F.record_grad():
            feats = emb(nids)
            assert np.all(F.asnumpy(feats) == np.zeros((len(nids), 1)))
            loss = F.sum(feats + 1, 0)
        loss.backward()
        optimizer.step()
        feats = emb(nids)
        if num_clients == 1:
            assert_almost_equal(F.asnumpy(feats), np.ones((len(nids), 1)) * -lr)
        rest = np.setdiff1d(np.arange(g.number_of_nodes()), F.asnumpy(nids))
        feats1 = emb(rest)
        assert np.all(F.asnumpy(feats1) == np.zeros((len(rest), 1)))

        policy = dgl.distributed.PartitionPolicy("node", g.get_partition_book())
        grad_sum = dgl.distributed.DistTensor(
            (g.number_of_nodes(), 1), F.float32, "emb1_sum", policy
        )
        if num_clients == 1:
            assert np.all(
                F.asnumpy(grad_sum[nids])
                == np.ones((len(nids), 1)) * num_clients
            )
        assert np.all(F.asnumpy(grad_sum[rest]) == np.zeros((len(rest), 1)))

        emb = DistEmbedding(g.number_of_nodes(), 1, "emb2", emb_init)
        with F.no_grad():
            feats1 = emb(nids)
        assert np.all(F.asnumpy(feats1) == 0)

        optimizer = SparseAdagrad([emb], lr=lr)
        with F.record_grad():
            feats1 = emb(nids)
            feats2 = emb(nids)
            feats = F.cat([feats1, feats2], 0)
            assert np.all(F.asnumpy(feats) == np.zeros((len(nids) * 2, 1)))
            loss = F.sum(feats + 1, 0)
        loss.backward()
        optimizer.step()
        with F.no_grad():
            feats = emb(nids)
        if num_clients == 1:
            assert_almost_equal(
                F.asnumpy(feats), np.ones((len(nids), 1)) * 1 * -lr
            )
        rest = np.setdiff1d(np.arange(g.number_of_nodes()), F.asnumpy(nids))
        feats1 = emb(rest)
        assert np.all(F.asnumpy(feats1) == np.zeros((len(rest), 1)))
    except NotImplementedError as e:
        pass
    except Exception as e:
        print(e)
        sys.exit(-1)


def check_dist_graph(g, num_clients, num_nodes, num_edges):
    # Test API
    assert g.number_of_nodes() == num_nodes
    assert g.number_of_edges() == num_edges

    # Test reading node data
    nids = F.arange(0, int(g.number_of_nodes() / 2))
    feats1 = g.ndata["features"][nids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == nids))

    # Test reading edge data
    eids = F.arange(0, int(g.number_of_edges() / 2))
    feats1 = g.edata["features"][eids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == eids))

    # Test edge_subgraph
    sg = g.edge_subgraph(eids)
    assert sg.num_edges() == len(eids)
    assert F.array_equal(sg.edata[dgl.EID], eids)

    # Test init node data
    new_shape = (g.number_of_nodes(), 2)
    test1 = dgl.distributed.DistTensor(new_shape, F.int32)
    g.ndata["test1"] = test1
    feats = g.ndata["test1"][nids]
    assert np.all(F.asnumpy(feats) == 0)
    assert test1.count_nonzero() == 0

    # reference to a one that exists
    test2 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test2", init_func=rand_init
    )
    test3 = dgl.distributed.DistTensor(new_shape, F.float32, "test2")
    assert np.all(F.asnumpy(test2[nids]) == F.asnumpy(test3[nids]))

    # create a tensor and destroy a tensor and create it again.
    test3 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test3", init_func=rand_init
    )
    del test3
    test3 = dgl.distributed.DistTensor(
        (g.number_of_nodes(), 3), F.float32, "test3"
    )
    del test3

    # add tests for anonymous distributed tensor.
    test3 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    data = test3[0:10]
    test4 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    del test3
    test5 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    assert np.sum(F.asnumpy(test5[0:10] != data)) > 0

    # test a persistent tesnor
    test4 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test4", init_func=rand_init, persistent=True
    )
    del test4
    try:
        test4 = dgl.distributed.DistTensor(
            (g.number_of_nodes(), 3), F.float32, "test4"
        )
        raise Exception("")
    except:
        pass

    # Test write data
    new_feats = F.ones((len(nids), 2), F.int32, F.cpu())
    g.ndata["test1"][nids] = new_feats
    feats = g.ndata["test1"][nids]
    assert np.all(F.asnumpy(feats) == 1)

    # Test metadata operations.
    assert len(g.ndata["features"]) == g.number_of_nodes()
    assert g.ndata["features"].shape == (g.number_of_nodes(), 1)
    assert g.ndata["features"].dtype == F.int64
    assert g.node_attr_schemes()["features"].dtype == F.int64
    assert g.node_attr_schemes()["test1"].dtype == F.int32
    assert g.node_attr_schemes()["features"].shape == (1,)

    selected_nodes = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    # Test node split
    nodes = node_split(selected_nodes, g.get_partition_book())
    nodes = F.asnumpy(nodes)
    # We only have one partition, so the local nodes are basically all nodes in the graph.
    local_nids = np.arange(g.number_of_nodes())
    for n in nodes:
        assert n in local_nids

    print("end")


def check_dist_emb_server_client(
    shared_mem, num_servers, num_clients, num_groups=1
):
    prepare_dist(num_servers)
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = (
        f"check_dist_emb_{shared_mem}_{num_servers}_{num_clients}_{num_groups}"
    )
    g.ndata["features"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata["features"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context("spawn")
    keep_alive = num_groups > 1
    for serv_id in range(num_servers):
        p = ctx.Process(
            target=run_server,
            args=(
                graph_name,
                serv_id,
                num_servers,
                num_clients,
                shared_mem,
                keep_alive,
            ),
        )
        serv_ps.append(p)
        p.start()

    cli_ps = []
    for cli_id in range(num_clients):
        for group_id in range(num_groups):
            print("start client[{}] for group[{}]".format(cli_id, group_id))
            p = ctx.Process(
                target=run_emb_client,
                args=(
                    graph_name,
                    0,
                    num_servers,
                    num_clients,
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    group_id,
                ),
            )
            p.start()
            time.sleep(1)  # avoid race condition when instantiating DistGraph
            cli_ps.append(p)

    for p in cli_ps:
        p.join()
        assert p.exitcode == 0

    if keep_alive:
        for p in serv_ps:
            assert p.is_alive()
        # force shutdown server
        dgl.distributed.shutdown_servers("kv_ip_config.txt", num_servers)
    for p in serv_ps:
        p.join()

    print("clients have terminated")


def check_server_client(shared_mem, num_servers, num_clients, num_groups=1):
    prepare_dist(num_servers)
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = f"check_server_client_{shared_mem}_{num_servers}_{num_clients}_{num_groups}"
    g.ndata["features"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata["features"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context("spawn")
    keep_alive = num_groups > 1
    for serv_id in range(num_servers):
        p = ctx.Process(
            target=run_server,
            args=(
                graph_name,
                serv_id,
                num_servers,
                num_clients,
                shared_mem,
                keep_alive,
            ),
        )
        serv_ps.append(p)
        p.start()

    # launch different client groups simultaneously
    cli_ps = []
    for cli_id in range(num_clients):
        for group_id in range(num_groups):
            print("start client[{}] for group[{}]".format(cli_id, group_id))
            p = ctx.Process(
                target=run_client,
                args=(
                    graph_name,
                    0,
                    num_servers,
                    num_clients,
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    group_id,
                ),
            )
            p.start()
            time.sleep(1)  # avoid race condition when instantiating DistGraph
            cli_ps.append(p)
    for p in cli_ps:
        p.join()

    if keep_alive:
        for p in serv_ps:
            assert p.is_alive()
        # force shutdown server
        dgl.distributed.shutdown_servers("kv_ip_config.txt", num_servers)
    for p in serv_ps:
        p.join()

    print("clients have terminated")


def check_server_client_hierarchy(shared_mem, num_servers, num_clients):
    prepare_dist(num_servers)
    g = create_random_graph(10000)

    # Partition the graph
    num_parts = 1
    graph_name = "dist_graph_test_2"
    g.ndata["features"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata["features"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(
        g,
        graph_name,
        num_parts,
        "/tmp/dist_graph",
        num_trainers_per_machine=num_clients,
    )

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context("spawn")
    for serv_id in range(num_servers):
        p = ctx.Process(
            target=run_server,
            args=(graph_name, serv_id, num_servers, num_clients, shared_mem),
        )
        serv_ps.append(p)
        p.start()

    cli_ps = []
    manager = mp.Manager()
    return_dict = manager.dict()
    node_mask = np.zeros((g.number_of_nodes(),), np.int32)
    edge_mask = np.zeros((g.number_of_edges(),), np.int32)
    nodes = np.random.choice(
        g.number_of_nodes(), g.number_of_nodes() // 10, replace=False
    )
    edges = np.random.choice(
        g.number_of_edges(), g.number_of_edges() // 10, replace=False
    )
    node_mask[nodes] = 1
    edge_mask[edges] = 1
    nodes = np.sort(nodes)
    edges = np.sort(edges)
    for cli_id in range(num_clients):
        print("start client", cli_id)
        p = ctx.Process(
            target=run_client_hierarchy,
            args=(
                graph_name,
                0,
                num_servers,
                node_mask,
                edge_mask,
                return_dict,
            ),
        )
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()
    for p in serv_ps:
        p.join()

    nodes1 = []
    edges1 = []
    for n, e in return_dict.values():
        nodes1.append(n)
        edges1.append(e)
    nodes1, _ = F.sort_1d(F.cat(nodes1, 0))
    edges1, _ = F.sort_1d(F.cat(edges1, 0))
    assert np.all(F.asnumpy(nodes1) == nodes)
    assert np.all(F.asnumpy(edges1) == edges)

    print("clients have terminated")


def run_client_hetero(
    graph_name, part_id, server_count, num_clients, num_nodes, num_edges
):
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    dgl.distributed.initialize("kv_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id, None
    )
    g = DistGraph(graph_name, gpb=gpb)
    check_dist_graph_hetero(g, num_clients, num_nodes, num_edges)


def create_random_hetero():
    num_nodes = {"n1": 10000, "n2": 10010, "n3": 10020}
    etypes = [("n1", "r1", "n2"), ("n1", "r2", "n3"), ("n2", "r3", "n3")]
    edges = {}
    for etype in etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=0.001,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    g = dgl.heterograph(edges, num_nodes)
    g.nodes["n1"].data["feat"] = F.unsqueeze(
        F.arange(0, g.number_of_nodes("n1")), 1
    )
    g.edges["r1"].data["feat"] = F.unsqueeze(
        F.arange(0, g.number_of_edges("r1")), 1
    )
    return g


def check_dist_graph_hetero(g, num_clients, num_nodes, num_edges):
    # Test API
    for ntype in num_nodes:
        assert ntype in g.ntypes
        assert num_nodes[ntype] == g.number_of_nodes(ntype)
    for etype in num_edges:
        assert etype in g.etypes
        assert num_edges[etype] == g.number_of_edges(etype)
    etypes = [("n1", "r1", "n2"), ("n1", "r2", "n3"), ("n2", "r3", "n3")]
    for i, etype in enumerate(g.canonical_etypes):
        assert etype[0] == etypes[i][0]
        assert etype[1] == etypes[i][1]
        assert etype[2] == etypes[i][2]
    assert g.number_of_nodes() == sum([num_nodes[ntype] for ntype in num_nodes])
    assert g.number_of_edges() == sum([num_edges[etype] for etype in num_edges])

    # Test reading node data
    nids = F.arange(0, int(g.number_of_nodes("n1") / 2))
    feats1 = g.nodes["n1"].data["feat"][nids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == nids))

    # Test reading edge data
    eids = F.arange(0, int(g.number_of_edges("r1") / 2))
    feats1 = g.edges["r1"].data["feat"][eids]
    feats = F.squeeze(feats1, 1)
    assert np.all(F.asnumpy(feats == eids))

    # Test edge_subgraph
    sg = g.edge_subgraph({"r1": eids})
    assert sg.num_edges() == len(eids)
    assert F.array_equal(sg.edata[dgl.EID], eids)
    sg = g.edge_subgraph({("n1", "r1", "n2"): eids})
    assert sg.num_edges() == len(eids)
    assert F.array_equal(sg.edata[dgl.EID], eids)

    # Test init node data
    new_shape = (g.number_of_nodes("n1"), 2)
    g.nodes["n1"].data["test1"] = dgl.distributed.DistTensor(new_shape, F.int32)
    feats = g.nodes["n1"].data["test1"][nids]
    assert np.all(F.asnumpy(feats) == 0)

    # create a tensor and destroy a tensor and create it again.
    test3 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test3", init_func=rand_init
    )
    del test3
    test3 = dgl.distributed.DistTensor(
        (g.number_of_nodes("n1"), 3), F.float32, "test3"
    )
    del test3

    # add tests for anonymous distributed tensor.
    test3 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    data = test3[0:10]
    test4 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    del test3
    test5 = dgl.distributed.DistTensor(
        new_shape, F.float32, init_func=rand_init
    )
    assert np.sum(F.asnumpy(test5[0:10] != data)) > 0

    # test a persistent tesnor
    test4 = dgl.distributed.DistTensor(
        new_shape, F.float32, "test4", init_func=rand_init, persistent=True
    )
    del test4
    try:
        test4 = dgl.distributed.DistTensor(
            (g.number_of_nodes("n1"), 3), F.float32, "test4"
        )
        raise Exception("")
    except:
        pass

    # Test write data
    new_feats = F.ones((len(nids), 2), F.int32, F.cpu())
    g.nodes["n1"].data["test1"][nids] = new_feats
    feats = g.nodes["n1"].data["test1"][nids]
    assert np.all(F.asnumpy(feats) == 1)

    # Test metadata operations.
    assert len(g.nodes["n1"].data["feat"]) == g.number_of_nodes("n1")
    assert g.nodes["n1"].data["feat"].shape == (g.number_of_nodes("n1"), 1)
    assert g.nodes["n1"].data["feat"].dtype == F.int64

    selected_nodes = (
        np.random.randint(0, 100, size=g.number_of_nodes("n1")) > 30
    )
    # Test node split
    nodes = node_split(selected_nodes, g.get_partition_book(), ntype="n1")
    nodes = F.asnumpy(nodes)
    # We only have one partition, so the local nodes are basically all nodes in the graph.
    local_nids = np.arange(g.number_of_nodes("n1"))
    for n in nodes:
        assert n in local_nids

    print("end")


def check_server_client_hetero(shared_mem, num_servers, num_clients):
    prepare_dist(num_servers)
    g = create_random_hetero()

    # Partition the graph
    num_parts = 1
    graph_name = "dist_graph_test_3"
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    # let's just test on one partition for now.
    # We cannot run multiple servers and clients on the same machine.
    serv_ps = []
    ctx = mp.get_context("spawn")
    for serv_id in range(num_servers):
        p = ctx.Process(
            target=run_server,
            args=(graph_name, serv_id, num_servers, num_clients, shared_mem),
        )
        serv_ps.append(p)
        p.start()

    cli_ps = []
    num_nodes = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    num_edges = {etype: g.number_of_edges(etype) for etype in g.etypes}
    for cli_id in range(num_clients):
        print("start client", cli_id)
        p = ctx.Process(
            target=run_client_hetero,
            args=(
                graph_name,
                0,
                num_servers,
                num_clients,
                num_nodes,
                num_edges,
            ),
        )
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()

    for p in serv_ps:
        p.join()

    print("clients have terminated")


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support some of operations in DistGraph",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def test_server_client():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    check_server_client_hierarchy(False, 1, 4)
    check_server_client_empty(True, 1, 1)
    check_server_client_hetero(True, 1, 1)
    check_server_client_hetero(False, 1, 1)
    check_server_client(True, 1, 1)
    check_server_client(False, 1, 1)
    # [TODO][Rhett] Tests for multiple groups may fail sometimes and
    # root cause is unknown. Let's disable them for now.
    # check_server_client(True, 2, 2)
    # check_server_client(True, 1, 1, 2)
    # check_server_client(False, 1, 1, 2)
    # check_server_client(True, 2, 2, 2)


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support distributed DistEmbedding",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet",
    reason="Mxnet doesn't support distributed DistEmbedding",
)
def test_dist_emb_server_client():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "distributed"
    check_dist_emb_server_client(True, 1, 1)
    check_dist_emb_server_client(False, 1, 1)
    # [TODO][Rhett] Tests for multiple groups may fail sometimes and
    # root cause is unknown. Let's disable them for now.
    # check_dist_emb_server_client(True, 2, 2)
    # check_dist_emb_server_client(True, 1, 1, 2)
    # check_dist_emb_server_client(False, 1, 1, 2)
    # check_dist_emb_server_client(True, 2, 2, 2)


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support some of operations in DistGraph",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet", reason="Turn off Mxnet support"
)
def test_standalone():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "standalone"

    g = create_random_graph(10000)
    # Partition the graph
    num_parts = 1
    graph_name = "dist_graph_test_3"
    g.ndata["features"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata["features"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    dgl.distributed.initialize("kv_ip_config.txt")
    dist_g = DistGraph(
        graph_name, part_config="/tmp/dist_graph/{}.json".format(graph_name)
    )
    check_dist_graph(dist_g, 1, g.number_of_nodes(), g.number_of_edges())
    dgl.distributed.exit_client()  # this is needed since there's two test here in one process


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="TF doesn't support distributed DistEmbedding",
)
@unittest.skipIf(
    dgl.backend.backend_name == "mxnet",
    reason="Mxnet doesn't support distributed DistEmbedding",
)
def test_standalone_node_emb():
    reset_envs()
    os.environ["DGL_DIST_MODE"] = "standalone"

    g = create_random_graph(10000)
    # Partition the graph
    num_parts = 1
    graph_name = "dist_graph_test_3"
    g.ndata["features"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
    g.edata["features"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
    partition_graph(g, graph_name, num_parts, "/tmp/dist_graph")

    dgl.distributed.initialize("kv_ip_config.txt")
    dist_g = DistGraph(
        graph_name, part_config="/tmp/dist_graph/{}.json".format(graph_name)
    )
    check_dist_emb(dist_g, 1, g.number_of_nodes(), g.number_of_edges())
    dgl.distributed.exit_client()  # this is needed since there's two test here in one process


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@pytest.mark.parametrize("hetero", [True, False])
def test_split(hetero):
    if hetero:
        g = create_random_hetero()
        ntype = "n1"
        etype = "r1"
    else:
        g = create_random_graph(10000)
        ntype = "_N"
        etype = "_E"
    num_parts = 4
    num_hops = 2
    partition_graph(
        g,
        "dist_graph_test",
        num_parts,
        "/tmp/dist_graph",
        num_hops=num_hops,
        part_method="metis",
    )

    node_mask = np.random.randint(0, 100, size=g.number_of_nodes(ntype)) > 30
    edge_mask = np.random.randint(0, 100, size=g.number_of_edges(etype)) > 30
    selected_nodes = np.nonzero(node_mask)[0]
    selected_edges = np.nonzero(edge_mask)[0]

    # The code now collects the roles of all client processes and use the information
    # to determine how to split the workloads. Here is to simulate the multi-client
    # use case.
    def set_roles(num_clients):
        dgl.distributed.role.CUR_ROLE = "default"
        dgl.distributed.role.GLOBAL_RANK = {i: i for i in range(num_clients)}
        dgl.distributed.role.PER_ROLE_RANK["default"] = {
            i: i for i in range(num_clients)
        }

    for i in range(num_parts):
        set_roles(num_parts)
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            "/tmp/dist_graph/dist_graph_test.json", i
        )
        local_nids = F.nonzero_1d(part_g.ndata["inner_node"])
        local_nids = F.gather_row(part_g.ndata[dgl.NID], local_nids)
        if hetero:
            ntype_ids, nids = gpb.map_to_per_ntype(local_nids)
            local_nids = F.asnumpy(nids)[F.asnumpy(ntype_ids) == 0]
        else:
            local_nids = F.asnumpy(local_nids)
        nodes1 = np.intersect1d(selected_nodes, local_nids)
        nodes2 = node_split(
            node_mask, gpb, ntype=ntype, rank=i, force_even=False
        )
        assert np.all(np.sort(nodes1) == np.sort(F.asnumpy(nodes2)))
        for n in F.asnumpy(nodes2):
            assert n in local_nids

        set_roles(num_parts * 2)
        nodes3 = node_split(
            node_mask, gpb, ntype=ntype, rank=i * 2, force_even=False
        )
        nodes4 = node_split(
            node_mask, gpb, ntype=ntype, rank=i * 2 + 1, force_even=False
        )
        nodes5 = F.cat([nodes3, nodes4], 0)
        assert np.all(np.sort(nodes1) == np.sort(F.asnumpy(nodes5)))

        set_roles(num_parts)
        local_eids = F.nonzero_1d(part_g.edata["inner_edge"])
        local_eids = F.gather_row(part_g.edata[dgl.EID], local_eids)
        if hetero:
            etype_ids, eids = gpb.map_to_per_etype(local_eids)
            local_eids = F.asnumpy(eids)[F.asnumpy(etype_ids) == 0]
        else:
            local_eids = F.asnumpy(local_eids)
        edges1 = np.intersect1d(selected_edges, local_eids)
        edges2 = edge_split(
            edge_mask, gpb, etype=etype, rank=i, force_even=False
        )
        assert np.all(np.sort(edges1) == np.sort(F.asnumpy(edges2)))
        for e in F.asnumpy(edges2):
            assert e in local_eids

        set_roles(num_parts * 2)
        edges3 = edge_split(
            edge_mask, gpb, etype=etype, rank=i * 2, force_even=False
        )
        edges4 = edge_split(
            edge_mask, gpb, etype=etype, rank=i * 2 + 1, force_even=False
        )
        edges5 = F.cat([edges3, edges4], 0)
        assert np.all(np.sort(edges1) == np.sort(F.asnumpy(edges5)))


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_split_even():
    g = create_random_graph(10000)
    num_parts = 4
    num_hops = 2
    partition_graph(
        g,
        "dist_graph_test",
        num_parts,
        "/tmp/dist_graph",
        num_hops=num_hops,
        part_method="metis",
    )

    node_mask = np.random.randint(0, 100, size=g.number_of_nodes()) > 30
    edge_mask = np.random.randint(0, 100, size=g.number_of_edges()) > 30
    selected_nodes = np.nonzero(node_mask)[0]
    selected_edges = np.nonzero(edge_mask)[0]
    all_nodes1 = []
    all_nodes2 = []
    all_edges1 = []
    all_edges2 = []

    # The code now collects the roles of all client processes and use the information
    # to determine how to split the workloads. Here is to simulate the multi-client
    # use case.
    def set_roles(num_clients):
        dgl.distributed.role.CUR_ROLE = "default"
        dgl.distributed.role.GLOBAL_RANK = {i: i for i in range(num_clients)}
        dgl.distributed.role.PER_ROLE_RANK["default"] = {
            i: i for i in range(num_clients)
        }

    for i in range(num_parts):
        set_roles(num_parts)
        part_g, node_feats, edge_feats, gpb, _, _, _ = load_partition(
            "/tmp/dist_graph/dist_graph_test.json", i
        )
        local_nids = F.nonzero_1d(part_g.ndata["inner_node"])
        local_nids = F.gather_row(part_g.ndata[dgl.NID], local_nids)
        nodes = node_split(node_mask, gpb, rank=i, force_even=True)
        all_nodes1.append(nodes)
        subset = np.intersect1d(F.asnumpy(nodes), F.asnumpy(local_nids))
        print(
            "part {} get {} nodes and {} are in the partition".format(
                i, len(nodes), len(subset)
            )
        )

        set_roles(num_parts * 2)
        nodes1 = node_split(node_mask, gpb, rank=i * 2, force_even=True)
        nodes2 = node_split(node_mask, gpb, rank=i * 2 + 1, force_even=True)
        nodes3, _ = F.sort_1d(F.cat([nodes1, nodes2], 0))
        all_nodes2.append(nodes3)
        subset = np.intersect1d(F.asnumpy(nodes), F.asnumpy(nodes3))
        print("intersection has", len(subset))

        set_roles(num_parts)
        local_eids = F.nonzero_1d(part_g.edata["inner_edge"])
        local_eids = F.gather_row(part_g.edata[dgl.EID], local_eids)
        edges = edge_split(edge_mask, gpb, rank=i, force_even=True)
        all_edges1.append(edges)
        subset = np.intersect1d(F.asnumpy(edges), F.asnumpy(local_eids))
        print(
            "part {} get {} edges and {} are in the partition".format(
                i, len(edges), len(subset)
            )
        )

        set_roles(num_parts * 2)
        edges1 = edge_split(edge_mask, gpb, rank=i * 2, force_even=True)
        edges2 = edge_split(edge_mask, gpb, rank=i * 2 + 1, force_even=True)
        edges3, _ = F.sort_1d(F.cat([edges1, edges2], 0))
        all_edges2.append(edges3)
        subset = np.intersect1d(F.asnumpy(edges), F.asnumpy(edges3))
        print("intersection has", len(subset))
    all_nodes1 = F.cat(all_nodes1, 0)
    all_edges1 = F.cat(all_edges1, 0)
    all_nodes2 = F.cat(all_nodes2, 0)
    all_edges2 = F.cat(all_edges2, 0)
    all_nodes = np.nonzero(node_mask)[0]
    all_edges = np.nonzero(edge_mask)[0]
    assert np.all(all_nodes == F.asnumpy(all_nodes1))
    assert np.all(all_edges == F.asnumpy(all_edges1))
    assert np.all(all_nodes == F.asnumpy(all_nodes2))
    assert np.all(all_edges == F.asnumpy(all_edges2))


def prepare_dist(num_servers=1):
    generate_ip_config("kv_ip_config.txt", 1, num_servers=num_servers)


if __name__ == "__main__":
    os.makedirs("/tmp/dist_graph", exist_ok=True)
    test_dist_emb_server_client()
    test_server_client()
    test_split(True)
    test_split(False)
    test_split_even()
    test_standalone()
    test_standalone_node_emb()
