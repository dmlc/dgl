import os

os.environ["OMP_NUM_THREADS"] = "1"
import multiprocessing as mp
import pickle
import random
import socket
import sys
import time
import unittest

import backend as F

import dgl
import numpy as np
import torch as th
from dgl import function as fn
from dgl.distributed import (
    DistEmbedding,
    DistGraph,
    DistGraphServer,
    load_partition_book,
    partition_graph,
)
from dgl.distributed.optim import SparseAdagrad, SparseAdam
from scipy import sparse as spsp

# Set seeds to make tests fully reproducible.
SEED = 12345  # random.randint(1, 99999)
F.seed(SEED)


def create_random_graph(n):
    arr = (
        spsp.random(n, n, density=0.001, format="coo", random_state=100) != 0
    ).astype(np.int64)
    return dgl.from_scipy(arr)


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
        sock.connect(("10.255.255.255", 1))
        ip_addr = sock.getsockname()[0]
    except ValueError:
        ip_addr = "127.0.0.1"
    finally:
        sock.close()
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    sock.listen(1)
    port = sock.getsockname()[1]
    sock.close()

    return ip_addr + " " + str(port)


def prepare_dist():
    ip_config = open("optim_ip_config.txt", "w")
    ip_addr = get_local_usable_addr()
    ip_config.write("{}\n".format(ip_addr))
    ip_config.close()


def run_server(graph_name, server_id, server_count, num_clients, shared_mem):
    g = DistGraphServer(
        server_id,
        "optim_ip_config.txt",
        num_clients,
        server_count,
        "/tmp/dist_graph/{}.json".format(graph_name),
        disable_shared_mem=not shared_mem,
    )
    print("start server", server_id)
    g.start()


def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    th.manual_seed(0)
    th.nn.init.uniform_(arr, 0, 1.0)
    return arr


def run_client(graph_name, cli_id, part_id, server_count):
    device = F.ctx()
    time.sleep(5)
    os.environ["DGL_NUM_SERVER"] = str(server_count)
    dgl.distributed.initialize("optim_ip_config.txt")
    gpb, graph_name, _, _ = load_partition_book(
        "/tmp/dist_graph/{}.json".format(graph_name), part_id
    )
    g = DistGraph(graph_name, gpb=gpb)
    policy = dgl.distributed.PartitionPolicy("node", g.get_partition_book())
    num_nodes = g.num_nodes()
    emb_dim = 4
    dgl_emb = DistEmbedding(
        num_nodes,
        emb_dim,
        name="optim",
        init_func=initializer,
        part_policy=policy,
    )
    dgl_emb_zero = DistEmbedding(
        num_nodes,
        emb_dim,
        name="optim-zero",
        init_func=initializer,
        part_policy=policy,
    )
    dgl_adam = SparseAdam(params=[dgl_emb, dgl_emb_zero], lr=0.01)
    dgl_adam._world_size = 1
    dgl_adam._rank = 0

    torch_emb = th.nn.Embedding(num_nodes, emb_dim, sparse=True)
    torch_emb_zero = th.nn.Embedding(num_nodes, emb_dim, sparse=True)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb.weight, 0, 1.0)
    th.manual_seed(0)
    th.nn.init.uniform_(torch_emb_zero.weight, 0, 1.0)
    torch_adam = th.optim.SparseAdam(
        list(torch_emb.parameters()) + list(torch_emb_zero.parameters()),
        lr=0.01,
    )

    labels = th.ones((4,)).long()
    idx = th.randint(0, num_nodes, size=(4,))
    dgl_value = dgl_emb(idx, device).to(th.device("cpu"))
    torch_value = torch_emb(idx)
    torch_adam.zero_grad()
    torch_loss = th.nn.functional.cross_entropy(torch_value, labels)
    torch_loss.backward()
    torch_adam.step()

    dgl_adam.zero_grad()
    dgl_loss = th.nn.functional.cross_entropy(dgl_value, labels)
    dgl_loss.backward()
    dgl_adam.step()

    assert F.allclose(
        dgl_emb.weight[0 : num_nodes // 2], torch_emb.weight[0 : num_nodes // 2]
    )


def check_sparse_adam(num_trainer=1, shared_mem=True):
    prepare_dist()
    g = create_random_graph(2000)
    num_servers = num_trainer
    num_clients = num_trainer
    num_parts = 1

    graph_name = "dist_graph_test"
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
            target=run_client, args=(graph_name, cli_id, 0, num_servers)
        )
        p.start()
        cli_ps.append(p)

    for p in cli_ps:
        p.join()

    for p in serv_ps:
        p.join()


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_sparse_opt():
    os.environ["DGL_DIST_MODE"] = "distributed"
    check_sparse_adam(1, True)
    check_sparse_adam(1, False)


if __name__ == "__main__":
    os.makedirs("/tmp/dist_graph", exist_ok=True)
    test_sparse_opt()
