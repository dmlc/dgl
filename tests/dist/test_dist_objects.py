import multiprocessing as mp
import os
import subprocess
import unittest

import numpy as np
import pytest
import utils

import dgl
import dgl.backend as F
from dgl.distributed import partition_graph

graph_name = os.environ.get("DIST_DGL_TEST_GRAPH_NAME", "random_test_graph")
target = os.environ.get("DIST_DGL_TEST_OBJECT_TYPE", "")
shared_workspace = os.environ.get("DIST_DGL_TEST_WORKSPACE")


def create_graph(num_part, dist_graph_path, hetero):
    if not hetero:
        g = dgl.rand_graph(10000, 42000)
        g.ndata["feat"] = F.unsqueeze(F.arange(0, g.number_of_nodes()), 1)
        g.edata["feat"] = F.unsqueeze(F.arange(0, g.number_of_edges()), 1)
        partition_graph(g, graph_name, num_part, dist_graph_path)
    else:
        from scipy import sparse as spsp

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
        partition_graph(g, graph_name, num_part, dist_graph_path)


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
@pytest.mark.parametrize("net_type", ["tensorpipe", "socket"])
@pytest.mark.parametrize("num_servers", [1, 4])
@pytest.mark.parametrize("num_clients", [1, 4])
@pytest.mark.parametrize("hetero", [False, True])
@pytest.mark.parametrize("shared_mem", [False, True])
def test_dist_objects(net_type, num_servers, num_clients, hetero, shared_mem):
    if not shared_mem and num_servers > 1:
        pytest.skip(
            f"Backup servers are not supported when shared memory is disabled"
        )
    ip_config = os.environ.get("DIST_DGL_TEST_IP_CONFIG", "ip_config.txt")
    workspace = os.environ.get(
        "DIST_DGL_TEST_WORKSPACE", "/shared_workspace/dgl_dist_tensor_test/"
    )

    ips = utils.get_ips(ip_config)
    num_part = len(ips)

    test_bin = os.path.join(
        os.environ.get("DIST_DGL_TEST_PY_BIN_DIR", "."), "run_dist_objects.py"
    )

    dist_graph_path = os.path.join(
        workspace, "hetero_dist_graph" if hetero else "dist_graph"
    )
    if not os.path.isdir(dist_graph_path):
        create_graph(num_part, dist_graph_path, hetero)

    base_envs = (
        f"DIST_DGL_TEST_WORKSPACE={workspace} "
        f"DIST_DGL_TEST_NUM_PART={num_part} "
        f"DIST_DGL_TEST_NUM_SERVER={num_servers} "
        f"DIST_DGL_TEST_NUM_CLIENT={num_clients} "
        f"DIST_DGL_TEST_NET_TYPE={net_type} "
        f"DIST_DGL_TEST_GRAPH_PATH={dist_graph_path} "
        f"DIST_DGL_TEST_IP_CONFIG={ip_config} "
    )

    procs = []
    # Start server
    server_id = 0
    for part_id, ip in enumerate(ips):
        for _ in range(num_servers):
            cmd_envs = (
                base_envs + f"DIST_DGL_TEST_SERVER_ID={server_id} "
                f"DIST_DGL_TEST_PART_ID={part_id} "
                f"DIST_DGL_TEST_SHARED_MEM={str(int(shared_mem))} "
                f"DIST_DGL_TEST_MODE=server "
            )
            procs.append(
                utils.execute_remote(f"{cmd_envs} python3 {test_bin}", ip)
            )
            server_id += 1
    # Start client processes
    for part_id, ip in enumerate(ips):
        for _ in range(num_clients):
            cmd_envs = (
                base_envs + f"DIST_DGL_TEST_PART_ID={part_id} "
                f"DIST_DGL_TEST_OBJECT_TYPE={target} "
                f"DIST_DGL_TEST_MODE=client "
            )
            procs.append(
                utils.execute_remote(f"{cmd_envs} python3 {test_bin}", ip)
            )

    for p in procs:
        p.join()
        assert p.exitcode == 0
