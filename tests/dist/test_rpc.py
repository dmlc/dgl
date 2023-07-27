import multiprocessing as mp
import os
import unittest

import pytest
import utils

dgl_envs = f"PYTHONUNBUFFERED=1 DMLC_LOG_DEBUG=1 DGLBACKEND={os.environ.get('DGLBACKEND')} DGL_LIBRARY_PATH={os.environ.get('DGL_LIBRARY_PATH')} PYTHONPATH={os.environ.get('PYTHONPATH')} "


@unittest.skipIf(os.name == "nt", reason="Do not support windows yet")
def test_rpc():
    ip_config = os.environ.get("DIST_DGL_TEST_IP_CONFIG", "ip_config.txt")
    num_clients = 1
    num_servers = 1
    ips = utils.get_ips(ip_config)
    num_machines = len(ips)
    test_bin = os.path.join(
        os.environ.get("DIST_DGL_TEST_PY_BIN_DIR", "."), "rpc_basic.py"
    )
    base_envs = (
        dgl_envs
        + f" DGL_DIST_MODE=distributed DIST_DGL_TEST_IP_CONFIG={ip_config} DIST_DGL_TEST_NUM_SERVERS={num_servers} "
    )
    procs = []
    # start server processes
    server_id = 0
    for ip in ips:
        for _ in range(num_servers):
            server_envs = (
                base_envs
                + f" DIST_DGL_TEST_ROLE=server DIST_DGL_TEST_SERVER_ID={server_id} DIST_DGL_TEST_NUM_CLIENTS={num_clients * num_machines} "
            )
            procs.append(
                utils.execute_remote(server_envs + " python3 " + test_bin, ip)
            )
            server_id += 1
    # start client processes
    client_envs = (
        base_envs + " DIST_DGL_TEST_ROLE=client DIST_DGL_TEST_GROUP_ID=0 "
    )
    for ip in ips:
        for _ in range(num_clients):
            procs.append(
                utils.execute_remote(client_envs + " python3 " + test_bin, ip)
            )
    for p in procs:
        p.join()
        assert p.exitcode == 0
