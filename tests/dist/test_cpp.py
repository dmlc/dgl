import os
import unittest
from utils import execute_remote


@unittest.skipIf(os.name == 'nt', reason='Do not support windows yet')
def test_tensorpipe_comm():
    base_dir = os.environ.get('DIST_DGL_TEST_CPP_BIN_DIR', '.')
    ip_config = os.environ.get('DIST_DGL_TEST_IP_CONFIG', 'ip_config.txt')
    client_bin = os.path.join(base_dir, 'rpc_client')
    server_bin = os.path.join(base_dir, 'rpc_server')
    ips = []
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) != 1:
                raise RuntimeError(
                    "Invalid format of ip_config:{}".format(ip_config))
            ips.append(result[0])
    num_machines = len(ips)
    procs = []
    for ip in ips:
        procs.append(execute_remote(server_bin + " " +
                    str(num_machines) + " " + ip, ip))
    for ip in ips:
        procs.append(execute_remote(client_bin + " " + ip_config, ip))
    for p in procs:
        p.join()
        assert p.exitcode == 0
