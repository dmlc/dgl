import subprocess
import os
import multiprocessing as mp
from typing import Optional
import unittest


def run(ssh_cmd):
    subprocess.check_call(ssh_cmd, shell=True)


def execute_remote(
    cmd: str,
    ip: str,
    port: Optional[int] = 22,
    username: Optional[str] = ""
) -> mp.Process:
    """Execute command line on remote machine via ssh.

    Args:
        cmd: User-defined command (udf) to execute on the remote host.
        ip: The ip-address of the host to run the command on.
        port: Port number that the host is listening on.
        username: Optional. If given, this will specify a username to use when issuing commands over SSH.
            Useful when your infra requires you to explicitly specify a username to avoid permission issues.

    Returns:
        Process: The Process whose run() is to run the `cmd` on the remote host. Returns when the cmd completes
            on the remote host.
    """
    ip_prefix = ""
    if username:
        ip_prefix += "{username}@".format(username=username)
    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = "ssh -o StrictHostKeyChecking=no -p {port} {ip_prefix}{ip} '{cmd}'".format(
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )
    ctx = mp.get_context('spawn')
    proc = ctx.Process(target=run, args=(ssh_cmd,))
    proc.start()
    return proc


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
