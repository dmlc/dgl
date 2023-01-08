import multiprocessing as mp
import os
import subprocess
from typing import Optional


def run(ssh_cmd):
    subprocess.check_call(ssh_cmd, shell=True)


def execute_remote(
    cmd: str, ip: str, port: Optional[int] = 22, username: Optional[str] = ""
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

    custom_port = os.getenv("DIST_DGL_TEST_SSH_PORT", "")
    if custom_port:
        port = custom_port

    custom_ssh_key = os.getenv("DIST_DGL_TEST_SSH_KEY", "")
    if custom_ssh_key:
        custom_ssh_key = os.path.expanduser(custom_ssh_key)
        custom_ssh_key = "-i " + custom_ssh_key

    ssh_setup = os.getenv("DIST_DGL_TEST_SSH_SETUP", "")
    if ssh_setup:
        cmd = ssh_setup + ";" + cmd
    # Construct ssh command that executes `cmd` on the remote host
    ssh_cmd = "ssh -o StrictHostKeyChecking=no {ssh_key} -p {port} {ip_prefix}{ip} '{cmd}'".format(
        ssh_key=custom_ssh_key,
        port=str(port),
        ip_prefix=ip_prefix,
        ip=ip,
        cmd=cmd,
    )
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=run, args=(ssh_cmd,))
    proc.start()
    return proc


def get_ips(ip_config):
    ips = []
    with open(ip_config) as f:
        for line in f:
            result = line.strip().split()
            if len(result) != 1:
                raise RuntimeError(
                    "Invalid format of ip_config:{}".format(ip_config)
                )
            ips.append(result[0])
    return ips
