import os
import random
import socket

import dgl

import numpy as np
import scipy.sparse as spsp


def generate_ip_config(file_name, num_machines, num_servers):
    """Get local IP and available ports, writes to file."""
    # get available IP in localhost
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        sock.connect(("10.255.255.255", 1))
        ip = sock.getsockname()[0]
    except ValueError:
        ip = "127.0.0.1"
    finally:
        sock.close()

    # scan available PORT
    ports = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    start = random.randint(10000, 30000)
    for port in range(start, 65535):
        try:
            sock.connect((ip, port))
            ports = []
        except:
            ports.append(port)
            if len(ports) == num_machines * num_servers:
                break
    sock.close()
    if len(ports) < num_machines * num_servers:
        raise RuntimeError(
            "Failed to get available IP/PORT with required numbers."
        )
    with open(file_name, "w") as f:
        for i in range(num_machines):
            f.write("{} {}\n".format(ip, ports[i * num_servers]))


def reset_envs():
    """Reset common environment variable which are set in tests."""
    for key in [
        "DGL_ROLE",
        "DGL_NUM_SAMPLER",
        "DGL_NUM_SERVER",
        "DGL_DIST_MODE",
        "DGL_NUM_CLIENT",
        "DGL_DIST_MAX_TRY_TIMES",
        "DGL_DIST_DEBUG",
    ]:
        if key in os.environ:
            os.environ.pop(key)


def create_random_graph(n):
    return dgl.rand_graph(n, int(n * n * 0.001))
