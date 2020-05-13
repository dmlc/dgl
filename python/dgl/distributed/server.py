"""Functions used by server."""

from . import rpc

def read_ip_config(filename):
    """Read network configuration information of server from file.

    The format of configuration file should be:

        [ip] [base_port] [server_count]

        172.31.40.143 30050 2
        172.31.36.140 30050 2
        172.31.47.147 30050 2
        172.31.30.180 30050 2

    Note that, DGL server supports backup servers that can share data with each others
    on the same machine via shared memory. So the server_count should be >= 1. For example, 
    if we set server_count to 5, it means that we have 1 main server and 4 backup servers on
    current machine. Note that, the count of server on each machine can be different.

    Parameters
    ----------
    filename : str
        name of configuration file.

    Returns
    -------
    dict
        server namebook. e.g.,

        [server_id]:[machine_id, ip, port, group_count]

          {0:[0, '172.31.40.143', 30050, 2],
           1:[0, '172.31.40.143', 30051, 2],
           2:[1, '172.31.36.140', 30050, 2],
           3:[1, '172.31.36.140', 30051, 2],
           4:[2, '172.31.47.147', 30050, 2],
           5:[2, '172.31.47.147', 30051, 2],
           6:[3, '172.31.30.180', 30050, 2],
           7:[3, '172.31.30.180', 30051, 2]}
    """
    assert len(filename) > 0, 'filename cannot be empty.'
    server_namebook = {}
    try:
        server_id = 0
        machine_id = 0
        lines = [line.rstrip('\n') for line in open(filename)]
        for line in lines:
            ip, port, server_count = line.split(' ')
            for s_count in range(int(server_count)):
                server_namebook[server_id] = [int(machine_id), ip, int(port)+s_count, int(server_count)]
                server_id += 1
            machine_id += 1
    except:
        print("Error: data format on each line should be: [ip] [base_port] [server_count]")
    return server_namebook

def start_server(server_id, ip_config, num_clients, queue_size=20*1024*1024*1024, net_type='socket'):
    """Start server.

    This is a blocking function -- it returns only when the server receives
    shutdown command from clients.

    Parameters
    ----------
    server_id : int
        Server ID starts from 0.
    ip_config : str
        Path of IP configuration file.
    num_clients : int
        Total number of clients that will be connected to server. 
        Note that, we do not support dynamic connection for now.
    queue_size : int
        Size (bytes) of server queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and 
        it will not allocate 20GB memory at once.
    net_type : str
        networking type, e.g., 'socket' (on default) or 'mpi' (do not support yet).
    """
    assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
    assert num_clients >= 0, 'num_client (%d) cannot be a negative number.' % num_client
    assert queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
    assert net_type in ('socket', 'mpi'), 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type
    rpc.set_rank(server_id)
    server_namebook = read_ip_config(ip_config)
    machine_id = server_namebook[server_id][0]
    ip = server_namebook[server_id][1]
    port = server_namebook[server_id][2]
    # group_count means the total number of server on each machine
    group_count = server_namebook[server_id][3]
    sender = rpc.create_sender(queue_size, net_type)
    receiver = rpc.create_receiver(queue_size, net_type)
    # wait all the senders connect to server.
    # Once all the senders connect to server, server will not accept new sender's connection
    print("Wait connections ...")
    rpc.receiver_wait(ip, port, num_clients)
    print("%d clients connected!" % num_clients)


def finalize():
    """Release resources of this server."""
    pass
