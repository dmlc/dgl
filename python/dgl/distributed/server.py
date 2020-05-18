"""Functions used by server."""

import time

from . import rpc

WAIT_SEC = 3

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
    rpc.register_service(rpc.CLIENT_REGISTER, rpc.ClientRegisterReuqest, rpc.ClientRegisterResponse)
    rpc.set_rank(server_id)
    server_namebook = rpc.read_ip_config(ip_config)
    machine_id = server_namebook[server_id][0]
    ip = server_namebook[server_id][1]
    port = server_namebook[server_id][2]
    # group_count means the total number of server on each machine
    group_count = server_namebook[server_id][3]
    rpc.create_sender(queue_size, net_type)
    rpc.create_receiver(queue_size, net_type)
    # wait all the senders connect to server.
    # Once all the senders connect to server, server will not 
    # accept new sender's connection
    print("Wait connections ...")
    rpc.receiver_wait(ip, port, num_clients)
    print("%d clients connected!" % num_clients)
    # Recv all the client's IP and assign ID to client
    addr_list = []
    client_namebook = {}
    for i in range(num_clients):
        req = rpc.recv_request()
        addr_list.append(req._ip_addr)
    addr_list.sort()
    for ID in range(len(addr_list)):
      client_namebook[ID] = addr_list[ID]
    # Connect back to clients
    time.sleep(WAIT_SEC)
    for ID, addr in client_namebook.items():
        client_ip, client_port = addr.split(':')
        rpc.add_receiver_addr(client_ip, client_port, ID)
    rpc.sender_connect()
    # send ID back to clients
    if rpc.get_rank() == 0:
        for ID, _ in client_namebook.items():
            register_res = rpc.ClientRegisterResponse(ID)
            rpc.send_repsonse(ID, register_res)

    while True:
        print("Recv...")
        time.sleep(1)

    finalize()

def finalize():
    """Release resources of this server."""
    rpc.finalize_sender()
    rpc.finalize_receiver()
