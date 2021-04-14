"""Functions used by server."""

import time

from . import rpc
from .constants import MAX_QUEUE_SIZE

def start_server(server_id, ip_config, num_servers, num_clients, server_state, \
    max_queue_size=MAX_QUEUE_SIZE, net_type='socket'):
    """Start DGL server, which will be shared with all the rpc services.

    This is a blocking function -- it returns only when the server shutdown.

    Parameters
    ----------
    server_id : int
        Current server ID (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of clients that will be connected to the server.
        Note that, we do not support dynamic connection for now. It means
        that when all the clients connect to server, no client will can be added
        to the cluster.
    server_state : ServerSate object
        Store in main data used by server.
    max_queue_size : int
        Maximal size (bytes) of server queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound because DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
    assert num_servers > 0, 'num_servers (%d) must be a positive number.' % num_servers
    assert num_clients >= 0, 'num_client (%d) cannot be a negative number.' % num_client
    assert max_queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
    assert net_type in ('socket'), 'net_type (%s) can only be \'socket\'' % net_type
    # Register signal handler.
    rpc.register_sig_handler()
    # Register some basic services
    rpc.register_service(rpc.CLIENT_REGISTER,
                         rpc.ClientRegisterRequest,
                         rpc.ClientRegisterResponse)
    rpc.register_service(rpc.SHUT_DOWN_SERVER,
                         rpc.ShutDownRequest,
                         None)
    rpc.register_service(rpc.GET_NUM_CLIENT,
                         rpc.GetNumberClientsRequest,
                         rpc.GetNumberClientsResponse)
    rpc.register_service(rpc.CLIENT_BARRIER,
                         rpc.ClientBarrierRequest,
                         rpc.ClientBarrierResponse)
    rpc.set_rank(server_id)
    server_namebook = rpc.read_ip_config(ip_config, num_servers)
    machine_id = server_namebook[server_id][0]
    rpc.set_machine_id(machine_id)
    ip_addr = server_namebook[server_id][1]
    port = server_namebook[server_id][2]
    rpc.create_sender(max_queue_size, net_type)
    rpc.create_receiver(max_queue_size, net_type)
    # wait all the senders connect to server.
    # Once all the senders connect to server, server will not
    # accept new sender's connection
    print("Wait connections ...")
    rpc.receiver_wait(ip_addr, port, num_clients)
    print("%d clients connected!" % num_clients)
    rpc.set_num_client(num_clients)
    # Recv all the client's IP and assign ID to clients
    addr_list = []
    client_namebook = {}
    for _ in range(num_clients):
        req, _ = rpc.recv_request()
        addr_list.append(req.ip_addr)
    addr_list.sort()
    for client_id, addr in enumerate(addr_list):
        client_namebook[client_id] = addr
    for client_id, addr in client_namebook.items():
        client_ip, client_port = addr.split(':')
        rpc.add_receiver_addr(client_ip, client_port, client_id)
    time.sleep(3) # wait client's socket ready. 3 sec is enough.
    rpc.sender_connect()
    if rpc.get_rank() == 0: # server_0 send all the IDs
        for client_id, _ in client_namebook.items():
            register_res = rpc.ClientRegisterResponse(client_id)
            rpc.send_response(client_id, register_res)
    # main service loop
    while True:
        req, client_id = rpc.recv_request()
        res = req.process_request(server_state)
        if res is not None:
            if isinstance(res, list):
                for response in res:
                    target_id, res_data = response
                    rpc.send_response(target_id, res_data)
            elif isinstance(res, str) and res == 'exit':
                break # break the loop and exit server
            else:
                rpc.send_response(client_id, res)
