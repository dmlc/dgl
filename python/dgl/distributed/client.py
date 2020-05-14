"""Functions used by client."""

import os
import socket
if os.name != 'nt':
    import fcntl
    import struct

from . import rpc

CLIENT_REGISTER = 32451
SHUT_DOWN_SERVER = 32452

class ClientRegisterReuqest(rpc.Request):
    """Register client to specified server node.

    This request will send client's ip to server.

    Parameters
    ----------
    ip_addr : str
        client's IP address
    """
    def __init__(self, ip_addr):
        self._ip_addr = ip_addr

    def __getstate__(self):
        return self._ip_addr

    def __setstate__(self, state):
        self._ip_addr = state

    def process_request(self, server_state):
        return # do nothing

def local_ip4_addr_list():
    """Return a set of IPv4 address
    """
    nic = set()
    for ix in socket.if_nameindex():
        name = ix[1]
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip = socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', name[:15].encode("UTF-8")))[20:24])
        nic.add(ip)
    return nic

def get_local_machine_id(server_namebook):
    """Given server_namebook, find local machine ID

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of server nodes, where key is the server's ID 
        (start from 0) and value is the server's machine_id, IP address, port, and group_count, e.g.,

          {0:'[0, 172.31.40.143, 30050, 2],
           1:'[0, 172.31.40.143, 30051, 2],
           2:'[1, 172.31.36.140, 30050, 2],
           3:'[1, 172.31.36.140, 30051, 2],
           4:'[2, 172.31.47.147, 30050, 2],
           5:'[2, 172.31.47.147, 30051, 2],
           6:'[3, 172.31.30.180, 30050, 2],
           7:'[3, 172.31.30.180, 30051, 2]}

    Returns
    -------
    int
        local machine ID
    """
    res = 0
    for ID, data in server_namebook.items():
        machine_id = data[0]
        ip = data[1]
        if ip in local_ip4_addr_list():
            res = machine_id
            break
    return res

def get_local_usable_addr():
    """Get local usable IP and port

    Returns
    -------
    str
        IP address, e.g., '192.168.8.12:50051'
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
        
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()

    return IP + ':' + str(port)

def connect_to_server(ip_config, queue_size=20*1024*1024*1024, net_type='socket'):
    """Connect this client to server.

    Parameters
    ----------
    ip_config : str
        Path of server IP configuration file.
    queue_size : int
        Size (bytes) of client queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and 
        it will not allocate 20GB memory at once.
    net_type : str
        networking type, e.g., 'socket' (on default) or 'mpi' (do not support yet).

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    assert queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
    assert net_type in ('socket', 'mpi'), 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type
    server_namebook = rpc.read_ip_config(ip_config)
    num_servers = len(server_namebook)
    group_count = []
    max_machine_id = 0
    for server_info in server_namebook.values():
        group_count.append(server_info[3])
        if server_info[0] > max_machine_id:
            max_machine_id = server_info[0]
    num_machines = max_machine_id+1
    machine_id = get_local_machine_id(server_namebook)
    rpc.create_sender(queue_size, net_type)
    rpc.create_receiver(queue_size, net_type)
    # Get connected with all server nodes
    for ID, addr in server_namebook.items():
        server_ip = addr[1]
        server_port = addr[2]
        rpc.add_receiver_addr(server_ip, server_port, ID)
    rpc.sender_connect()
    # Get local usable IP address and port
    ip_addr = get_local_usable_addr()
    client_ip, client_port = ip_addr.split(':')
    # Register client on server
    print("111")
    register_req = ClientRegisterReuqest(ip_addr)
    print("222")
    msg_seq = rpc.incr_msg_seq()
    print("333")
    data, _ = rpc.serialize_to_payload(register_req)
    print("444")
    for server_id in range(num_servers):
        # client_id = 0 is temp ID because we don't assign client ID yet
        msg = rpc.RPCMessage(service_id=CLIENT_REGISTER, 
                             msg_seq=msg_seq, 
                             client_id=0, 
                             server_id=server_id,
                             data=data, 
                             tensors=[])
    print("555")
        #rpc.send_rpc_message(msg)


def finalize():
    """Release resources of this client."""
    rpc.finalize_sender()
    rpc.finalize_receiver()

def shutdown_servers():
    """Issue commands to remote servers to shut them down.

    Raises
    ------
    ConnectionError : If anything wrong with the connection.
    """
    req = ShutDownRequest()

