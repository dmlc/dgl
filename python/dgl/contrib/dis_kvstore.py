# This file contains DGL distributed kvstore APIs.
from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _network_wait, _add_receiver_addr
from ..network import _receiver_wait, _sender_connect
from ..network import _send_kv_msg, _recv_kv_msg
from ..network import _clear_kv_msg
from ..network import KVMsgType, KVStoreMsg

from .. import backend as F
from .._ffi.ndarray import empty_shared_mem

import os
import numpy as np
import socket
if os.name != 'nt':
    import fcntl
    import struct

def read_ip_config(filename):
    """Read networking configuration from file.

    Format:

        [server_id] [ip] [port]

        0 172.31.40.143 50050
        1 172.31.36.140 50050
        2 172.31.47.147 50050
        3 172.31.30.180 50050

    Parameters
    ----------
    filename : str
        name of target configure file.

    Returns
    -------
    dict
        server namebook, e.g., 

          {0:'172.31.40.143:50050',
           1:'172.31.36.140:50050',
           2:'172.31.47.147:50050',
           3:'172.31.30.180:50050'}
    """
    assert len(filename) > 0, 'filename cannot be empty.'

    server_namebook = {}

    try:
        lines = [line.rstrip('\n') for line in open(filename)]
        for line in lines:
            ID, ip, port = line.split(' ')
            server_namebook[int(ID)] = ip+':'+port
    except:
        print("Incorrect format IP configure file, the data format on each line should be: [server_id] [ip] [port]")

    return server_namebook


def start_server(server_id, ip_config, num_client, ndata, edata, ndata_g2l=None, edata_g2l=None, msg_queue_size=2*1024*1024*1024):
    """Start a kvserver node. 

    This function will be blocked by server.start() api.

    Parameters
    ----------
    server_id : int
        ID of current server node (start from 0)
    ip_config : str
        Filename of server IP configure file.
    num_client : int
        Total number of client nodes
    ndata : dict of tensor (mx.ndarray or torch.tensor)
        node data
    edata : dict of tensor (mx.ndarray or torch.tensor)
        edge data
    ndata_g2l : dict of tensor (mx.ndarray or torch.tensor)
        global2local mapping of node data
    edata_g2l : dict of tensor (mx.ndarray or torch.tensor)
        global2local mapping of edge data
    msg_queue_size : int
        Size of message queue (2GB by default)
    """
    assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
    assert len(ip_config) > 0, 'ip_config cannot be empty.'
    assert num_client > 0, 'num_client (%d) cnanot be a negative number.' % num_client

    server_namebook = read_ip_config(ip_config)

    server = KVServer(
        server_id=server_id, 
        server_addr=server_namebook[server_id],
        num_client=num_client,
        msg_queue_size=msg_queue_size)

    for name, data in ndata.items():
        server.init_data(name=name, data_tensor=data)

    for name, data in edata.items():
        server.init_data(name=name, data_tensor=data)

    if ndata_g2l is not None:
        for name, data in ndata_g2l.items():
            server.set_global2local(name=name, global2local=data)

    if edata_g2l is not None:
        for name, data in edata_g2l.items():
            server.set_global2local(name=name, global2local=data)

    print("start server %d on %s" % (server.get_id(), server.get_addr()))

    server.start()


def start_client(ip_config, ndata_partition_book, edata_partition_book, close_shared_mem=False, msg_queue_size=2*1024*1024*1024):
    """Start a kvclient node.

    Parameters
    ----------
    ip_config : str
        Filename of server IP configure file.
    ndata_partition_book : dict of tensor (mx.ndarray or torch.tensor)
        Data mapping of node ID to server ID
    edata_partition_book : dict of tensor (mx.ndarray or torch.tensor)
        Data mapping of edge ID to server ID
    close_shared_mem : bool
        Close local shared-memory tensor access.
    msg_queue_size : int
        Size of message queue (2GB by default)

    Returns
    -------
    KVClient
        client handle
    """
    assert len(ip_config) > 0, 'ip_config cannot be empty.'
    assert len(ndata_partition_book) > 0, 'ndata_partition_book cannot be empty.'
    assert len(edata_partition_book) > 0, 'edata_partition_book cannot be empty.'

    server_namebook = read_ip_config(ip_config)

    client = KVClient(server_namebook=server_namebook, close_shared_mem=close_shared_mem, msg_queue_size=msg_queue_size)

    for name, data in ndata_partition_book.items():
        client.set_partition_book(name=name, partition_book=data)

    for name, data in edata_partition_book.items():
        client.set_partition_book(name=name, partition_book=data)

    client.connect()

    print("Client %d (%s) connected to kvstore ..." % (client.get_id(), client.get_addr()))

    return client
    

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large graph features or graph embeddings 
    across machines in a distributed setting. User can re-wriite _push_handler and _pull_handler 
    to support flexibale algorithms.

    Note that, DO NOT use KVServer in multiple threads on Python because this behavior is not defined.

    For now, KVServer can only run in CPU, and we will support GPU KVServer in the future.

    Parameters
    ----------
    server_id : int
        ID of current kvserver node (start from 0).
    server_addr : str
        IP address and port of current KVServer node, e.g., '127.0.0.1:50051'.
    num_client : int
        Total number of clients connecting to server.
    msg_queue_size : int
        Size of message queue (2GB by default)
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi' (do not support yet).
    """
    def __init__(self, server_id, server_addr, num_client, msg_queue_size=2 * 1024 * 1024 * 1024, net_type='socket'):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert len(server_addr.split(':')) == 2, 'Incorrect IP format: %s' % server_addr
        assert num_client >= 0, 'num_client (%d) cannot be a negative number.' % num_client
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type

        # check if target data has been initialized
        self._has_data = set()
        # Store the tensor data with data name
        self._data_store = {}
        # Used for barrier() API on KVClient
        self._barrier_count = 0
        # Server ID starts from zero
        self._server_id = server_id
        self._addr = server_addr
        # client_namebook will be received from client nodes
        self._client_namebook = {}
        self._client_count = num_client
        # Create C communicator of sender and receiver
        self._sender = _create_sender(net_type, msg_queue_size)
        self._receiver = _create_receiver(net_type, msg_queue_size)
        # A naive garbage collocetion for kvstore
        self._garbage_msg = []


    def __del__(self):
        """Finalize KVServer
        """
        # Finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)


    def set_global2local(self, name, global2local):
        """Set a data mapping of global ID to local ID.

        Parameters
        ----------
        name : str
            data name
        global2local : list or tensor (mx.ndarray or torch.tensor)
            A data mapping of global ID to local ID. KVStore will use global ID automatically 
            if this global2local is not been set.
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(global2local) > 0, 'global2local cannot be empty.'

        if isinstance(global2local, list):
            global2local = F.tensor(global2local)

        shared_data = empty_shared_mem(name+'-g2l-'+str(self._server_id), True, global2local.shape, 'int64')
        dlpack = shared_data.to_dlpack()
        self._data_store[name+'-g2l-'] = F.zerocopy_from_dlpack(dlpack)
        self._data_store[name+'-g2l-'][:] = global2local[:]
        self._has_data.add(name+'-g2l-')


    def init_data(self, name, data_tensor):
        """Initialize data on KVServer with data name.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(data_tensor) > 0, 'data_tensor cannot be empty.'

        shared_data = empty_shared_mem(name+'-data-'+str(self._server_id), True, data_tensor.shape, 'float32')
        dlpack = shared_data.to_dlpack()
        self._data_store[name+'-data-'] = F.zerocopy_from_dlpack(dlpack)
        self._data_store[name+'-data-'][:] = data_tensor[:]
        self._has_data.add(name+'-data-')


    def get_id(self):
        """Get current server id.

        Return
        ------
        int
            KVServer ID
        """
        return self._server_id


    def get_addr(self):
        """Get current server IP address

        Return
        ------
        str
            IP address
        """
        return self._addr


    def start(self):
        """Start service of KVServer
        """
        # Get connected with all client nodes
        server_ip, server_port = self._addr.split(':')
        _receiver_wait(self._receiver, server_ip, int(server_port), self._client_count)

        # recv client addr and assign ID for clients
        addr_list = []
        for i in range(self._client_count):
            msg = _recv_kv_msg(self._receiver)
            assert msg.type == KVMsgType.IP_ID
            addr_list.append(msg.name)

        self._sort_addr(addr_list)
        for ID in range(len(addr_list)):
            self._client_namebook[ID] = addr_list[ID]

        _network_wait()

        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)

        _sender_connect(self._sender)

        if self._server_id == 0:
            # assign ID to client nodes
            for client_id, addr in self._client_namebook.items():
                msg = KVStoreMsg(
                    type=KVMsgType.IP_ID,
                    rank=self._server_id,
                    name=str(client_id),
                    id=None,
                    data=None,
                    c_ptr=None)
                _send_kv_msg(self._sender, msg, client_id)

            # send serilaized shared-memory tensor information to clients
            shared_tensor = ''
            for name in self._has_data:
                shared_tensor += self._serialize_shared_tensor(
                    name, 
                    F.shape(self._data_store[name]), 
                    F.dtype(self._data_store[name]))

                shared_tensor += '|'

            msg = KVStoreMsg(
                type=KVMsgType.IP_ID,
                rank=self._server_id,
                name=shared_tensor,
                id=None,
                data=None,
                c_ptr=None)

            for client_id in range(len(self._client_namebook)):
                _send_kv_msg(self._sender, msg, client_id)

        # Service loop
        while True:
            msg = _recv_kv_msg(self._receiver)
            # PUSH message
            if msg.type == KVMsgType.PUSH:
                if (msg.name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[msg.name+'-g2l-'][msg.id]
                else:
                    local_id = msg.id
                self._push_handler(msg.name+'-data-', local_id, msg.data, self._data_store)
            # PULL message
            elif msg.type == KVMsgType.PULL:
                if (msg.name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[msg.name+'-g2l-'][msg.id]
                else:
                    local_id = msg.id
                res_tensor = self._pull_handler(msg.name+'-data-', local_id, self._data_store)
                back_msg = KVStoreMsg(
                    type=KVMsgType.PULL_BACK,
                    rank=self._server_id,
                    name=msg.name,
                    id=msg.id,
                    data=res_tensor,
                    c_ptr=None)
                _send_kv_msg(self._sender, back_msg, msg.rank)
            # Barrier message
            elif msg.type == KVMsgType.BARRIER:
                self._barrier_count += 1
                if self._barrier_count == self._client_count:
                    back_msg = KVStoreMsg(
                        type=KVMsgType.BARRIER,
                        rank=self._server_id,
                        name=None,
                        id=None,
                        data=None,
                        c_ptr=None)
                    for i in range(self._client_count):
                        _send_kv_msg(self._sender, back_msg, i)
                    self._barrier_count = 0
            # FINAL message
            elif msg.type == KVMsgType.FINAL:
                print("Exit KVStore service, server ID: %d" % self._server_id)
                break # exit loop
            else:
                raise RuntimeError('Unknown type of kvstore message: %d' % msg.type.value)

            self._garbage_msg.append(msg)
            if len(self._garbage_msg) > 1000:
                _clear_kv_msg(self._garbage_msg)
                self._garbage_msg = []


    def _push_handler(self, name, ID, data, target):
        """Default handler for PUSH message. 

        On default, _push_handler perform SET operation on the target tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list.
        data : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of id
        target : dict of data
            self._data_store
        """
        target[name][ID] = data


    def _pull_handler(self, name, ID, target):
        """Default handler for PULL operation.

        On default, _pull_handler perform index select operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list.
        target : dict of data
            self._data_store

        Return
        ------
        tensor
            a tensor with the same row size of ID.
        """
        return target[name][ID]


    def _serialize_shared_tensor(self, name, shape, dtype):
        """Serialize shared tensor

        Parameters
        ----------
        name : str
            tensor name
        shape : tuple of int
            tensor shape
        dtype : str
            data type

        Returns
        -------
        str
            serialized string
        """
        str_data = name
        str_data += '/'
        for s in shape:
            str_data += str(s)
            str_data += '/'
        if 'float32' in str(dtype):
            str_data += 'float32'
        elif 'int64' in str(dtype):
            str_data += 'int64'
        else:
            raise RuntimeError('We can only process int64 and float32 shared-memory tensor now.')

        return str_data


    def _sort_addr(self, addr_list):
        """Sort client address list

        Parameters
        ----------
        addr_list : list of str
            IP address list
        """
        return addr_list.sort()


class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer. If one server node and one client node
    are on the same machine, they can commuincated using shared-memory tensor (close_shared_mem=False), 
    instead of TCP/IP connections.

    Note that, DO NOT use KVClient in multiple threads on Python because this behavior is not defined.

    For now, KVClient can only run in CPU, and we will support GPU KVClient in the future.

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's IP address and port, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
    close_shared_mem : bool
        DO NOT use shared-memory access on local machine.
    msg_queue_size : int
        Size of message queue (2GB by default).
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, server_namebook, close_shared_mem=False,  msg_queue_size=2 * 1024 * 1024 * 1024, net_type='socket'):
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type

        if close_shared_mem == True:
            print("The shared-memory tensor has been closed, all data connections will go through TCP/IP network.")

        # check if target data has a ID mapping for global ID to local ID
        self._has_data = set()
        # This is used to store local data, which can share memory with local KVServer.
        self._data_store = {}
        # This is used to check if we can access server data locally
        self._local_server_id = set()
        # Server information
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._close_shared_mem = close_shared_mem
        # client ID will be assign by server after connecting to server
        self._client_id = -1
        # create C communicator of sender and receiver
        self._sender = _create_sender(net_type, msg_queue_size)
        self._receiver = _create_receiver(net_type, msg_queue_size)
        # A naive garbage collocetion for kvstore
        self._garbage_msg = []



    def __del__(self):
        """Finalize KVClient
        """
        # finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)


    def set_partition_book(self, name, partition_book):
        """Set partition book for KVClient. 

        Using partition book, client can know the corresponded server ID of each data.

        Parameters
        ----------
        name : str
            data name
        partition_book : list or tensor (mx.ndarray or torch.tensor)
            A book that maps global ID to target server ID.
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(partition_book) > 0, 'partition_book cannot be empty.'

        if isinstance(partition_book, list):
            self._data_store[name+'-part-'] = F.tensor(partition_book)
        else:
            self._data_store[name+'-part-'] = partition_book

        self._has_data.add(name+'-part-')


    def connect(self):
        """Connect to all the KVServer nodes
        """
        for ID, addr in self._server_namebook.items():
            server_ip, server_port = addr.split(':')
            _add_receiver_addr(self._sender, server_ip, int(server_port), ID)
        _sender_connect(self._sender)

        self._addr = self._get_local_addr()
        client_ip, client_port = self._addr.split(':')

        # find local server nodes
        for ID, addr in self._server_namebook.items():
            server_ip, server_port = addr.split(':')
            if server_ip in self._ip4_addr_list():
                self._local_server_id.add(ID)

        # send addr to server nodes
        msg = KVStoreMsg(
            type=KVMsgType.IP_ID,
            rank=0,
            name=self._addr,
            id=None,
            data=None,
            c_ptr=None)

        for server_id in range(self._server_count):
            _send_kv_msg(self._sender, msg, server_id)

        _receiver_wait(self._receiver, client_ip, int(client_port), self._server_count)

        # recv client id
        msg = _recv_kv_msg(self._receiver)
        assert msg.rank == 0
        self._client_id = int(msg.name)

        # recv name of shared tensor from server 0
        msg = _recv_kv_msg(self._receiver)
        assert msg.rank == 0
        data_str = msg.name.split('|')
        # open shared tensor on local machine
        for data in data_str:
            if data != '' and self._close_shared_mem == False:
                tensor_name, shape, dtype = self._deserialize_shared_tensor(data)
                for server_id in self._local_server_id:
                    shared_data = empty_shared_mem(tensor_name+str(server_id), False, shape, dtype)
                    dlpack = shared_data.to_dlpack()
                    self._data_store[tensor_name+str(server_id)] = F.zerocopy_from_dlpack(dlpack)
                    self._has_data.add(tensor_name+str(server_id))


    def push(self, name, id_tensor, data_tensor):
        """Push message to KVServer.

        Note that push() is an async operation that will return immediately after calling.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor (mx.ndarray or torch.tensor)
            a vector storing the global data ID
        data_tensor : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of data ID
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        assert F.shape(id_tensor)[0] == F.shape(data_tensor)[0], 'The data must has the same row size with ID.'

        # partition data (we can move this part of code into C-api if needed)
        server_id = self._data_store[name+'-part-'][id_tensor]
        # sort index by server id
        sorted_id = F.tensor(np.argsort(F.asnumpy(server_id)))
        id_tensor = id_tensor[sorted_id]
        data_tensor = data_tensor[sorted_id]
        server, count = np.unique(F.asnumpy(server_id), return_counts=True)
        # push data to server by order
        start = 0
        for idx in range(len(server)):
            end = start + count[idx]
            if start == end: # don't have any data for target server
                continue
            partial_id = id_tensor[start:end]
            partial_data = data_tensor[start:end]

            if server[idx] in self._local_server_id and self._close_shared_mem == False:
                if (name+'-g2l-'+str(server[idx]) in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'+str(server[idx])][partial_id]
                else:
                    local_id = partial_id
                self._push_handler(name+'-data-'+str(server[idx]), local_id, data_tensor, self._data_store)
            else:
                msg = KVStoreMsg(
                    type=KVMsgType.PUSH, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    data=partial_data,
                    c_ptr=None)
                _send_kv_msg(self._sender, msg, server[idx])

            start += count[idx]


    def pull(self, name, id_tensor):
        """Pull message from KVServer.

        Parameters 
        ----------
        name : str
            data name
        id_tensor : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list

        Returns
        -------
        tensor
            a data tensor with the same row size of id_tensor.
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'

        if len(self._garbage_msg) > 1000:
            _clear_kv_msg(self._garbage_msg)
            self._garbage_msg = []

        # partition data (we can move this part of code into C-api if needed)
        server_id = self._data_store[name+'-part-'][id_tensor]
        # sort index by server id
        sorted_id = np.argsort(F.asnumpy(server_id))
        # we need return data with original order of ID
        back_sorted_id = F.tensor(np.argsort(sorted_id))
        id_tensor = id_tensor[F.tensor(sorted_id)]
        server, count = np.unique(F.asnumpy(server_id), return_counts=True)
        # pull data from server by server order
        start = 0
        pull_count = 0
        local_data = {}
        for idx in range(len(server)):
            end = start + count[idx]
            if start == end:  # don't have any data in target server
                continue
            partial_id = id_tensor[start:end]

            if server[idx] in self._local_server_id and self._close_shared_mem == False:
                if (name+'-g2l-'+str(server[idx]) in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'+str(server[idx])][partial_id]
                else:
                    local_id = partial_id
                local_data[server[idx]] = self._pull_handler(name+'-data-'+str(server[idx]), local_id, self._data_store)
            else:
                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=partial_id, 
                    data=None,
                    c_ptr=None)
                _send_kv_msg(self._sender, msg, server[idx])
                pull_count += 1

            start += count[idx]

        msg_list = []
        for server_id, data in local_data.items():
            local_msg = KVStoreMsg(
                type=KVMsgType.PULL_BACK, 
                rank=server_id, 
                name=name, 
                id=None,
                data=data,
                c_ptr=None)
            msg_list.append(local_msg)
            self._garbage_msg.append(local_msg)

        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            msg_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)

        # sort msg by server id
        msg_list.sort(key=self._takeId)
        data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)

        return data_tensor[back_sorted_id] # return data with original index order


    def barrier(self):
        """Barrier for all client nodes

        This API will be blocked untill all the clients call this API.
        """
        msg = KVStoreMsg( 
            type=KVMsgType.BARRIER,
            rank=self._client_id,
            name=None,
            id=None,
            data=None,
            c_ptr=None)

        for server_id in range(self._server_count):
            _send_kv_msg(self._sender, msg, server_id)

        for server_id in range(self._server_count):
            back_msg = _recv_kv_msg(self._receiver)
            assert back_msg.type == KVMsgType.BARRIER, 'Recv kv msg error.'


    def shut_down(self):
        """Shut down all KVServer nodes.

        We usually invoke this API by just one client (e.g., client_0).
        """
        for server_id in range(self._server_count):
            msg = KVStoreMsg(
                type=KVMsgType.FINAL,
                rank=self._client_id,
                name=None,
                id=None,
                data=None,
                c_ptr=None)
            _send_kv_msg(self._sender, msg, server_id)


    def get_id(self):
        """Get client id

        Return
        ------
        int
            KVClient ID
        """
        return self._client_id


    def get_addr(self):
        """Get client IP address

        Return
        ------
        str
            IP address
        """
        return self._addr


    def _get_local_addr(self):
        """Get local available IP and port

        Return
        ------
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


    def _get_ip_address(self, NICname):
        """Return IP by given a NIC name
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            struct.pack('256s', NICname[:15].encode("UTF-8"))
        )[20:24])


    def _ip4_addr_list(self):
        """Return a set of IPv4 address
        """
        nic = set()

        for ix in socket.if_nameindex():
            name = ix[1]
            ip = self._get_ip_address(name)

            nic.add(ip)

        return nic


    def _takeId(self, elem):
        """Used by sort
        """
        return elem.rank


    def _push_handler(self, name, ID, data, target):
        """Default handler for local PUSH message. 

        On default, _push_handler perform SET operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list.
        data : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of id
        target : tensor (mx.ndarray or torch.tensor)
            the target tensor
        """
        target[name][ID] = data


    def _pull_handler(self, name, ID, target):
        """Default handler for local PULL operation.

        On default, _pull_handler perform index select operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs that has been re-mapped to local id.
        target : tensor (mx.ndarray or torch.tensor)
            the target tensor

        Return
        ------
        tensor
            a tensor with the same row size of ID
        """
        return target[name][ID]
    

    def _deserialize_shared_tensor(self, data):
        """Deserialize shared tensor information sent from server

        Parameters
        ----------
        data : str
            serialized string

        Returns
        -------
        str
            tensor name
        tuple of int
            tensor shape
        str
            data type
        """
        data_list = data.split('/')
        tensor_name = data_list[0]
        data_type = data_list[-1]
        tensor_shape = []
        for i in range(1, len(data_list)-1):
            tensor_shape.append(int(data_list[i]))
        tensor_shape = tuple(tensor_shape)

        return tensor_name, tensor_shape, data_type

