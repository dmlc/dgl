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
import time
import random
import numpy as np
import socket

if os.name != 'nt':
    import fcntl
    import struct


def read_ip_config(filename):
    """Read network configuration information of kvstore from file.

    The format of configuration file should be:

        [ip] [base_port] [server_count]

        172.31.40.143 30050 2
        172.31.36.140 30050 2
        172.31.47.147 30050 2
        172.31.30.180 30050 2

    Note that, DGL KVStore supports multiple servers that can share data with each other 
    on the same machine via shared-tensor. So the server_count should be >= 1.

    Parameters
    ----------
    filename : str
        name of configuration file.

    Returns
    -------
    dict
        server namebook. e.g.,

        [server_id]:[machine_id, ip, port, group_count]

          {0:'[0, 172.31.40.143, 30050, 2],
           1:'[0, 172.31.40.143, 30051, 2],
           2:'[1, 172.31.36.140, 30050, 2],
           3:'[1, 172.31.36.140, 30051, 2],
           4:'[2, 172.31.47.147, 30050, 2],
           5:'[2, 172.31.47.147, 30051, 2],
           6:'[3, 172.31.30.180, 30050, 2],
           7:'[3, 172.31.30.180, 30051, 2]}
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


class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or 
    graph embeddings across machines in a distributed setting. Also, user can re-wriite _push_handler() 
    and _pull_handler() API to support flexibale algorithms.

    DGL kvstore supports multiple-servers on single-machine. That means we can lunach many servers on the same machine and all of 
    these servers will share the same shared-memory tensor for load-balance.

    Note that, DO NOT use KVServer in multiple threads on Python because this behavior is not defined.

    For now, KVServer can only run in CPU. We will support GPU KVServer in the future.

    Parameters
    ----------
    server_id : int
        KVServer's ID (start from 0).
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's machine_id, IP address and port, e.g.,

          {0:'[0, 172.31.40.143, 30050],
           1:'[0, 172.31.40.143, 30051],
           2:'[1, 172.31.36.140, 30050],
           3:'[1, 172.31.36.140, 30051],
           4:'[2, 172.31.47.147, 30050],
           5:'[2, 172.31.47.147, 30051],
           6:'[3, 172.31.30.180, 30050],
           7:'[3, 172.31.30.180, 30051]}

    num_client : int
        Total number of client nodes.
    queue_size : int
        Sise (bytes) of kvstore message queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound number and DGL will not allocate 20GB memory.
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi' (do not support yet).
    """
    def __init__(self, server_id, server_namebook, num_client, queue_size=20*1024*1024*1024, net_type='socket'):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert num_client >= 0, 'num_client (%d) cannot be a negative number.' % num_client
        assert queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type

        # check if target data has been initialized
        self._has_data = set()
        # Store the tensor data with specified data name
        self._data_store = {}
        # Used for barrier() API on KVClient
        self._barrier_count = 0
        # Server information
        self._server_id = server_id
        self._server_namebook = server_namebook
        self._machine_id = server_namebook[server_id][0]
        self._ip = server_namebook[server_id][1]
        self._port = server_namebook[server_id][2]
        self._group_count = server_namebook[server_id][3]
        # client_namebook will be sent from remote client nodes
        self._client_namebook = {}
        self._client_count = num_client
        # Create C communicator of sender and receiver
        self._sender = _create_sender(net_type, queue_size)
        self._receiver = _create_receiver(net_type, queue_size)
        # Delete temp file when kvstore service is closed
        self._open_file_list = []
        # record for total message count
        self._msg_count = 0


    def __del__(self):
        """Finalize KVServer
        """
        # Finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)
        # Delete temp file when kvstore service is closed
        for file in self._open_file_list:
            if (os.path.exists(file)):
                os.remove(file)


    def set_global2local(self, name, global2local=None):
        """Set data mapping of global ID to local ID.

        Parameters
        ----------
        name : str
            data name
        global2local : list or tensor (mx.ndarray or torch.tensor)
            A data mapping of global ID to local ID. KVStore will use global ID by default 
            if the global2local is not been set.

            Note that, if the global2local is None KVServer will read shared-tensor.
        """
        assert len(name) > 0, 'name cannot be empty.'

        if global2local is not None: # Create shared-tensor
            if isinstance(global2local, list):
                global2local = F.tensor(global2local)
            shared_data = empty_shared_mem(name+'-g2l-', True, global2local.shape, 'int64')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-g2l-'] = F.zerocopy_from_dlpack(dlpack)
            self._data_store[name+'-g2l-'][:] = global2local[:]
            # write data information to temp file that can be read by other processes
            self._write_data_shape(name+'-g2l-shape', global2local)
            self._open_file_list.append(name+'-g2l-shape')
        else: # Read shared-tensor
            while True:
                if (os.path.exists(name+'-g2l-shape')):
                    time.sleep(2) # wait writing finish
                    break
                else:
                    time.sleep(2) # wait until the file been created
            data_shape = self._read_data_shape(name+'-g2l-shape')
            shared_data = empty_shared_mem(name+'-g2l-', False, data_shape, 'int64')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-g2l-'] = F.zerocopy_from_dlpack(dlpack)

        self._has_data.add(name+'-g2l-')


    def init_data(self, name, data_tensor=None):
        """Initialize data tensor on KVServe.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor (mx.ndarray or torch.tensor)
            data tensor

            Note that, if the data_tensor is None KVServer will read shared-tensor.
        """
        assert len(name) > 0, 'name cannot be empty.'

        if data_tensor is not None: # Create shared-tensor
            shared_data = empty_shared_mem(name+'-data-', True, data_tensor.shape, 'float32')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-data-'] = F.zerocopy_from_dlpack(dlpack)
            self._data_store[name+'-data-'][:] = data_tensor[:]
            self._write_data_shape(name+'-data-shape', data_tensor)
            self._open_file_list.append(name+'-data-shape')
        else: # Read shared-tensor
            while True:
                if (os.path.exists(name+'-data-shape')):
                    break
                else:
                    time.sleep(2) # wait until the file been created
            data_shape = self._read_data_shape(name+'-data-shape')
            shared_data = empty_shared_mem(name+'-data-', False, data_shape, 'float32')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-data-'] = F.zerocopy_from_dlpack(dlpack)

        self._has_data.add(name+'-data-')


    def get_id(self):
        """Get current server id

        Return
        ------
        int
            KVServer ID
        """
        return self._server_id


    def get_addr(self):
        """Get current server IP address and port

        Return
        ------
        str
            IP address and port
        """
        return self._ip + ':' + str(self._port)


    def get_machine_id(self):
        """Get local machine ID

        Return
        -------
        int
            machine ID
        """
        return self._machine_id


    def get_group_count(self):
        """Get count of server inside a machine

        Return
        ------
        int
            count of server
        """
        return self._group_count


    def get_message_count(self):
        """Get total message count on current KVServer

        Return
        ------
        int
            count of message
        """
        return self._msg_count


    def print(self):
        """Print server information (Used by debug)
        """
        print("----- KVStore Info -----")
        print("server id: %d" % self.get_id())
        print("data:")
        for name, data in self._data_store.items():
            print(name)
            print(data)
        print("------------------------")


    def start(self):
        """Start service of KVServer.

        The start() api performs the following things:

          1. Get connected with all client nodes.
          2. Recv client address information.
          3. assign client ID to each client node.
          4. send shared-tensor information to each client node.
          5. Service loop for listening requests from client nodes.

        """
        # Get connected with all client nodes
        _receiver_wait(self._receiver, self._ip, self._port, self._client_count)

        # recv client address information
        addr_list = []
        for i in range(self._client_count):
            msg = _recv_kv_msg(self._receiver)
            assert msg.type == KVMsgType.IP_ID
            addr_list.append(msg.name)

        # Assign client ID to each client node
        addr_list.sort()
        for ID in range(len(addr_list)):
            self._client_namebook[ID] = addr_list[ID]

        _network_wait()

        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)

        _sender_connect(self._sender)

        if self._server_id == 0:
            for client_id in range(len(self._client_namebook)):
                msg = KVStoreMsg(
                    type=KVMsgType.IP_ID,
                    rank=self._server_id,
                    name=str(client_id),
                    id=None,
                    data=None,
                    c_ptr=None)
                _send_kv_msg(self._sender, msg, client_id)

        # Send shared-tensor information to each client node
        if self._server_id == 0:
            shared_tensor = ''
            for name in self._has_data:
                shared_tensor += self._serialize_shared_tensor(
                    name, F.dtype(self._data_store[name]))
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

        print('KVStore service %d start successfully! Listen for request ...' % self.get_id())

        # Service loop
        while True:
            msg = _recv_kv_msg(self._receiver)
            # Push message
            if msg.type == KVMsgType.PUSH:
                if (msg.name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[msg.name+'-g2l-'][msg.id]
                else:
                    local_id = msg.id
                self._push_handler(msg.name+'-data-', local_id, msg.data, self._data_store)
            # Pull message
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
                    for client_id in range(self._client_count):
                        _send_kv_msg(self._sender, back_msg, client_id)
                    self._barrier_count = 0  
            # Final message              
            elif msg.type == KVMsgType.FINAL:
                print("Exit KVStore service %d, solved message count: %d" % (self.get_id(), self.get_message_count()))
                break # exit loop
            else:
                raise RuntimeError('Unknown type of kvstore message: %d' % msg.type.value)

            _clear_kv_msg(msg)

            self._msg_count += 1


    def _serialize_shared_tensor(self, name, dtype):
        """Serialize shared tensor information.

        Parameters
        ----------
        name : str
            tensor name
        dtype : str
            data type

        Returns
        -------
        str
            serialized string
        """
        assert len(name) > 0, 'data name cannot be empty.'

        str_data = name
        str_data += '/'
        if 'float32' in str(dtype):
            str_data += 'float32'
        elif 'int64' in str(dtype):
            str_data += 'int64'
        else:
            raise RuntimeError('We can only process int64 and float32 shared-memory tensor now.')

        return str_data


    def _write_data_shape(self, filename, data):
        """Write data shape to a temp file.

        Parameters
        ----------
        filename : str
            name of temp file.
        data : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        assert len(filename) > 0, 'filename cannot be empty.'

        if(os.path.exists(filename)):
            os.remove(filename)

        shape = F.shape(data)
        str_data = ''
        f = open(filename, "a");
        for s in shape:
            str_data += str(s)
            str_data += '|'
        f.write(str_data)
        f.close()


    def _read_data_shape(self, filename):
        """Read data shape from a tmp file.

        Parameters
        ----------
        filename : str
            name of temp file

        Return
        ------
        tuple
            data shape
        """
        assert len(filename) > 0, 'filename cannot be empty.'

        f = open(filename, "r")
        str_data = f.read()
        data_list = str_data.split('|')
        data_shape = []
        for i in range(len(data_list)-1):
            data_shape.append(int(data_list[i]))
        f.close()

        return data_shape


    def _push_handler(self, name, ID, data, target):
        """Default handler for PUSH message. 

        On default, _push_handler perform update operation for the tensor.

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

        On default, _pull_handler perform get operation for the tensor.

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


class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer. If the server node and client node are on the 
    same machine, they can commuincate with each other using local shared-memory tensor, instead of TCP/IP connections.

    Note that, DO NOT use KVClient in multiple threads on Python because this behavior is not defined.

    For now, KVClient can only run in CPU, and we will support GPU KVClient in the future.

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's machine_id, IP address and port, and group_count, e.g.,

          {0:'[0, 172.31.40.143, 30050, 2],
           1:'[0, 172.31.40.143, 30051, 2],
           2:'[1, 172.31.36.140, 30050, 2],
           3:'[1, 172.31.36.140, 30051, 2],
           4:'[2, 172.31.47.147, 30050, 2],
           5:'[2, 172.31.47.147, 30051, 2],
           6:'[3, 172.31.30.180, 30050, 2],
           7:'[3, 172.31.30.180, 30051, 2]}

    queue_size : int
        Sise (bytes) of kvstore message queue buffer (~20 GB on default).
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, server_namebook, queue_size=20*1024*1024*1024, net_type='socket'):
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert queue_size > 0, 'queue_size (%d) cannot be a negative number.' % queue_size
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type

        # check if target data has been initialized
        self._has_data = set()
        # This is used to store local data, which can share memory with local KVServer.
        self._data_store = {}
        # Server information
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._group_count = server_namebook[0][3]
        # client ID will be assign by server after connecting to server
        self._client_id = -1
        # Get local machine id via server_namebook
        self._machine_id = self._get_local_machine_id()
        # create C communicator of sender and receiver
        self._sender = _create_sender(net_type, queue_size)
        self._receiver = _create_receiver(net_type, queue_size)
        # Delete temp file when kvstore service is closed
        self._open_file_list = []
        # Gargage_collection
        self._garbage_msg = []
        # Used load-balance
        random.seed(time.time())


    def __del__(self):
        """Finalize KVClient
        """
        # finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)
        # Delete temp file whhen kvstore service is closed
        for file in self._open_file_list:
            if(os.path.exists(file)):
                os.remove(file)


    def set_partition_book(self, name, partition_book=None):
        """Partition book contains the data mapping of global ID to machine ID.

        Parameters
        ----------
        name : str
            data name
        partition_book : list or tensor (mx.ndarray or torch.tensor)
            Mapping global ID to target machine ID.

        Note that, if the partition_book is None KVClient will read shared-tensor by name.
        """
        assert len(name) > 0, 'name connot be empty.'

        if partition_book is not None: # Create shared-tensor
            if isinstance(partition_book, list):
                partition_book = F.tensor(partition_book)
            shared_data = empty_shared_mem(name+'-part-', True, partition_book.shape, 'int64')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-part-'] = F.zerocopy_from_dlpack(dlpack)
            self._data_store[name+'-part-'][:] = partition_book[:]
            self._write_data_shape(name+'-part-shape', partition_book)
            self._open_file_list.append(name+'-part-shape')
        else: # Read shared-tensor
            while True:
                if (os.path.exists(name+'-part-shape')):
                    time.sleep(2) # wait writing finish
                    break
                else:
                    time.sleep(2) # wait until the file been created    
            data_shape = self._read_data_shape(name+'-part-shape')
            shared_data = empty_shared_mem(name+'-part-', False, data_shape, 'int64')
            dlpack = shared_data.to_dlpack()
            self._data_store[name+'-part-'] = F.zerocopy_from_dlpack(dlpack)

        self._has_data.add(name+'-part-')


    def connect(self):
        """Connect to all the KVServer nodes

        The connect() api performs the following things:

          1. Get connected with all server nodes.
          2. Send client address information to server.
          3. Recv client ID from server.
          4. Recv shared-tensor information from server.

        """
        # Get connected with all server nodes
        for ID, addr in self._server_namebook.items():
            server_ip = addr[1]
            server_port = addr[2]
            _add_receiver_addr(self._sender, server_ip, server_port, ID)
        _sender_connect(self._sender)

        # Send client address to server nodes
        self._addr = self._get_local_usable_addr()
        client_ip, client_port = self._addr.split(':')

        msg = KVStoreMsg(
            type=KVMsgType.IP_ID,
            rank=0, # a tmp client ID
            name=self._addr,
            id=None,
            data=None,
            c_ptr=None)

        for server_id in range(self._server_count):
            _send_kv_msg(self._sender, msg, server_id)

        _receiver_wait(self._receiver, client_ip, int(client_port), self._server_count)

        # Recv client ID from server
        msg = _recv_kv_msg(self._receiver)
        assert msg.rank == 0
        self._client_id = int(msg.name)

        # Recv shared-tensor information from server
        msg = _recv_kv_msg(self._receiver)
        assert msg.rank == 0
        data_str = msg.name.split('|')
        for data in data_str:
            if data != '':
                tensor_name, dtype = self._deserialize_shared_tensor(data)
                while True:
                    if (os.path.exists(tensor_name+'shape')):
                        time.sleep(2) # wait writing finish
                        break
                    else:
                        time.sleep(2) # wait until the file been created 
                shape = self._read_data_shape(tensor_name+'shape')
                shared_data = empty_shared_mem(tensor_name, False, shape, dtype)
                dlpack = shared_data.to_dlpack()
                self._data_store[tensor_name] = F.zerocopy_from_dlpack(dlpack)
                self._has_data.add(tensor_name)

        print("KVClient %d connect to kvstore successfully!" % self.get_id())


    def print(self):
        """Print client information (Used by debug)
        """
        print("----- KVClient Info -----")
        print("client id: %d" % self.get_id())
        print("data:")
        for name, data in self._data_store.items():
            print(name)
            print(data)
        print("-------------------------")


    def get_id(self):
        """Get current client id

        Return
        ------
        int
            KVClient ID
        """
        return self._client_id


    def get_addr(self):
        """Get current client IP address

        Return
        ------
        str
            IP address
        """
        return self._addr


    def get_machine_id(self):
        """Get local machine ID

        Return
        -------
        int
            machine ID
        """
        return self._machine_id


    def push(self, name, id_tensor, data_tensor):
        """Push data to KVServer.

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

        # partition data
        machine_id = self._data_store[name+'-part-'][id_tensor]
        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        id_tensor = id_tensor[sorted_id]
        data_tensor = data_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        # push data to server by order
        start = 0
        local_id = None
        local_data = None
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            partial_data = data_tensor[start:end]
            if machine[idx] == self._machine_id: # local push
                # Note that DO NOT push local data right now because we can overlap
                # communication-local_push here
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id]
                else:
                    local_id = partial_id
                local_data = partial_data
            else: # push data to remote server
                msg = KVStoreMsg(
                    type=KVMsgType.PUSH, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    data=partial_data,
                    c_ptr=None)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)

            start += count[idx]

        if local_id is not None: # local push
            self._push_handler(name+'-data-', local_id, local_data, self._data_store)
    

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

        for msg in self._garbage_msg:
            _clear_kv_msg(msg)
        self._garbage_msg = []

        # partition data
        machine_id = self._data_store[name+'-part-'][id_tensor]
        # sort index by machine id
        sorted_id = F.tensor(np.argsort(F.asnumpy(machine_id)))
        back_sorted_id = F.tensor(np.argsort(F.asnumpy(sorted_id)))
        id_tensor = id_tensor[sorted_id]
        machine, count = np.unique(F.asnumpy(machine_id), return_counts=True)
        # pull data from server by order
        start = 0
        pull_count = 0
        local_id = None
        for idx in range(len(machine)):
            end = start + count[idx]
            if start == end: # No data for target machine
                continue
            partial_id = id_tensor[start:end]
            if machine[idx] == self._machine_id: # local pull
                # Note that DO NOT pull local data right now because we can overlap
                # communication-local_pull here
                if (name+'-g2l-' in self._has_data) == True:
                    local_id = self._data_store[name+'-g2l-'][partial_id]
                else:
                    local_id = partial_id
            else: # pull data from remote server
                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=partial_id,
                    data=None,
                    c_ptr=None)
                # randomly select a server node in target machine for load-balance
                s_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                _send_kv_msg(self._sender, msg, s_id)
                pull_count += 1

            start += count[idx]           

        msg_list = []
        if local_id is not None: # local pull
            local_data = self._pull_handler(name+'-data-', local_id, self._data_store)
            s_id = random.randint(self._machine_id*self._group_count, (self._machine_id+1)*self._group_count-1)
            local_msg = KVStoreMsg(
                type=KVMsgType.PULL_BACK, 
                rank=s_id,
                name=name, 
                id=None,
                data=local_data,
                c_ptr=None)
            msg_list.append(local_msg)
            self._garbage_msg.append(local_msg)

        # wait message from server nodes
        for idx in range(pull_count):
            remote_msg = _recv_kv_msg(self._receiver)
            msg_list.append(remote_msg)
            self._garbage_msg.append(remote_msg)

        # sort msg by server id and merge tensor together
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


    def _get_local_usable_addr(self):
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


    def _get_local_machine_id(self):
        """Get local machine ID from server_namebook

        Return
        ------
        int
            local machine ID
        """
        res = 0
        for ID, data in self._server_namebook.items():
            machine_id = data[0]
            ip = data[1]
            if ip in self._local_ip4_addr_list():
                res = machine_id
                break

        return res


    def _local_ip4_addr_list(self):
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
        str
            data type
        """
        data_list = data.split('/')
        tensor_name = data_list[0]
        data_type = data_list[-1]

        return tensor_name, data_type


    def _write_data_shape(self, filename, data):
        """Write data shape to a temp file.

        Parameters
        ----------
        filename : str
            name of temp file.
        data : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        assert len(filename) > 0, 'filename cannot be empty.'

        if(os.path.exists(filename)):
            os.remove(filename)

        shape = F.shape(data)
        str_data = ''
        f = open(filename, "a");
        for s in shape:
            str_data += str(s)
            str_data += '|'
        f.write(str_data)
        f.close()


    def _read_data_shape(self, filename):
        """Read data shape from a tmp file.

        Parameters
        ----------
        filename : str
            name of temp file

        Return
        ------
        tuple
            data shape
        """
        assert len(filename) > 0, 'filename cannot be empty.'

        f = open(filename, "r")
        str_data = f.read()
        data_list = str_data.split('|')
        data_shape = []
        for i in range(len(data_list)-1):
            data_shape.append(int(data_list[i]))

        f.close()

        return data_shape


    def _takeId(self, elem):
        """Used by sort message list
        """
        return elem.rank


    def _push_handler(self, name, ID, data, target):
        """Default handler for PUSH message. 

        On default, _push_handler perform update operation for the tensor.

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

        On default, _pull_handler perform get operation for the tensor.

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
    