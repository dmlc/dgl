# This file contains DGL distributed kvstore APIs.
from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _network_wait, _add_receiver_addr
from ..network import _receiver_wait, _sender_connect
from ..network import _send_kv_msg, _recv_kv_msg
from ..network import KVMsgType, KVStoreMsg

import numpy as np
import dgl.backend as F

def ReadNetworkConfigure(filename, format=1, client_num=1):
    """Read networking configuration from file. The configuration file could be two formats.

    Format 1:

        [server_or_client] [ip:port] [server_id_or_client_id]

        server 172.31.40.143:50050 0
        client 172.31.40.143:50051 0
        client 172.31.36.140:50051 1
        client 172.31.47.147:50051 2
        client 172.31.30.180:50051 3

    For this configuration, we have 1 server node and 4 client nodes.

    Format 2:

        [server_id] [ip] [base_port]

        0 172.31.40.143 50050
        1 172.31.36.140 50050
        2 172.31.47.147 50050
        3 172.31.30.180 50050

    For this configuration, users need to set the client_num (e.g., client_num = 4). In that case,
    we start 1 server node and 4 client nodes on each machine. The '50050' is the base port for server nodes, 
    and the ports of client nodes on each machine are: '50051', '50052', '50053', and '50054'.

    The second format of configuration can help us to launch a large set of clients on each machine.


    Parameters
    ----------
    filename : str
        name of target configure file.
    format : int
        Format of config file, 1 or 2.
    client_num : int
        number of client. Used by format 2.

    Returns
    -------
    dict
        server namebook, 

         e.g., {0:'172.31.40.143:50050'}, for the first config.
    dict
        client namebook, for the first config.

         e.g., {0:'172.31.40.143:50051',
                1:'172.31.36.140:50051',
                2:'172.31.47.147:50051',
                3:'172.31.30.180:50051'}
    """
    assert len(filename) > 0, 'filename cannot be empty.'
    assert client_num > 0, 'client_num (%d) cannot be a negative number.' % client_num

    server_namebook = {}
    client_namebook = {}

    lines = [line.rstrip('\n') for line in open(filename)]
    if format == 1:
        for line in lines:
            node_type, addr, node_id = line.split(' ')
            if node_type == 'server':
                server_namebook[int(node_id)] = addr
            elif node_type == 'client':
                client_namebook[int(node_id)] = addr
            else:
                raise RuntimeError("Unknown node type: %s", node_type)
    elif format == 2:
        base = 0
        for line in lines:
            server_id, addr, port = line.split(' ')
            server_namebook[int(server_id)] = addr + ':' + port
            for i in range(client_num):
                client_namebook[i+base] = addr + ':' + str(int(port)+1+i)
            base += client_num
    else:
        raise RuntimeError('Unknown file format: %d', format)

    return server_namebook, client_namebook


class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or 
    graph embeddings across machines in a distributed setting. User can re-wriite _push_handler 
    and _pull_handler to support flexibale algorithms.

    Note that, DO NOT share one KVServer in multiple threads on python because this behavior is not defined.

    For now, KVServer can only run in CPU, we will support GPU KVServer in the future.

    Parameters
    ----------
    server_id : int
        KVServer's ID (start from 0).
    client_namebook : dict
        IP address namebook of KVClient, where the key is the client's ID 
        (start from 0) and the value is client's IP address and port, e.g.,

            { 0:'168.12.23.45:50051', 
              1:'168.12.23.21:50051', 
              2:'168.12.46.12:50051' }
    server_addr : str
        IP address and port of current KVServer node, e.g., '127.0.0.1:50051'.
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi' (do not support yet).
    """
    def __init__(self, server_id, client_namebook, server_addr, net_type='socket'):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert len(client_namebook) > 0, 'client_namebook cannot be empty.'
        assert len(server_addr.split(':')) == 2, 'Incorrect IP format: %s' % server_addr
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type
        # check if target data (using data name) has been initialized
        self._has_data = set()
        # check if target data (using data name) has a ID mapping for global ID to local ID
        self._has_global_to_local = set()
        # Store the tensor data with data name
        self._data_store = {}
        # Store the ID mapping of data tensor with data name
        self._global_to_local = {} 
        # Used for barrier() API on KVClient
        self._barrier_count = 0
        # Server ID starts from zero
        self._server_id = server_id
        self._addr = server_addr
        self._client_namebook = client_namebook
        self._client_count = len(client_namebook)
        # Create C communicator of sender and receiver
        self._sender = _create_sender(net_type)
        self._receiver = _create_receiver(net_type)


    def __del__(self):
        """Finalize KVServer
        """
        # Finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)


    def set_global_to_local(self, name, global_to_local):
        """Set a data mapping of global ID to local ID with data name.

        Parameters
        ----------
        name : str
            data name
        global_to_local : list or tensor (mx.ndarray or torch.tensor)
            A book mapping of global ID to local ID. 
            KVStore will use global ID if this mapping is not been set.
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(global_to_local) > 0, 'global_to_local cannot be empty.'

        if isinstance(global_to_local, list):
            self._global_to_local[name] = F.tensor(global_to_local)
        else:
            self._global_to_local[name] = global_to_local

        self._has_global_to_local.add(name)


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

        self._data_store[name] = data_tensor
        self._has_data.add(name)


    def get_id(self):
        """Get server id

        Return
        ------
        int
            KVServer ID
        """
        return self._server_id


    def start(self):
        """Start service of KVServer
        """
        server_ip, server_port = self._addr.split(':')
        _receiver_wait(self._receiver, server_ip, int(server_port), self._client_count)
        _network_wait() # wait client's start
        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)
        _sender_connect(self._sender)
        # Service loop
        while True:
            msg = _recv_kv_msg(self._receiver)
            if msg.type == KVMsgType.INIT:
                if (msg.name in self._has_data) == False:
                    # we hack the msg format here:
                    # msg.id store the shape of target tensor
                    # msg.data has two row, and the first row is 
                    # the init_type, [0, 0] means 'zero' and [1,1]
                    # means 'uniform'. The second row is the min & max threshold.
                    data_shape = F.asnumpy(msg.id).tolist()
                    row_0 = (F.asnumpy(msg.data).tolist())[0] 
                    row_1 = (F.asnumpy(msg.data).tolist())[1]
                    init_type = 'zero' if row_0[0] == 0.0 else 'uniform'
                    self._init_data_from_client(
                        name=msg.name,
                        shape=data_shape,
                        init_type=init_type,
                        low=row_1[0],
                        high=row_1[1])
                    self._is_init.add(msg.name)
            elif msg.type == KVMsgType.PUSH:
                if (msg.name in self._has_global_to_local) == True:
                    local_id = self._global_to_local[msg.name][msg.id]
                else:
                    local_id = msg.id
                self._push_handler(msg.name, local_id, msg.data, self._data_store)
            elif msg.type == KVMsgType.PULL:
                if (msg.name in self._has_global_to_local) == True:
                    local_id = self._global_to_local[msg.name][msg.id]
                else:
                    local_id = msg.id
                res_tensor = self._pull_handler(msg.name, local_id, self._data_store)
                back_msg = KVStoreMsg(
                    type=KVMsgType.PULL_BACK,
                    rank=self._server_id,
                    name=msg.name,
                    id=msg.id,
                    data=res_tensor)
                _send_kv_msg(self._sender, back_msg, msg.rank)
            elif msg.type == KVMsgType.BARRIER:
                self._barrier_count += 1
                if self._barrier_count == self._client_count:
                    back_msg = KVStoreMsg(
                        type=KVMsgType.BARRIER,
                        rank=self._server_id,
                        name=None,
                        id=None,
                        data=None)
                    for i in range(self._client_count):
                        _send_kv_msg(self._sender, back_msg, i)
                    self._barrier_count = 0
            elif msg.type == KVMsgType.FINAL:
                print("Exit KVStore service, server ID: %d" % self._server_id)
                break # exit loop
            else:
                raise RuntimeError('Unknown type of kvstore message: %d' % msg.type.value)


    def _init_data_from_client(self, name, shape, init_type, low, high):
        """Initialize data from the message send by client.

        Parameters
        ----------
        name : str
            data name
        shape : list of int
            The shape of tensor
        init_type : str
            initialize method, including 'zero' and 'uniform'
        low : float
            min threshold, if needed
        high : float
            max threshold, if needed
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(shape) > 0, 'shape cannot be empty.'

        if init_type == 'uniform':
            self._data_store[name] = F.uniform(
                shape=shape,
                dtype=F.float32,
                ctx=F.cpu(),
                low=low,
                high=high)
        elif init_type == 'zero':
            self._data_store[name] = F.zeros(
                shape=shape,
                dtype=F.float32,
                ctx=F.cpu())
        else:
            raise RuntimeError('Unknown initial method')


    def _push_handler(self, name, ID, data, target):
        """Default handler for PUSH message. 

        On default, _push_handler perform ADD operation for the tensor.

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
        target[name][ID] += data


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


class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer.

    Note that, DO NOT share one KVClient in multiple threads in python because this behavior is not defined.

    For now, KVClient can only run in CPU, we will support GPU KVClient in the future.

    Parameters
    ----------
    client_id : int
        KVClient's ID (start from 0)
    local_server_id : int
        Server that locates in the same machine with current client node. 
        By using this ID, client knows that how to use shared-memory for accessing server data instead of using tcp/ip network.
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's IP address and port, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
    client_addr : str
        IP address and port of current KVClient, e.g., '168.12.23.22:50051'
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, client_id, local_server_id, server_namebook, client_addr, net_type='socket'):
        assert client_id >= 0, 'client_id (%d) cannot be a nagative number.' % client_id
        assert local_server_id >= 0, 'local_server_id (%d) cannot be a negative number.' % local_server_id
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert len(client_addr.split(':')) == 2, 'Incorrect IP format: %s' % client_addr
        assert net_type == 'socket' or net_type == 'mpi', 'net_type (%s) can only be \'socket\' or \'mpi\'.' % net_type
        # This is used to store local data, which can share memory with local KVServer.
        self._data_store = {}
        # Store the ID mapping of data tensor with data name
        self._global_to_local = {}
        # Store the ID mapping for data ID and server ID
        self._partition_book = {}
        # check if target data (using data name) has a ID mapping for global ID to local ID
        self._has_global_to_local = set()
        # This is used to check if we can access server data locally
        self._local_server_id = local_server_id
        # Client ID starts from zero
        self._client_id = client_id
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._addr = client_addr
        # create C communicator of sender and receiver
        self._sender = _create_sender(net_type)
        self._receiver = _create_receiver(net_type)


    def __del__(self):
        """Finalize KVClient
        """
        # finalize C communicator of sender and receiver
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)


    def set_partition_book(self, name, partition_book):
        """Initialize partition book for KVClient. 
        Using partition book, client can know the corresponding server ID 
        of each element in a target data tensor.

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
            self._partition_book[name] = F.tensor(partition_book)
        else:
            self._partition_book[name] = partition_book


    def set_global_to_local(self, name, global_to_local):
        """Set a data mapping of global ID to local ID with data name.

        Parameters
        ----------
        name : str
            data name
        global_to_local : list or tensor (mx.ndarray or torch.tensor)
            A book mapping of global ID to local ID. 
            KVStore will use global ID if this mapping is not been set.
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(global_to_local) > 0, 'global_to_local cannot be empty.'

        if isinstance(global_to_local, list):
            self._global_to_local[name] = F.tensor(global_to_local)
        else:
            self._global_to_local[name] = global_to_local

        self._has_global_to_local.add(name)  


    def init_data(self, name, data_tensor):
        """Initialize local data on client. This data can be shared with local server nodes.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor (mx.ndarray or torch.tensor)
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(data_tensor) > 0, 'data_tensor cannot be empty.'

        self._data_store[name] = data_tensor


    def init_data_on_server(self, name, shape, init_type='zero', low=0.0, high=0.0):
        """Initialize kvstore tensor

        we hack the msg format here: msg.id store the shape of target tensor,
        msg.data has two row, and the first row is the init_type, 
        [0, 0] means 'zero' and [1,1] means 'uniform'. 
        The second row is the min & max threshold.

        Parameters
        ----------
        name : str
            data name
        shape : list of int
            shape of tensor
        init_type : str
            initialize method, including 'zero' and 'uniform'
        low : float
            min threshold, if use 'uniform'
        high : float
            max threshold, if use 'uniform'
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(shape) > 0, 'shape cannot be empty.'
        assert init_type == 'zero' or net_type == 'uniform', 'init_type (%s) can only be \'zero\' or \'uniform\'.' % init_type

        tensor_shape = F.tensor(shape)
        init_type = 0.0 if init_type == 'zero' else 1.0
        threshold = F.tensor([[init_type, init_type], [low, high]])

        msg = KVStoreMsg(
            type=KVMsgType.INIT,
            rank=self._client_id,
            name=name,
            id=tensor_shape,
            data=threshold)

        for server_id in range(self._server_count):
            _send_kv_msg(self._sender, msg, server_id)

    
    def connect(self):
        """Connect to all the KVServer nodes
        """
        for ID, addr in self._server_namebook.items():
            server_ip, server_port = addr.split(':')
            _add_receiver_addr(self._sender, server_ip, int(server_port), ID)
        _sender_connect(self._sender)
        client_ip, client_port = self._addr.split(':')
        _receiver_wait(self._receiver, client_ip, int(client_port), self._server_count)


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
        server_id = self._partition_book[name][id_tensor]
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
            if server[idx] == self._local_server_id:  # local push
                if (name in self._has_global_to_local) == True:
                    local_id = self._global_to_local[name][partial_id]
                else:
                    local_id = partial_id
                self._local_push_handler(name, local_id, partial_data, self._data_store)
            else: # push data to remote server
                msg = KVStoreMsg(
                    type=KVMsgType.PUSH, 
                    rank=self._client_id, 
                    name=name,
                    id=partial_id, 
                    data=partial_data)
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

        # partition data (we can move this part of code into C-api if needed)
        server_id = self._partition_book[name][id_tensor]
        # sort index by server id
        sorted_id = np.argsort(F.asnumpy(server_id))
        # we need return data with original order of ID
        back_sorted_id = F.tensor(np.argsort(sorted_id))
        id_tensor = id_tensor[F.tensor(sorted_id)]
        server, count = np.unique(F.asnumpy(server_id), return_counts=True)
        # pull data from server by server order
        start = 0
        pull_count = 0
        local_data = None
        for idx in range(len(server)):
            end = start + count[idx]
            if start == end:  # don't have any data in target server
                continue
            partial_id = id_tensor[start:end]
            if server[idx] == self._local_server_id: # local pull
                if (name in self._has_global_to_local) == True:
                    local_id = self._global_to_local[name][partial_id]
                else:
                    local_id = partial_id  
                local_data = self._local_pull_handler(name, local_id, self._data_store)
            else: # pull data from remote server
                msg = KVStoreMsg(
                    type=KVMsgType.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=partial_id, 
                    data=None)
                _send_kv_msg(self._sender, msg, server[idx])
                pull_count += 1

            start += count[idx]

        msg_list = []
        if local_data is not None:
            local_msg = KVStoreMsg(
                type=KVMsgType.PULL_BACK, 
                rank=self._local_server_id, 
                name=name, 
                id=None,
                data=local_data)
            msg_list.append(local_msg)

        # wait message from server nodes
        for idx in range(pull_count):
            msg_list.append(_recv_kv_msg(self._receiver))

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
            data=None)

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
                data=None)
            _send_kv_msg(self._sender, msg, server_id)


    def get_id(self):
        """Get client id

        Return
        ------
        int
            KVClient ID
        """
        return self._client_id

    def _takeId(self, elem):
        return elem.rank


    def _local_push_handler(self, name, ID, data, target):
        """Default handler for local PUSH message. 

        On default, _push_handler perform ADD operation for the tensor.

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
        target[name][ID] += data


    def _local_pull_handler(self, name, ID, target):
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
    
