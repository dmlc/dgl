# This file contains DGL distributed kvstore APIs.
from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _network_wait, _add_receiver_addr
from ..network import _receiver_wait, _sender_connect
from ..network import _send_kv_msg, _recv_kv_msg
from ..network import KVMsgType, KVStoreMsg

import math
import dgl.backend as F
import numpy as np

def ReadNetworkConfigure(filename):
    """Read networking configuration from file.

    The config file is like:

        server 172.31.40.143:50050 0
        client 172.31.40.143:50051 0
        client 172.31.36.140:50051 1
        client 172.31.47.147:50051 2
        client 172.31.30.180:50051 3

    Here we have 1 server node and 4 client nodes.

    Parameters
    ----------
    filename : str
        name of target configure file

    Returns
    -------
    dict
        server namebook
    dict
        client namebook
    """
    server_namebook = {}
    client_namebook = {}
    lines = [line.rstrip('\n') for line in open(filename)]
    for line in lines:
        node_type, addr, node_id = line.split(' ')
        if node_type == 'server':
            server_namebook[int(node_id)] = addr
        elif node_type == 'client':
            client_namebook[int(node_id)] = addr
        else:
            raise RuntimeError("Unknown node type: %s", node_type)

    return server_namebook, client_namebook

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or 
    graph embeddings across machines in a distributed setting. User can re-wriite _push_handler 
    and _pull_handler to support flexibale models.

    Note that, DO NOT use KVServer in multiple threads! 

    Parameters
    ----------
    server_id : int
        KVServer's ID (start from 0). 
    client_namebook : dict
        IP address namebook of KVClient, where the key is the client's ID 
        (start from 0) and the value is client's IP address, e.g.,

            { 0:'168.12.23.45:50051', 
              1:'168.12.23.21:50051', 
              2:'168.12.46.12:50051' }
    server_addr : str
        IP address of current KVServer node, e.g., '127.0.0.1:50051'.
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi' (do not support yet).
    """
    def __init__(self, server_id, client_namebook, server_addr, net_type='socket'):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert len(client_namebook) > 0, 'client_namebook cannot be empty.'
        assert len(server_addr.split(':')) == 2, 'Incorrect IP format: %s' % server_addr
        self._is_init = set()  # Contains tensor name
        self._data_store = {}  # Key is name (string) and value is data (tensor)
        self._barrier_count = 0;
        self._server_id = server_id
        self._client_namebook = client_namebook
        self._client_count = len(client_namebook)
        self._addr = server_addr
        self._sender = _create_sender(net_type)
        self._receiver = _create_receiver(net_type)

    def __del__(self):
        """Finalize KVServer
        """
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)

    def init_data(self, name, data_tensor):
        """KVServer supports data initialization on server.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        self._data_store[name] = data_tensor
        self._is_init.add(name)

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
                if (msg.name in self._is_init) == False:
                    # we hack the msg format here:
                    # msg.id store the shape of target tensor
                    # msg.data has two row, and the first row is 
                    # the init_type, [0, 0] means 'zero' and [1,1]
                    # means 'uniform'. The second row is the min & max threshold.
                    data_shape = F.asnumpy(msg.id).tolist()
                    row_0 = (F.asnumpy(msg.data).tolist())[0] 
                    row_1 = (F.asnumpy(msg.data).tolist())[1]
                    init_type = 'zero' if row_0[0] == 0.0 else 'uniform'
                    self._init_data(name=msg.name,
                        shape=data_shape,
                        init_type=init_type,
                        low=row_1[0],
                        high=row_1[1])
                    self._is_init.add(msg.name)
            elif msg.type == KVMsgType.PUSH:
                self._push_handler(msg.name, msg.id, msg.data, self._data_store)
            elif msg.type == KVMsgType.PULL:
                res_tensor = self._pull_handler(msg.name, msg.id, self._data_store)
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

    def get_id(self):
        """Get server id

        Return
        ------
        int
            KVServer ID
        """
        return self._server_id

    def _init_data(self, name, shape, init_type, low, high):
        """Initialize kvstore tensor.

        Parameters
        ----------
        name : str
            data name
        shape : list of int
            The tensor shape
        init_type : str
            initialize method, including 'zero' and 'uniform'
        low : float
            min threshold
        high : float
            max threshold
        """
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
        target : target tensor (mx.ndarray or torch.tensor)
            the target tensor
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
            a vector storing the IDs that has been re-mapped to local id.
        target : tensor (mx.ndarray or torch.tensor)
            the target tensor

        Return
        ------
        tensor
            a tensor with the same row size of ID
        """
        return target[name][ID]

class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer on DGL trainer.

    There are five operations supported by KVClient:

      * init_data(name, server_id, shape, init_type, low, high): 
          initialize tensor on target KVServer.
      * push(name, server_id, id_tensor, data_tensor): 
          push sparse data to KVServer given specified ID.
      * pull(name, server_id, id_tensor): 
          pull sparse data from KVServer given specified ID.
      * pull_wait(): 
          wait scheduled pull operation finish its job.
      * shut_down(): 
          shut down all KVServer nodes.

    Note that, DO NOT use KVClient in multiple threads!

    Parameters
    ----------
    client_id : int
        KVClient's ID (start from 0)
    local_server_id : int
        server id in current local nodes
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's IP address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
    client_addr : str
        IP address of current KVClient, e.g., '168.12.23.22:50051'
    partition_book : list of int
        A book mapping for global ID and server ID
    global_to_local : list of int
        A book mapping for global ID and local ID. This flag is optional, and if it is None, we use original ID.
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, client_id, local_server_id, server_namebook, client_addr, partition_book, global_to_local=None, net_type='socket'):
        assert client_id >= 0, 'client_id (%d) cannot be a nagative number.' % client_id
        assert local_server_id >= 0, 'local_server_id (%d) cannot be a negative number.' % local_server_id
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert len(client_addr.split(':')) == 2, 'Incorrect IP format: %s' % client_addr
        assert len(partition_book) > 0, 'partition_book cannot be empty.'
        self._client_id = client_id
        self._data_store = {}  # Key is name (string) and value is data (tensor)
        self._local_server_id = local_server_id
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._addr = client_addr
        self._partition_book = F.tensor(partition_book)
        self._global_to_local = F.tensor(global_to_local)
        self._sender = _create_sender(net_type)
        self._receiver = _create_receiver(net_type)

    def __del__(self):
        """Finalize KVClient
        """
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)

    def connect(self):
        """Connect to all KVServer nodes
        """
        for ID, addr in self._server_namebook.items():
            server_ip, server_port = addr.split(':')
            _add_receiver_addr(self._sender, server_ip, int(server_port), ID)
        _sender_connect(self._sender)
        client_ip, client_port = self._addr.split(':')
        _receiver_wait(self._receiver, client_ip, int(client_port), self._server_count)

    def init_remote_data(self, name, server_id, shape, init_type='zero', low=0.0, high=0.0):
        """Initialize kvstore tensor

        we hack the msg format here: msg.id store the shape of target tensor,
        msg.data has two row, and the first row is the init_type, 
        [0, 0] means 'zero' and [1,1] means 'uniform'. 
        The second row is the min & max threshold.

        Parameters
        ----------
        name : str
            data name
        server_id : int
            target server id
        shape : list of int
            shape of tensor
        init_type : str
            initialize method, including 'zero' and 'uniform'
        low : float
            min threshold, if use 'uniform'
        high : float
            max threshold, if use 'uniform'
        """
        tensor_shape = F.tensor(shape)
        init_type = 0.0 if init_type == 'zero' else 1.0
        threshold = F.tensor([[init_type, init_type], [low, high]])
        msg = KVStoreMsg(
            type=KVMsgType.INIT,
            rank=self._client_id,
            name=name,
            id=tensor_shape,
            data=threshold)
        _send_kv_msg(self._sender, msg, server_id)

    def init_local_data(self, name, data_tensor):
        """Initialize local data on client, and this data can be shared with local server nodes.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor
        """
        self._data_store[name] = data_tensor
        
    def push(self, name, id_tensor, data_tensor):
        """Push sparse message to KVServer.

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
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        assert F.shape(id_tensor)[0] == F.shape(data_tensor)[0], 'The data must has the same row size with ID.'
        # partition data (we can move this into C-api if needed)
        server_id = self._partition_book[id_tensor]
        sorted_id = F.tensor(np.argsort(F.asnumpy(server_id)))
        id_tensor = id_tensor[sorted_id]  # id sorted by server
        data_tensor = data_tensor[sorted_id] # data sorted by server
        server, count = np.unique(F.asnumpy(server_id), return_counts=True)
        # push data
        start = 0
        for idx in range(len(server)):
            end = start + count[idx]
            ID = None
            if self._global_to_local is not None:
                ID = self._global_to_local[id_tensor[start:end]]
            else:
                ID = id_tensor[start:end]
            if len(ID) == 0: # don't have any data in target server
                start += count[idx]
                continue
            if server[idx] == self._local_server_id:  # update local data
                self._push_handler(name, ID, data_tensor[start:end], self._data_store)
            else: # push data to remote server
                msg = KVStoreMsg(
                    type=KVMsgType.PUSH, 
                    rank=self._client_id, 
                    name=name, 
                    id=ID, 
                    data=data_tensor[start:end])
                _send_kv_msg(self._sender, msg, server[idx])
            start += count[idx]

    def _push_handler(self, name, ID, data, target):
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

    def pull(self, name, id_tensor):
        """Pull sparse message from KVServer

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor (mx.ndarray or torch.tensor)
            a vector storing the ID list
        """
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        # partition data (we can move this into C-api if needed)
        server_id = self._partition_book[id_tensor]
        sorted_id = np.argsort(F.asnumpy(server_id))
        back_sorted_id = F.tensor(np.argsort(sorted_id))
        id_tensor = id_tensor[F.tensor(sorted_id)] # id sorted by server
        server, count = np.unique(F.asnumpy(server_id), return_counts=True)
        start = 0
        pull_count = 0
        for idx in range(len(server)):
            end = start + count[idx]
            ID = None
            if self._global_to_local is not None:
                ID = self._global_to_local[id_tensor[start:end]]
            else:
                ID = id_tensor[start:end]  
            if len(ID) == 0: # don't have any data in target server
                start += count[idx]
                continue                
            if server[idx] == self._local_server_id: # local pull
                self._pull_handler(name, ID, self._data_store)
            else: # pull data from remote server
                msg = KVStoreMsg(
                    type=KVStoreMsg.PULL, 
                    rank=self._client_id, 
                    name=name, 
                    id=ID, 
                    data=None)
                _send_kv_msg(self._sender, msg, server_id)
                pull_count += 1
            start += count[idx]

        msg_list = []
        for idx in range(pull_count):
            msg_list.append(_recv_kv_msg(self._receiver))

        # sort msg by server id
        msg_list.sort(key=self._takeId)
        data_tensor = F.cat(seq=[msg.data for msg in msg_list], dim=0)

        return data_tensor[back_sorted_id]

    def _takeId(elem):
        return elem.rank

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
        """Shutdown all KVServer nodes

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
