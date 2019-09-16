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
    graph embeddings across machines in a distributed setting or storing them in one standalone 
    machine with big memory capability. DGL KVServer uses a very simple range-partition scheme to 
    partition data into different KVServer nodes. For example, if the total embedding size is 200 and 
    we have two KVServer nodes, the data (0~99) will be stored in kvserver_0, and the data (100~199) will 
    be stored in kvserver_1.

    For KVServer, user can re-wriite UDF function for _push_handler and _pull_handler.

    DO NOT use KVServer in multiple threads!

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
        IP address of current KVServer node, e.g., '127.0.0.1:50051'
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, server_id, client_namebook, server_addr, net_type='socket'):
        assert server_id >= 0, 'server_id cannot be a negative number.'
        assert len(client_namebook) > 0, 'client_namebook cannot be empty.'
        assert len(server_addr.split(':')) == 2, 'Incorrect IP format.'
        self._is_init = set()  # Contains tensor name
        self._data_store = {}  # Key is name string and value is tensor
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
                # convert global ID to local ID
                local_id = self._remap_id(msg.name, msg.id)
                self._push_handler(msg.name, local_id, msg.data)
            elif msg.type == KVMsgType.PULL:
                # convert global ID to local ID
                local_id = self._remap_id(msg.name, msg.id)
                res_tensor = self._pull_handler(msg.name, local_id)
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
                print("Exit KVStore service, server ID: %d" % self.get_id())
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

    def _push_handler(self, name, ID, data):
        """User-defined handler for PUSH message. 

        On default, _push_handler perform ADD operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs that has been re-mapped to local id.
        data : tensor (mx.ndarray or torch.tensor)
            a matrix with the same row size of id
        """
        for idx in range(ID.shape[0]):  # For each row
            self._data_store[name][ID[idx]] += data[idx]

    def _pull_handler(self, name, ID):
        """User-defined handler for PULL operation.

        On default, _pull_handler perform gather_row() operation for the tensor.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs that has been re-mapped to local id.

        Return
        ------
        tensor
            a matrix with the same row size of ID
        """
        new_tensor = F.gather_row(self._data_store[name], ID)
        return new_tensor

    def _remap_id(self, name, ID):
        """Re-mapping global-ID to local-ID.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the global data ID

        Return
        ------
        tensor
            re-mapped lcoal ID
        """
        row_size = self._data_store[name].shape[0]
        return ID % row_size

class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer on DGL trainer.

    There are three operations supported by KVClient:

      * init_data(name, shape, low, high): initialize tensor on KVServer
      * push(name, id, data): push data to KVServer
      * pull(name, id): pull data from KVServer
      * shut_down(): shut down all KVServer nodes

    DO NOT use KVClient in multiple threads!

    Parameters
    ----------
    client_id : int
        KVClient's ID (start from 0)
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's IP address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
    client_addr : str
        IP address of current KVClient, e.g., '168.12.23.22:50051'
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, client_id, server_namebook, client_addr, net_type='socket'):
        assert client_id >= 0, 'client_id cannot be a nagative number.'
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert len(client_addr.split(':')) == 2, 'Incorrect IP format.'
        # self._data_size is a key-value store where the key is data name 
        # and value is the size of tensor. It is used to partition data into
        # different KVServer nodes.
        self._data_size = {}
        self._client_id = client_id
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._addr = client_addr
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

    def init_data(self, name, shape, init_type='zero', low=0.0, high=0.0):
        """Initialize kvstore tensor

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
        self._data_size[name] = shape[0]
        count = math.ceil(shape[0] / self._server_count)
        # We hack the msg format here
        init_type = 0.0 if init_type == 'zero' else 1.0
        threshold = F.tensor([[init_type, init_type], [low, high]])
        # partition shape on server
        for server_id in range(self._server_count):
            par_shape = shape.copy()
            if shape[0] - server_id*count >= count:
                par_shape[0] = count
            else:
                par_shape[0] = shape[0] - server_id*count
            tensor_shape = F.tensor(par_shape)
            msg = KVStoreMsg(
                type=KVMsgType.INIT,
                rank=self._client_id,
                name=name,
                id=tensor_shape,
                data=threshold)
            _send_kv_msg(self._sender, msg, server_id)
        
    def push(self, name, ID, data):
        """Push sparse message to KVServer

        The push() API will partition message into different 
        KVServer nodes automatically.

        Note that we assume the row Ids in ID is in the ascending order.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the global IDs
        data : tensor (mx.ndarray or torch.tensor)
            a tensor with the same row size of id
        """
        assert F.ndim(ID) == 1, 'ID must be a vector.'
        assert F.shape(ID)[0] == F.shape(data)[0], 'The data must has the same row size with ID.'
        group_size = [0] * self._server_count
        numpy_id = F.asnumpy(ID)
        count = math.ceil(self._data_size[name] / self._server_count)
        server_id = numpy_id / count
        id_list, id_count = np.unique(server_id, return_counts=True)
        for idx in range(len(id_list)):
            group_size[int(id_list[idx])] += id_count[idx]
        min_idx = 0
        max_idx = 0
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            max_idx += group_size[idx]
            range_id = ID[min_idx:max_idx]
            range_data = data[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=KVMsgType.PUSH,
                rank=self._client_id,
                name=name,
                id=range_id,
                data=range_data)
            _send_kv_msg(self._sender, msg, idx)

    def push_all(self, name, data):
        """Push the whole data to KVServer

        The push_all() API will partition message into different
        KVServer nodes automatically.

        Note that we assume the row Ids in ID is in the ascending order.

        Parameters
        ----------
        name : str
            data name
        data : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        ID = F.zerocopy_from_numpy(np.arange(F.shape(data)[0]))
        self.push(name, ID, data)

    def pull(self, name, ID):
        """Pull sparse message from KVServer

        Note that we assume the row Ids in ID is in the ascending order.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs

        Return
        ------
        tensor
            a tensor with the same row size of ID
        """
        assert F.ndim(ID) == 1, 'ID must be a vector.'
        group_size = [0] * self._server_count
        numpy_id = F.asnumpy(ID)
        count = math.ceil(self._data_size[name] / self._server_count)
        server_id = numpy_id / count
        id_list, id_count = np.unique(server_id, return_counts=True)
        for idx in range(len(id_list)):
            group_size[int(id_list[idx])] += id_count[idx]
        min_idx = 0
        max_idx = 0
        server_count = 0
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            server_count += 1
            max_idx += group_size[idx]
            range_id = ID[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=KVMsgType.PULL,
                rank=self._client_id,
                name=name,
                id=range_id,
                data=None)
            _send_kv_msg(self._sender, msg, idx)
        # Recv back message
        msg_list = []
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            msg = _recv_kv_msg(self._receiver)
            assert msg.type == KVMsgType.PULL_BACK, 'Recv kv msg error.'
            msg_list.append(msg)

        return self._merge_msg(msg_list)

    def pull_all(self, name):
        """Pull the whole data from KVServer

        Note that we assume the row Ids in ID is in the ascending order.
        
        Parameters
        ----------
        name : str
            data name

        Return
        ------
        tensor
            target data tensor
        """
        ID = F.zerocopy_from_numpy(np.arange(self._data_size[name]))
        return self.pull(name, ID)
    
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

    def _sort_func(self, msg):
        """Sort function for KVStoreMsg: sort message by rank

        Parameters
        ----------
        msg : KVStoreMsg
            KVstore message
        """
        return msg.rank

    def _merge_msg(self, msg_list):
        """Merge separated message to a big matrix

        Parameters
        ----------
        msg_list : list
            a list of KVStoreMsg

        Return
        ------
        tensor (mx.ndarray or torch.tensor)
            a merged data matrix
        """
        msg_list.sort(key=self._sort_func)
        return F.cat([msg.data for msg in msg_list], 0)