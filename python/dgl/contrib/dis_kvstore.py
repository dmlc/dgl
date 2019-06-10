# This file contains DGL distributed kvstore APIs.
from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _add_receiver_addr, _receiver_wait
from ..network import _send_kv_msg, _recv_kv_msg
from ..network import _sender_connect, SocketSync
from ..network import _PUSH_MSG, _PULL_MSG, _PULL_BACK_MSG
from ..network import KVStoreMsg

import math

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers use KVServer to hold large-scale graph features or 
    graph embeddings across machines, or storing them in one standalone machine 
    with big memory capability.

    Parameters
    ----------
    server_id : int
        KVStore ID (start from 0). DGL KVServer uses a simple 
        range-partition scheme to partition data on KVServer.
    client_namebook : dict
        IP address namebook of KVClient, where key is client's ID 
        and value is client's IP address, e.g.,

            { 0:'168.12.23.45:50051', 
              1:'168.12.23.21:50051', 
              2:'168.12.46.12:50051' }
        The client's ID also starts from 0.
    server_addr : str
        IP address of current KVServer, e.g., '127.0.0.1:50051'
    """
    def __init__(self, server_id, client_namebook, server_addr):
        assert server_id >= 0, 'server_id must be greater than or equal to 0.'
        assert len(client_namebook) > 0, 'client_namebook cannot be empty.'
        # self._data_store is a key-value store 
        # where the key is data name and value is tensor (mx.ndarray or torch.tensor)
        self._data_store = {}
        self._server_id = server_id
        self._client_namebook = client_namebook
        self._addr = server_addr
        self._sender = _create_sender()
        self._receiver = _create_receiver()

    def __del__(self):
        """Finalize the service of KVServer.
        """
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)

    def add_data(self, name, data):
        """Add tensor data to KVServer

        Parameters
        ----------
        name : str
            data name
        data : tensor (mx.ndarray or torch.tensor)
            graph embedding (or graph feature) matrix
        """
        self._data_store[name] = data

    def start(self):
        """Start the service of KVServer
        """
        server_ip, server_port = self._addr.split(':')
        _receiver_wait(self._receiver, server_ip, int(server_port), len(self._client_namebook))
        SocketSync() # wait client setup
        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)
        _sender_connect(self._sender)
        # Service loop (use Ctl+ C to exit)
        while True:
            msg = _recv_kv_msg(self._receiver)
            if msg.type == _PUSH_MSG:
                self._push_handler(msg.name, msg.id, msg.data)
            elif msg.type == _PULL_MSG:
                res_tensor = self._pull_handler(msg.name, msg.id)
                back_msg = KVStoreMsg(
                    type=_PULL_BACK_MSG,
                    rank=self._server_id,
                    name=msg.name,
                    id=msg.id,
                    data=res_tensor)
                _send_kv_msg(self._sender, back_msg, msg.rank)
            else:
                raise RuntimeError('Unknown message type: %d' % msg.type)

    def _push_handler(self, name, ID, data):
        """User-defined msg handler for push operation

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs
        data : tensor (mx.ndarray or torch.tensor)
            a data matrix with the same row size of id
        """
        pass

    def _pull_handler(self, name, ID):
        """User-defined msg handler for pull operation

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs

        Return
        ------
        tensor
            a data matrix with the same row size of ID
        """    
        pass

class KVClient(object):
    """KVClient is used to send and recv message to/from KVServer on DGL trainer.

    There are two operations supported by KVClient:

      * push(): push tensor data to KVServer
      * pull(): pull tensor data from KVServer with target ID

    Parameters
    ----------
    client_id : int
        KVClient's ID (start from 0)
    server_namebook: dict
        IP address namebook of KVServer, where key is KVServer's ID 
        and value is server's IP address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
        The KVServer's ID also starts from 0.
    """
    def __init__(self, client_id, server_namebook, client_addr):
        assert client_id >= 0, 'client_id must be greater than or equal to 0.'
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        # self._data_size is a key-value store 
        # where the key is data name and value is the size of tensor
        # self._data_size and self._group_size is used to partition message 
        # into different KVServer nodes.
        self._data_size = {}
        self._group_size = [0] * len(server_namebook)
        self._client_id = client_id
        self._server_namebook = server_namebook
        self._addr = client_addr
        self._sender = _create_sender()
        self._receiver = _create_receiver()

    def __del__(self):
        """Finalize KVClient (Disconnect to KVServer)
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
        _receiver_wait(self._receiver, client_ip, int(client_port), len(self._server_namebook))

    def set_data_size(self, name, size):
        """Let KVClient know the data size of kvstore

        Parameters
        ----------
        name : str
            data name
        size: int
            data size
        """
        self._data_size[name] = size
        
    def push(self, name, ID, data):
        """Push message to KVServer

        The push() function will partition message and push them into
        different KVServer nodes automatically.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a tensor vector storing the IDs
        data : tensor (mx.ndarray or torch.tensor)
            a tensor matrix with the same row size of id
        """
        assert ID.dim() == 1, 'ID must be a vector.'
        assert data.size(0) == ID.size(0), 'The data must has the same row size with ID vector.'
        assert self._data_size[name] > 0, 'Please invoke set_data_size() before push().'
        for id in ID:
            server_id = self._get_server_id(id.item(), name)
            self._group_size[server_id] += 1
        min_idx = 0
        max_idx = 0
        for idx in range(len(self._server_namebook)):
            if self._group_size[idx] == 0:
                continue
            max_idx += self._group_size[idx]
            range_id = ID[min_idx:max_idx]
            range_data = data[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=_PUSH_MSG,
                rank=self._client_id,
                name=name,
                id=range_id,
                data=range_data)
            _send_kv_msg(self._sender, msg, idx)

        self._group_size = [0] * len(self._server_namebook)

    def pull(self, name, ID):
        """Pull message from KVServer

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs

        Return
        ------
        tensor
            a data matrix with the same row size of ID
        """
        assert ID.dim() == 1, 'ID must be a vector.'
        assert self._data_size[name] > 0, 'Please invoke set_data_size() before pull().'
        for id in ID:
            server_id = self._get_server_id(id.item(), name)
            group_count[server_id] += 1
        min_idx = 0
        max_idx = 0
        server_count = 0
        for idx in range(len(self._server_namebook)):
            if self._group_size[idx] == 0:
                continue
            server_count += 1
            max_idx += self._group_size[idx]
            range_id = ID[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=_PULL_MSG,
                rank=self._client_id,
                name=name,
                id=range_id)
            _send_kv_msg(self._sender, msg, idx)
        # Recv back message
        msg_list = []
        for idx in range(server_count):
            msg = _recv_kv_msg(self._receiver)
            msg_list.add(msg)

        self._group_size = [0] * len(self._server_namebook)

        return self._merge_msg_to_matrix(msg_list)
    
    def _get_server_id(self, id, name):
        """Get target server id by given a data id

        Parameters
        ----------
        id : int
            data id
        name : str
            data name

        Return
        ------
        int
           target server id
        """
        count = math.ceil(self._data_size[name] / len(self._server_namebook))
        return int(id / count)

    def _merge_msg_to_matrix(self, msg_list):
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
        pass