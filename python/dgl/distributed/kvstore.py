"""Define distributed kvstore"""

import time

from . import rpc
from .constants import get_type_str

from .. import backend as F

KVSTORE_PULL = 901231

class PullResponse(rpc.Response):
    """Send the sliced data tensor back to client.

    Parameters
    ----------
    data_tensor: tensor
        a tensor with the same row size of data ID.
    """
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getstate__(self):
        return self.data_tensor

    def __setstate__(self, state):
        self.data_tensor = state

class PullRequest(rpc.Request):
    """Send ID tensor to server and get target data tensor as response.

    Parameters
    ----------
    name : str
        data name
    id_tensor : tensor
        a vector storing the data ID
    """
    def __init__(self, name, id_tensor):
    	self.name = name
        self.id_tensor = id_tensor

    def __getstate__(self):
        return self.name, self.id_tensor

    def __setstate__(self, state):
        self.name, self.id_tensor = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        local_id = kv.part_policy[self.name].to_local(self.id_tensor)
        data = kv.pull_handler(self.name, local_id)
        res = PullResponse(data)
        return res

KVSTORE_PUSH = 901232

class PushRequest(rpc.Response):
    """Send ID tensor and data tensor to target server and
    update kvstore's data.

    This request has no response.

    Parameters
    ----------
    name : str
        data name
    id_tensor : tensor
        a vector storing the data ID
    data_tensor : tensor
        a tensor with the same row size of data ID
    """
    def __init__(self, name, id_tensor, data_tensor):
        self.name = name
        self.id_tensor = id_tensor
        self.data_tensor = data_tensor

    def __getstate__(self):
        return self.name, self.id_tensor, self.data_tensor

    def __setstate__(self, state):
        self.name, self.id_tensor, self.data_tensor = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        local_id = kv.part_policy[self.name].to_local(self.id_tensor)
        kv.push_handler(self.name, local_id, self.data_tensor)

INIT_DATA = 901233

class InitDataResponse(rpc.Response):
    """Send confirmation response (just a server ID) of
    InitDataRequest to client.

    Parameters
    ----------
    server_id : int
        ID of current server
    """
    def __init__(self, server_id):
        self.server_id = server_id

    def __getstate__(self):
        return self.server_id

    def __setstate__(self, state):
        self.server_id = state

class InitDataRequest(rpc.Request):
    """Send meta data to server to init data tensor on target kvserver.

    Parameters
    ----------
    name : str
        data name
    shape : tuple
        data shape
    dtype : str
        data type string, e.g., 'int64', 'float32', etc.
    part_policy_str : str
        partition policy string, 'e.g., edge' or 'node'.
    init_func : function
        user-defined init function
    """
    def __init__(self, name, shape, dtype, part_policy_str, init_func):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.part_policy_str = part_policy_str
        self.init_func = init_func

    def __getstate__(self):
        return self.name, self.shape, self.dtype, self.part_policy_str, self.init_func

    def __setstate__(self, state):
        self.name, self.shape, self.dtype, self.part_policy_str, self.init_func = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        dtype = F.data_type_dict[self.dtype]
        if kv.is_main_server(): # main server
            data_tensor = self.init_func(data_shape, dtype, F.cpu())
            kv.init_data(name=name, data_tensor=data_tensor)
        else: # backup server will read data from shared-memory
            kv.init_data(name=name)
        # Find the same partition policy from exsiting plolicy list.
        for _, policy in kv.part_policy.items():
            if policy.policy_str == self.part_policy_str:
                kv.part_policy[self.name] = policy
                res = InitDataResponse(kv.server_id)
                return res
        raise RuntimeError("Cannot find any partition policy match \
            the policy string : %s" % self.part_policy_str)

BARRIER = 901235

class BarrierResponse(rpc.Response):
    """Send an unblock signal (just a server ID) to client.

    Parameters
    ----------
    server_id : int
        ID of current server
    """
    def __init__(self, server_id):
        self.server_id = server_id

    def __getstate__(self):
        return self.server_id

    def __setstate__(self, state):
        self.server_id = state

class BarrierRequest(rpc.Request):
    """Send a barrier signal (just a client ID) to server.

    Parameters
    ----------
    client_id : int
        ID of current client
    """
    def __init__(self, client_id):
        self.client_id = client_id

    def __getstate__(self):
        return self.client_id

    def __setstate__(self):
        self.client_id = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        kv.barrier_count += 1
        if kv.barrier_count == kv.num_clients:
            res_list = []
            for cli_id in range(kv.num_clients):
                res_list.append(BarrierResponse(kv.server_id))
            return res_list
        else:
            return None

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or
    graph embeddings across machines in a distributed setting. KVServer depends on DGL rpc
    infrastructure thats support backup servers, which means we can lunach many KVServers
    on the same machine for load-balancing.

    DO NOT use KVServer in mult-threads because this behavior is not defined. For now, KVServer
    can only support CPU-to-CPU communication. We may support GPU-communication in the future.

    Parameters
    ----------
    server_id : int
        ID of current server (starts from 0).
    ip_config : str
        Path of IP configuration file.
    num_clients : int
        Total number of KVClients that will be connected to the KVServer.
    """
    def __init__(self, server_id, ip_config, num_clients):
        assert server_id >= 0, 'server_id (%d) cannot be a negative number.' % server_id
        assert len(ip_config) > 0, 'ip_config cannot be empty.'
        assert num_clients >= 0, 'num_clients (%d) cannot be a negative number.' % num_clients
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        # Store the tensor data with specified data name
        self._data_store = {}
        # Store the partition information, e.g, partition_book and g2l mapping
        self._part_policy = {}
        # Used for barrier() API on KVClient
        self._barrier_count = 0
        self._server_id = server_id
        self._server_namebook = rpc.read_ip_config(ip_config)
        self._machine_id = self._server_namebook[server_id][0]
        self._group_count = self._server_namebook[server_id][3]
        self._num_clients = num_clients
        # TODO(chao) : remove tmp file
        self._open_file_list = []
        # user-defined push and pull handler
        self._push_handler = self._default_push_handler
        self._pull_handler = self._default_pull_handler

    def __del__(self):
        """Finalize KVServer
        """
        # TODO(chao) : remove tmp file
        # Delete temp file when kvstore service is closed
        for file in self._open_file_list:
            if (os.path.exists(file)):
                os.remove(file)

     @property
     def server_id(self):
         return self._server_id

     @property
     def group_count(self):
         return self._group_count

     @property
     def barrier_count(self):
     	return self._barrier_count

     @property
     def num_clients(self):
     	return self._num_clients
     
     @property
     def data_store(self):
         return self._data_store

     @property
     def partition_policy(self):
         return self._partition_policy
     
     @property
     def push_handler(self):
         return self._push_handler

     @property
     def pull_handler(self):
         return self._pull_handler

    def init_data(self, name, data_tensor=None):
        """Init data tensor on kvserver.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor
            If the data_tensor is None, KVServer will read shared-memory by name.
        """
        # TODO(chao) : Once empty_shared_mem delete the shape and dtype arguments, 
        # we can remove tmp file
        assert len(name) > 0, 'name cannot be empty.'
        if data_tensor is not None: # Create shared-tensor
            data_type = get_type_str(F.dtype(data_tensor))
            shared_data = empty_shared_mem(name+'-data-', True, data_tensor.shape, data_type)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
            self._data_store[name][:] = data_tensor[:]
            self._write_data_meta_to_file(name+'-meta-'+str(self._machine_id), data_tensor)
            self._open_file_list.append(name+'-meta-'+str(self._machine_id))
        else: # Read shared-tensor
            while True:
                if (os.path.exists(name+'-meta-'+str(self._machine_id))):
                    break
                else:
                    time.sleep(1) # wait until the file been created
            data_shape, data_type = self._read_data_meta_from_file(name+'-meta-'+str(self._machine_id))
            shared_data = empty_shared_mem(name+'-data-', False, data_shape, data_type)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)

    def set_partition_policy(self, name, policy):
    	"""Set partition policy 

        Set a partition policy to target data.

        Parameters
        ----------
        name : str
            data name
        policy : PartitionPolicy
            KVStore assume the policy has already been in shared-memory
        """
        assert len(name) > 0, 'name cannot be empty.'
        self._part_policy[name] = policy

    def _write_data_meta_to_file(self, filename, data):
        """Write data meta infor to a temp file.

        Parameters
        ----------
        filename : str
            name of temp file.
        data : tensor (mx.ndarray or torch.tensor)
            data tensor
        """
        # TODO(chao) : Once empty_shared_mem delete the shape and dtype arguments, 
        # we can remove tmp file
        assert len(filename) > 0, 'filename cannot be empty.'
        if(os.path.exists(filename)):
            os.remove(filename)

        shape = F.shape(data)
        str_data = ''
        str_data += get_type_str(F.dtype(data))
        str_data += '|'
        f = open(filename, "a");
        for s in shape:
            str_data += str(s)
            str_data += '|'
        f.write(str_data)
        f.close()

    def _read_data_meta_from_file(self, filename):
        """Read data meta info from a tmp file.

        Parameters
        ----------
        filename : str
            name of temp file

        Return
        ------
        tuple
            data shape
        """
        # TODO(chao) : Once empty_shared_mem delete the shape and dtype arguments, 
        # we can remove tmp file
        assert len(filename) > 0, 'filename cannot be empty.'
        f = open(filename, "r")
        str_data = f.read()
        data_list = str_data.split('|')
        data_type = data_list[0]
        data_shape = []
        for i in range(1, len(data_list)-1):
            data_shape.append(int(data_list[i]))
        f.close()

        return data_shape, data_type

class KVClient(object):
    """KVClient is used to push/pull data to/from KVServer. If the target kvclient and kvserver are
    in the same machine, they can communicate with each other using local shared-memory automatically,
    instead of going through the tcp/ip RPC.

    DO NOT use KVClient in multi-threads because this behavior is not defined. For now, KVClient
    can only support CPU-to-CPU communication. We may support GPU-communication in the future.

    Parameters
    ----------
    ip_config : str
        Path of IP configuration file.
    """
    def __init__(self，ip_config):
        assert len(ip_config) > 0, 'ip_config cannot be empty.'
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        self._data_store = {}
        self._server_count = len(server_namebook)
        self._group_count = server_namebook[0][3]
        self._machine_count = int(self._server_count / self._group_count)
        self._machine_id = rpc.get_machine_id()
        self._client_id = rpc.get_rank()
        # Delete temp file when kvstore service is closed
        self._open_file_list = []
        # User-defined pull handler
        self._pull_handler = None
        # User-defined push handler
        self._push_handler = None
        random.seed(time.time())

    def __del__(self):
        """Finalize KVClient
        """
        # Delete temp file whhen kvstore service is closed
        for file in self._open_file_list:
            if(os.path.exists(file)):
                os.remove(file)

    def init_data(self, name, shape, dtype, policy):
        """Send message to kvserver to initialize new data and 
        get corresponded shared-tensor (e.g., partition_book, g2l) on kvclient. 

        The new data will be initialized to zeros.

        Note that, this API must be invoked after the conenct() API. 

        Parameters
        ----------
        name : str
            data name
        shape : list or tuple of int
            data shape
        dtype : dtype
            data type
        target_name : str
            target name is used to find existing partition_book and g2l mapping.
        """


    def get_shared_data():
    	"""
    	"""

    def push(self, name, id_tensor, data_tensor):
        pass

    def pull(self, name, id_tensor):
        pass

    def barrier(self):
        pass