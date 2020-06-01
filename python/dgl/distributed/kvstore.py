"""Define distributed kvstore"""

import os
import time
import random
import numpy as np

from . import rpc
from .graph_partition_book import PartitionPolicy

from .. import backend as F
from .._ffi.ndarray import empty_shared_mem

def TypeToIndex(dtype):
    """Convert dtype to int index
    """
    if dtype == F.float16:
        return 30030
    elif dtype == F.float32:
        return 30031
    elif dtype == F.float64:
        return 30032
    elif dtype == F.uint8:
        return 30033
    elif dtype == F.int8:
        return 30034
    elif dtype == F.int16:
        return 30035
    elif dtype == F.int32:
        return 30036
    elif dtype == F.int64:
        return 30037
    else:
        raise RuntimeError("Cannot support dtype: %s" % str(dtype))

def IndexToType(index):
    """Convert int index to dtype
    """
    if index == 30030:
        return F.float16
    elif index == 30031:
        return F.float32
    elif index == 30032:
        return F.float64
    elif index == 30033:
        return F.uint8
    elif index == 30034:
        return F.int8
    elif index == 30035:
        return F.int16
    elif index == 30036:
        return F.int32
    elif index == 30037:
        return F.int64
    else:
        raise RuntimeError("Cannot support dtype index: %d" % index)

def write_data_meta_to_shared_mem(dataname, data):
    """Write data meta to shared-memory tensor

    Parameters
    ----------
    dataname : str
        name of shared-memory tensor
    data : tensor
        data tensor
    """
    assert len(dataname) > 0, 'dataname cannot be empty.'
    shape = list(F.shape(data))
    if len(shape) > 9:
        raise RuntimeError("Cannot support dim larger than 9.")
    type_idx = TypeToIndex(F.dtype(data))
    tmp_list = [-1] * 10
    tmp_list[0] = type_idx
    for i in range(len(shape)):
        tmp_list[i+1] = shape[i]
    meta_tensor = F.tensor(tmp_list, F.int32)
    shared_data = empty_shared_mem(dataname, True, (10), 'int32')
    dlpack = shared_data.to_dlpack()
    data_tensor = F.zerocopy_from_dlpack(dlpack)
    data_tensor[:] = meta_tensor[:]

def read_data_meta_from_shared_mem(dataname):
    """Read meta data from shared-memory tensor

    Parameters
    ----------
    dataname : str
        name of shared-memory tensor

    Returns
    -------
    dtype
        data type
    tuple
        data shape
    """
    assert len(dataname) > 0, 'dataname cannot be empty.'
    shared_data = empty_shared_mem(dataname, False, (10), 'int32')
    dlpack = shared_data.to_dlpack()
    meta_tensor = F.zerocopy_from_dlpack(dlpack)
    tmp_list = (F.asnumpy(meta_tensor)).tolist()
    dtype = IndexToType(tmp_list[0])
    shape = []
    for i in range(len(tmp_list)):
        if i != 0 and tmp_list[i] != -1:
            shape.append(tmp_list[i])
    return dtype, tuple(shape)

def check_file_exists(filename):
    """Check if file exists.

    Parameters
    ----------
    filename : str
        Path of file

    Returns
    -------
    True if file exists.
    """
    if os.path.exists(filename):
        return True
    return False

############################ Register KVStore Requsts and Responses ###############################

KVSTORE_PULL = 901231

class PullResponse(rpc.Response):
    """Send the sliced data tensor back to the client.

    Parameters
    ----------
    server_id : int
        ID of current server
    data_tensor : tensor
        sliced data tensor
    """
    def __init__(self, server_id, data_tensor):
        self.server_id = server_id
        self.data_tensor = data_tensor

    def __getstate__(self):
        return self.server_id, self.data_tensor

    def __setstate__(self, state):
        self.server_id, self.data_tensor = state

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
        data = kv.pull_handler(kv.data_store, self.name, local_id)
        res = PullResponse(kv.server_id, data)
        return res

KVSTORE_PUSH = 901232

class PushRequest(rpc.Response):
    """Send ID tensor and data tensor to server and update kvstore's data.

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
        kv.push_handler(kv.data_store, self.name, local_id, self.data_tensor)
        return None

INIT_DATA = 901233
INIT_MSG = 'Init'

class InitDataResponse(rpc.Response):
    """Send a confirmation response (just a short string message) of
    InitDataRequest to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class InitDataRequest(rpc.Request):
    """Send meta data to server and init data tensor
    on server using UDF init function.

    Parameters
    ----------
    name : str
        data name
    shape : tuple
        data shape
    dtype : str
        data type string, e.g., 'int64', 'float32', etc.
    policy_str : str
        partition-policy string, e.g., 'edge' or 'node'.
    init_func : function
        UDF init function.
    """
    def __init__(self, name, shape, dtype, policy_str, init_func):
        self.name = name
        self.shape = shape
        self.dtype = dtype
        self.policy_str = policy_str
        self.init_func = init_func

    def __getstate__(self):
        return self.name, self.shape, self.dtype, self.policy_str, self.init_func

    def __setstate__(self, state):
        self.name, self.shape, self.dtype, self.policy_str, self.init_func = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        dtype = F.data_type_dict[self.dtype]
        if kv.is_backup_server() == False:
            data_tensor = self.init_func(self.shape, dtype)
            kv.init_data(name=self.name, data_tensor=data_tensor)
        else: # backup server will read data from shared-memory
            kv.init_data(name=self.name)
        # Find the same partition policy from exsiting plolicy list.
        for _, policy in kv.part_policy.items():
            if policy.policy_str == self.policy_str:
                kv.part_policy[self.name] = policy
                res = InitDataResponse(INIT_MSG)
                return res
        raise RuntimeError("Cannot find any partition policy match \
            the requested policy : %s" % self.policy_str)

BARRIER = 901234
BARRIER_MSG = 'Barrier'

class BarrierResponse(rpc.Response):
    """Send an confimation signal (just a short string message) of
    BarrierRequest to client.

    Parameters
    ----------
    msg : string
        string msg
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class BarrierRequest(rpc.Request):
    """Send a barrier signal (just a short string message) to server.

    Parameters
    ----------
    msg : string
        string msg
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

    def process_request(self, server_state):
        assert self.msg == BARRIER_MSG
        kv = server_state.kv_store
        kv.incr_barrier_count()
        if kv.barrier_count == kv.num_clients:
            kv.barrier_zero()
            res_list = []
            for target_id in range(kv.num_clients):
                res_list.append((target_id, BarrierResponse(BARRIER_MSG)))
            return res_list
        else:
            return None # No response

REGISTER_PULL = 901235
REGISTER_PULL_MSG = 'Register_Pull'

class RegisterPullHandlerResponse(rpc.Response):
    """Send a confirmation signal (just a short string message) of
    RegisterPullHandler to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class RegisterPullHandlerRequest(rpc.Request):
    """Send an UDF and register Pull handler on server.

    Parameters
    ----------
    pull_func : func
        UDF pull handler
    """
    def __init__(self, pull_func):
        self.pull_func = pull_func

    def __getstate__(self):
        return self.pull_func

    def __setstate__(self, state):
        self.pull_func = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        kv.pull_handler = self.pull_func
        res = RegisterPullHandlerResponse(REGISTER_PULL_MSG)
        return res

REGISTER_PUSH = 901236
REGISTER_PUSH_MSG = 'Register_Push'

class RegisterPushHandlerResponse(rpc.Response):
    """Send a confirmation signal (just a short string message) of
    RegisterPushHandler to client.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

class RegisterPushHandlerRequest(rpc.Request):
    """Send an UDF to register Push handler on server.

    Parameters
    ----------
    push_func : func
        UDF push handler
    """
    def __init__(self, push_func):
        self.push_func = push_func

    def __getstate__(self):
        return self.push_func

    def __setstate__(self, state):
        self.push_func = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        kv.push_handler = self.push_func
        res = RegisterPushHandlerResponse(REGISTER_PUSH_MSG)
        return res

GET_SHARED = 901237
GET_SHARED_MSG = 'Get_Shated'

class GetSharedDataResponse(rpc.Response):
    """Send meta data of shared-memory tensor to client.

    TODO(chao): We will support new shared-memory API and we can
    just send the data name to client.

    Parameters
    ----------
    meta : dict
        a dict of meta, e.g.,

        {'data_0' : (shape, dtype, policy_str),
         'data_1' : (shape, dtype, policy_str)}
    """
    def __init__(self, meta):
        self.meta = meta

    def __getstate__(self):
        return self.meta

    def __setstate__(self, state):
        self.meta = state

class GetSharedDataRequest(rpc.Request):
    """Send a signal (just a client ID) to get the
    meta data of shared-tensor from server.

    Parameters
    ----------
    msg : string
        string message
    """
    def __init__(self, msg):
        self.msg = msg

    def __getstate__(self):
        return self.msg

    def __setstate__(self, state):
        self.msg = state

    def process_request(self, server_state):
        assert self.msg == GET_SHARED_MSG
        meta = {}
        kv = server_state.kv_store
        for name, data in kv.data_store.items():
            meta[name] = (F.shape(data), 
                          F.reverse_data_type_dict[F.dtype(data)],
                          kv.part_policy[name].policy_str)
        res = GetSharedDataResponse(meta)
        return res

GET_PART_SHAPE = 901238

class GetPartShapeResponse(rpc.Response):
    """Send the partitioned data shape back to client.

    Parameters
    ----------
    shape : tuple
        shape of tensor
    """
    def __init__(self, shape):
        self.shape = shape

    def __getstate__(self):
        return self.shape

    def __setstate__(self, state):
        self.shape = state

class GetPartShapeRequest(rpc.Request):
    """Send data name to get the partitioned data shape from server.

    Parameters
    ----------
    name : str
        data name
    """
    def __init__(self, name):
        self.name = name

    def __getstate__(self):
        return self.name

    def __setstate__(self, state):
        self.name = state

    def process_request(self, server_state):
        kv = server_state.kv_store
        data_shape = F.shape(kv.data_store[self.name])
        res = GetPartShapeResponse(data_shape)
        return res

############################ KVServer ###############################

def default_push_handler(target, name, id_tensor, data_tensor):
    """Default handler for PUSH message. 

    On default, _push_handler perform scatter_row() operation for the tensor.

    Parameters
    ----------
    target : tensor
        target tensor
    name : str
        data name
    id_tensor : tensor
        a vector storing the ID list.
    data_tensor : tensor
        a tensor with the same row size of id
    """
    target[name] = F.scatter_row(target[name], id_tensor, data_tensor)

def default_pull_handler(target, name, id_tensor):
    """Default handler for PULL operation.

    On default, _pull_handler perform gather_row() operation for the tensor.

    Parameters
    ----------
    target : tensor
        target tensor
    name : str
        data name
    id_tensor : tensor
        a vector storing the ID list.

    Return
    ------
    tensor
        a tensor with the same row size of ID.
    """
    return F.gather_row(target[name], id_tensor)


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
        assert check_file_exists(ip_config), 'Cannot open file: %s' % ip_config
        assert num_clients >= 0, 'num_clients (%d) cannot be a negative number.' % num_clients
        # Register services on server
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        rpc.register_service(BARRIER, 
                             BarrierRequest, 
                             BarrierResponse)
        rpc.register_service(REGISTER_PUSH, 
                             RegisterPushHandlerRequest,
                             RegisterPushHandlerResponse)
        rpc.register_service(REGISTER_PULL,
                             RegisterPullHandlerRequest,
                             RegisterPullHandlerResponse)
        rpc.register_service(GET_SHARED,
                             GetSharedDataRequest,
                             GetSharedDataResponse)
        rpc.register_service(GET_PART_SHAPE,
                             GetPartShapeRequest,
                             GetPartShapeResponse)
        # Store the tensor data with specified data name
        self._data_store = {}
        # Store the partition information with specified data name
        self._part_policy = {}
        # Basic information
        self._server_id = server_id
        self._server_namebook = rpc.read_ip_config(ip_config)
        # TODO(chao): machine_id can be removed if we use new shared-memory API
        self._machine_id = self._server_namebook[server_id][0]
        self._group_count = self._server_namebook[server_id][3]
        # We assume partition_id is equal to machine_id
        self._part_id = self._machine_id
        self._num_clients = num_clients
        self._barrier_count = 0
        # push and pull handler
        self._push_handler = default_push_handler
        self._pull_handler = default_pull_handler

    @property
    def server_id(self):
        return self._server_id

    @property
    def barrier_count(self):
        return self._barrier_count

    def incr_barrier_count(self):
        self._barrier_count +=1

    def barrier_zero(self):
        self._barrier_count = 0

    @property
    def num_clients(self):
        return self._num_clients
     
    @property
    def data_store(self):
        return self._data_store

    @property
    def part_policy(self):
        return self._part_policy
     
    @property
    def push_handler(self):
        return self._push_handler

    @property
    def pull_handler(self):
        return self._pull_handler

    @pull_handler.setter
    def pull_handler(self, pull_handler):
        self._pull_handler = pull_handler

    @push_handler.setter
    def push_handler(self, push_handler):
        self._push_handler = push_handler

    def is_backup_server(self):
        """Return True if current server is a backup server.
        """
        if self._server_id % self._group_count == 0:
            return False
        return True

    def init_data(self, name, data_tensor=None):
        """Init data tensor on kvserver.

        Parameters
        ----------
        name : str
            data name
        data_tensor : tensor
            If the data_tensor is None, KVServer will read shared-memory by name.
        """
        # TODO(chao) : remove tmp file using new shared-tensor API
        assert len(name) > 0, 'name cannot be empty.'
        if data_tensor is not None: # Create shared-tensor
            data_type = F.reverse_data_type_dict[F.dtype(data_tensor)]
            shared_data = empty_shared_mem(name+'-kvdata-', True, data_tensor.shape, data_type)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
            self._data_store[name][:] = data_tensor[:]
            write_data_meta_to_shared_mem(name+'-kvmeta-'+str(self._machine_id), data_tensor)
        else: # Read shared-tensor
            while True:
                if (check_file_exists(name+'-kvmeta-'+str(self._machine_id))):
                    break
                else:
                    time.sleep(1) # wait until the file been created
            data_shape, data_type = read_data_meta_from_shared_mem(name+'-kvmeta-'+str(self._machine_id))
            shared_data = empty_shared_mem(name+'-kvdata-', False, data_shape, data_type)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)

    def set_partition_policy(self, name, policy_str, partition_book):
        """Set a partition policy to target data.

        Parameters
        ----------
        name : str
            data name
        policy_str : str
            partition-policy string, e.g., 'edge' or 'node'.
        partition_book : GraphPartitionBook or RangePartitionBook
            Store the partition information
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert policy_str in ('edge', 'node'), 'policy_str must be \'edge\' or \'node\'.'
        self._part_policy[name] = PartitionPolicy(policy_str, self._part_id, partition_book)

############################ KVClient ###############################

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
    def __init__(self, ip_config):
        assert rpc.get_rank() != -1, 'RPC client is not started!'
        assert check_file_exists(ip_config), 'Cannot open file: %s' % ip_config
        # Register services on client
        rpc.register_service(KVSTORE_PULL,
                             PullRequest,
                             PullResponse)
        rpc.register_service(KVSTORE_PUSH,
                             PushRequest,
                             None)
        rpc.register_service(INIT_DATA,
                             InitDataRequest,
                             InitDataResponse)
        rpc.register_service(BARRIER,
                             BarrierRequest,
                             BarrierResponse)
        rpc.register_service(REGISTER_PUSH,
                             RegisterPushHandlerRequest,
                             RegisterPushHandlerResponse)
        rpc.register_service(REGISTER_PULL,
                             RegisterPullHandlerRequest,
                             RegisterPullHandlerResponse)
        rpc.register_service(GET_SHARED,
                             GetSharedDataRequest,
                             GetSharedDataResponse)
        rpc.register_service(GET_PART_SHAPE,
                             GetPartShapeRequest,
                             GetPartShapeResponse)
        # Store the tensor data with specified data name
        self._data_store = {}
        # Store the partition information with specified data name
        self._part_policy = {}
        # Store the full data shape across kvserver
        self._full_data_shape = {}
        # Store all the data name
        self._data_name_list = []
        # Basic information
        self._server_namebook = rpc.read_ip_config(ip_config)
        self._server_count = len(self._server_namebook)
        self._group_count = self._server_namebook[0][3]
        self._machine_count = int(self._server_count / self._group_count)
        self._client_id = rpc.get_rank()
        self._machine_id = rpc.get_machine_id()
        self._part_id = self._machine_id
        self._main_server_id = self._machine_id * self._group_count
        # push and pull handler
        self._pull_handler = default_pull_handler
        self._push_handler = default_push_handler
        random.seed(time.time())

    def barrier(self):
        """Barrier for all client nodes

        This API will be blocked untill all the clients call this API.
        """
        request = BarrierRequest(BARRIER_MSG)
        # send request to all the server nodes
        for server_id in range(self._server_count):
            rpc.send_request(server_id, request)
        # recv response from all the server nodes
        for _ in range(self._server_count):
            response = rpc.recv_response()
            assert response.msg == BARRIER_MSG

    def register_push_handler(self, func):
        """Register UDF push function on server.

        client_0 will send this request to all servers, and the other
        clients will just invoke the barrier() api.

        Parameters
        ----------
        func : UDF push function
        """
        if self._client_id == 0:
            request = RegisterPushHandlerRequest(func)
            # send request to all the server nodes
            for server_id in range(self._server_count):
                rpc.send_request(server_id, request)
            # recv response from all the server nodes
            for _ in range(self._server_count):
                response = rpc.recv_response()
                assert response.msg == REGISTER_PUSH_MSG
        self._push_handler = func
        self.barrier()

    def register_pull_handler(self, func):
        """Register UDF pull function on server.

        client_0 will send this request to all servers, and the other
        clients will just invoke the barrier() api.

        Parameters
        ----------
        func : UDF pull function
        """
        if self._client_id == 0:
            request = RegisterPullHandlerRequest(func)
            # send request to all the server nodes
            for server_id in range(self._server_count):
                rpc.send_request(server_id, request)
            # recv response from all the server nodes
            for _ in range(self._server_namebook):
                response = rpc.recv_response()
                assert response.msg == REGISTER_PULL_MSG
        self._pull_handler = func
        self.barrier()

    def get_shared_data(self, partition_book):
        """Mapping shared-memory tensor from server to client.

        Parameters
        ----------
        partition_book : GraphPartitionBook or RangePartitionBook
            Store the partition information
        """
        # Get shared data from server side
        request = GetSharedDataRequest(GET_SHARED_MSG)
        rpc.send_request(self._main_server_id, request)
        response = rpc.recv_response()
        for name, meta in response.meta.items():
            shape, dtype, policy_str = meta
            shared_data = empty_shared_mem(name+'-kvdata-', False, shape, dtype)
            dlpack = shared_data.to_dlpack()
            self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
            self._part_policy[name] = PartitionPolicy(policy_str, self._part_id, partition_book)
            self._data_name_list.append(name)
        # Get full data shape across servers
        for name, meta in response.meta.items():
            shape, _, _ = meta
            data_shape = list(shape)
            data_shape[0] = 0
            request = GetPartShapeRequest(name)
            # send request to all main server nodes
            for machine_id in range(self._machine_count):
                server_id = machine_id * self._group_count
                rpc.send_request(server_id, request)
            # recv response from all the main server nodes
            for _ in range(self._machine_count):
                response = rpc.recv_response()
                data_shape[0] += response.shape[0]
            self._full_data_shape[name] = tuple(data_shape)

    def init_data(self, name, shape, dtype, policy_str, partition_book, init_func):
        """Send message to kvserver to initialize new data tensor and mapping this
        data from server side to client side.

        Parameters
        ----------
        name : str
            data name
        shape : list or tuple of int
            data shape
        dtype : dtype
            data type
        policy_str : str
            partition-policy string, e.g., 'edge' or 'node'.
        partition_book : GraphPartitionBook or RangePartitionBook
            Store the partition information
        init_func : func
            UDF init function
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert len(shape) > 0, 'shape cannot be empty'
        assert policy_str in ('edge', 'node'), 'policy_str must be \'edge\' or \'node\'.'
        shape = list(shape)
        policy_list = []
        if self._client_id == 0:
            for machine_id in range(self._machine_count):
                policy = PartitionPolicy(policy_str, machine_id, partition_book)
                part_shape = shape.copy()
                part_shape[0] = policy.get_data_size()
                request = InitDataRequest(name,
                                          tuple(part_shape), 
                                          F.reverse_data_type_dict[dtype],
                                          policy_str,
                                          init_func)
                for n in range(self._group_count):
                    server_id = machine_id * self._group_count + n
                    rpc.send_request(server_id, request)
            for _ in range(self._server_count):
                response = rpc.recv_response()
                assert response.msg == INIT_MSG
        self.barrier()
        self._part_policy[name] = PartitionPolicy(policy_str, self._part_id, partition_book)
        data_shape, data_type = read_data_meta_from_shared_mem(name+'-kvmeta-'+str(self._machine_id))
        shared_data = empty_shared_mem(name+'-kvdata-', False, data_shape, data_type)
        dlpack = shared_data.to_dlpack()
        self._data_store[name] = F.zerocopy_from_dlpack(dlpack)
        self._data_name_list.append(name)
        self._full_data_shape[name] = tuple(shape)

    def data_name_list(self):
        return self._data_name_list
    
    def get_data_meta(self, name):
        """Get meta data (data_type, data_shape, partition_policy)
        """
        assert len(name) > 0, 'name cannot be empty.'
        data_type = F.dtype(self._data_store[name])
        data_shape = self._full_data_shape[name]
        part_policy = self._part_policy[name]
        return (data_type, data_shape, part_policy)

    def push(self, name, id_tensor, data_tensor):
        """Push data to KVServer.

        Note that, the push() is an non-blocking operation that will return immediately.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the global data ID
        data_tensor : tensor
            a tensor with the same row size of data ID
        """
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        assert F.shape(id_tensor)[0] == F.shape(data_tensor)[0], 'The data must has the same row size with ID.'
        # partition data
        machine_id = self._part_policy[name].to_partid(id_tensor)
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
                local_id =  self._part_policy[name].to_local(partial_id)
                local_data = partial_data
            else: # push data to remote server
                request = PushRequest(name, partial_id, partial_data)
                # randomly select a server node in target machine for load-balance
                server_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                rpc.send_request(server_id, request)
            start += count[idx]
        if local_id is not None: # local push
            self._push_handler(self._data_store, name, local_id, local_data)

    def pull(self, name, id_tensor):
        """Pull message from KVServer.

        Parameters
        ----------
        name : str
            data name
        id_tensor : tensor
            a vector storing the ID list

        Returns
        -------
        tensor
            a data tensor with the same row size of id_tensor.
        """
        #TODO(chao) : add C++ rpc interface and add fast pull 
        assert len(name) > 0, 'name cannot be empty.'
        assert F.ndim(id_tensor) == 1, 'ID must be a vector.'
        # partition data
        machine_id = self._part_policy[name].to_partid(id_tensor)
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
                local_id =  self._part_policy[name].to_local(partial_id)
            else: # pull data from remote server
                request = PullRequest(name, partial_id)
                # randomly select a server node in target machine for load-balance
                server_id = random.randint(machine[idx]*self._group_count, (machine[idx]+1)*self._group_count-1)
                rpc.send_request(s_id, request)
                pull_count += 1
            start += count[idx]
        # recv response
        response_list = []
        if local_id is not None: # local pull
            local_data = self._pull_handler(self._data_store, name, local_id)
            server_id = self._main_server_id
            local_response = PullResponse(server_id, local_data)
            response_list.append(local_response)
        # wait response from remote server nodes
        for _ in range(pull_count):
            remote_response = rpc.recv_response()
            response_list.append(remote_response)
        # sort response by server_id and concat tensor
        response_list.sort(key=self._takeId)
        data_tensor = F.cat(seq=[response.data_tensor for response in response_list], dim=0)
        return data_tensor[back_sorted_id] # return data with original index order

    def _takeId(self, elem):
        """Used by sort response list
        """
        return elem.server_id
