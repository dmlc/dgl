"""RPC components. They are typically functions or utilities used by both
server and clients."""
import abc
import pickle

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from ..base import DGLError
from .. import backend as F

__all__ = ['set_rank', 'get_rank', 'Request', 'Response', 'register_service', \
'create_sender', 'create_receiver', 'finalize_sender', 'finalize_receiver', \
'receiver_wait', 'add_receiver_addr', 'sender_connect', 'read_ip_config', \
'get_num_machines', 'set_num_machines', 'get_machine_id', 'set_machine_id', \
'send_request', 'recv_request', 'send_response', 'recv_response', 'remote_call']

REQUEST_CLASS_TO_SERVICE_ID = {}
RESPONSE_CLASS_TO_SERVICE_ID = {}
SERVICE_ID_TO_PROPERTY = {}

def read_ip_config(filename):
    """Read network configuration information of server from file.

    The format of configuration file should be:

        [ip] [base_port] [server_count]

        172.31.40.143 30050 2
        172.31.36.140 30050 2
        172.31.47.147 30050 2
        172.31.30.180 30050 2

    Note that, DGL supports multiple backup servers that shares data with each others
    on the same machine via shared-memory tensor. The server_count should be >= 1. For example,
    if we set server_count to 5, it means that we have 1 main server and 4 backup servers on
    current machine.

    Parameters
    ----------
    filename : str
        Path of IP configuration file.

    Returns
    -------
    dict
        server namebook.
        The key is server_id (int)
        The value is [machine_id, ip, port, group_count] ([int, str, int, int])

        e.g.,

          {0:[0, '172.31.40.143', 30050, 2],
           1:[0, '172.31.40.143', 30051, 2],
           2:[1, '172.31.36.140', 30050, 2],
           3:[1, '172.31.36.140', 30051, 2],
           4:[2, '172.31.47.147', 30050, 2],
           5:[2, '172.31.47.147', 30051, 2],
           6:[3, '172.31.30.180', 30050, 2],
           7:[3, '172.31.30.180', 30051, 2]}
    """
    assert len(filename) > 0, 'filename cannot be empty.'
    server_namebook = {}
    try:
        server_id = 0
        machine_id = 0
        lines = [line.rstrip('\n') for line in open(filename)]
        for line in lines:
            ip_addr, port, server_count = line.split(' ')
            for s_count in range(int(server_count)):
                server_namebook[server_id] = \
                [int(machine_id), ip_addr, int(port)+s_count, int(server_count)]
                server_id += 1
            machine_id += 1
    except ValueError:
        print("Error: data format on each line should be: [ip] [base_port] [server_count]")
    return server_namebook

def create_sender(max_queue_size, net_type):
    """Create rpc sender of this process.

    Parameters
    ----------
    max_queue_size : int
        Maximal size (bytes) of network queue buffer.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    _CAPI_DGLRPCCreateSender(int(max_queue_size), net_type)

def create_receiver(max_queue_size, net_type):
    """Create rpc receiver of this process.

    Parameters
    ----------
    max_queue_size : int
        Maximal size (bytes) of network queue buffer.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    _CAPI_DGLRPCCreateReceiver(int(max_queue_size), net_type)

def finalize_sender():
    """Finalize rpc sender of this process.
    """
    _CAPI_DGLRPCFinalizeSender()

def finalize_receiver():
    """Finalize rpc receiver of this process.
    """
    _CAPI_DGLRPCFinalizeReceiver()

def receiver_wait(ip_addr, port, num_senders):
    """Wait all of the senders' connections.

    This api will be blocked until all the senders connect to the receiver.

    Parameters
    ----------
    ip_addr : str
        receiver's IP address, e,g, '192.168.8.12'
    port : int
        receiver's port
    num_senders : int
        total number of senders
    """
    _CAPI_DGLRPCReceiverWait(ip_addr, int(port), int(num_senders))

def add_receiver_addr(ip_addr, port, recv_id):
    """Add Receiver's IP address to sender's namebook.

    Parameters
    ----------
    ip_addr : str
        receiver's IP address, e,g, '192.168.8.12'
    port : int
        receiver's listening port
    recv_id : int
        receiver's ID
    """
    _CAPI_DGLRPCAddReceiver(ip_addr, int(port), int(recv_id))

def sender_connect():
    """Connect to all the receivers.
    """
    _CAPI_DGLRPCSenderConnect()

def set_rank(rank):
    """Set the rank of this process.

    If the process is a client, this is equal to client ID. Otherwise, the process
    is a server and this is equal to server ID.

    Parameters
    ----------
    rank : int
        Rank value
    """
    _CAPI_DGLRPCSetRank(int(rank))

def get_rank():
    """Get the rank of this process.

    If the process is a client, this is equal to client ID. Otherwise, the process
    is a server and this is equal to server ID.

    Returns
    -------
    int
        Rank value
    """
    return _CAPI_DGLRPCGetRank()

def set_machine_id(machine_id):
    """Set current machine ID

    Parameters
    ----------
    machine_id : int
        Current machine ID
    """
    _CAPI_DGLRPCSetMachineID(int(machine_id))

def get_machine_id():
    """Get current machine ID

    Returns
    -------
    int
        machine ID
    """
    return _CAPI_DGLRPCGetMachineID()

def set_num_machines(num_machines):
    """Set number of machine

    Parameters
    ----------
    num_machines : int
        Number of machine
    """
    _CAPI_DGLRPCSetNumMachines(int(num_machines))

def get_num_machines():
    """Get number of machines

    Returns
    -------
    int
        number of machines
    """
    return _CAPI_DGLRPCGetNumMachines()

def set_num_server(num_server):
    """Set the total number of server.
    """
    _CAPI_DGLRPCSetNumServer(int(num_server))

def get_num_server():
    """Get the total number of server.
    """
    return _CAPI_DGLRPCGetNumServer()

def incr_msg_seq():
    """Increment the message sequence number and return the old one.

    Returns
    -------
    long
        Message sequence number
    """
    return _CAPI_DGLRPCIncrMsgSeq()

def get_msg_seq():
    """Get the current message sequence number.

    Returns
    -------
    long
        Message sequence number
    """
    return _CAPI_DGLRPCGetMsgSeq()

def set_msg_seq(msg_seq):
    """Set the current message sequence number.

    Parameters
    ----------
    msg_seq : int
        sequence number of current rpc message.
    """
    _CAPI_DGLRPCSetMsgSeq(int(msg_seq))

def register_service(service_id, req_cls, res_cls=None):
    """Register a service to RPC.

    Parameter
    ---------
    service_id : int
        Service ID.
    req_cls : class
        Request class.
    res_cls : class, optional
        Response class. If none, the service has no response.
    """
    REQUEST_CLASS_TO_SERVICE_ID[req_cls] = service_id
    if res_cls is not None:
        RESPONSE_CLASS_TO_SERVICE_ID[res_cls] = service_id
    SERVICE_ID_TO_PROPERTY[service_id] = (req_cls, res_cls)

def get_service_property(service_id):
    """Get service property.

    Parameters
    ----------
    service_id : int
        Service ID.

    Returns
    -------
    (class, class)
        (Request class, Response class)
    """
    return SERVICE_ID_TO_PROPERTY[service_id]

class Request:
    """Base request class"""

    @abc.abstractmethod
    def __getstate__(self):
        """Get serializable states.

        Must be inherited by subclasses. For array members, return them as
        individual return values (i.e., do not put them in containers like
        dictionary or list).
        """

    @abc.abstractmethod
    def __setstate__(self, state):
        """Construct the request object from serialized states.

        Must be inherited by subclasses.
        """

    @abc.abstractmethod
    def process_request(self, server_state):
        """Server-side function to process the request.

        Must be inherited by subclasses.

        Parameters
        ----------
        server_state : ServerState
            Server state data.

        Returns
        -------
        Response
            Response of this request or None if no response.
        """

    @property
    def service_id(self):
        """Get service ID."""
        cls = self.__class__
        sid = REQUEST_CLASS_TO_SERVICE_ID.get(cls, None)
        if sid is None:
            raise DGLError('Request class {} has not been registered as a service.'.format(cls))
        return sid

class Response:
    """Base response class"""

    @abc.abstractmethod
    def __getstate__(self):
        """Get serializable states.

        Must be inherited by subclasses. For array members, return them as
        individual return values (i.e., do not put them in containers like
        dictionary or list).
        """

    @abc.abstractmethod
    def __setstate__(self, state):
        """Construct the response object from serialized states.

        Must be inherited by subclasses.
        """

    @property
    def service_id(self):
        """Get service ID."""
        cls = self.__class__
        sid = RESPONSE_CLASS_TO_SERVICE_ID.get(cls, None)
        if sid is None:
            raise DGLError('Response class {} has not been registered as a service.'.format(cls))
        return sid

def serialize_to_payload(serializable):
    """Serialize an object to payloads.

    The object must have implemented the __getstate__ function.

    Parameters
    ----------
    serializable : object
        Any serializable object.

    Returns
    -------
    bytearray
        Serialized payload buffer.
    list[Tensor]
        A list of tensor payloads.
    """
    state = serializable.__getstate__()
    if not isinstance(state, tuple):
        state = (state,)
    nonarray_pos = []
    nonarray_state = []
    array_state = []
    for i, arr_state in enumerate(state):
        if F.is_tensor(arr_state):
            array_state.append(arr_state)
        else:
            nonarray_state.append(arr_state)
            nonarray_pos.append(i)
    data = bytearray(pickle.dumps((nonarray_pos, nonarray_state)))
    return data, array_state

def deserialize_from_payload(cls, data, tensors):
    """Deserialize and reconstruct the object from payload.

    The object must have implemented the __setstate__ function.

    Parameters
    ----------
    cls : class
        The object class.
    data : bytearray
        Serialized data buffer.
    tensors : list[Tensor]
        A list of tensor payloads.

    Returns
    -------
    object
        De-serialized object of class cls.
    """
    pos, nonarray_state = pickle.loads(data)
    state = [None] * (len(nonarray_state) + len(tensors))
    for i, no_state in zip(pos, nonarray_state):
        state[i] = no_state
    if len(tensors) != 0:
        j = 0
        state_len = len(state)
        for i in range(state_len):
            if state[i] is None:
                state[i] = tensors[j]
                j += 1
    if len(state) == 1:
        state = state[0]
    else:
        state = tuple(state)
    obj = cls.__new__(cls)
    obj.__setstate__(state)
    return obj

@register_object('rpc.RPCMessage')
class RPCMessage(ObjectBase):
    """Serialized RPC message that can be sent to remote processes.

    This class can be used as argument or return value for C API.

    Attributes
    ----------
    service_id : int
        The remote service ID the message wishes to invoke.
    msg_seq : int
        Sequence number of this message.
    client_id : int
        The client ID.
    server_id : int
        The server ID.
    data : bytearray
        Payload buffer carried by this request.
    tensors : list[tensor]
        Extra payloads in the form of tensors.
    """
    def __init__(self, service_id, msg_seq, client_id, server_id, data, tensors):
        self.__init_handle_by_constructor__(
            _CAPI_DGLRPCCreateRPCMessage,
            int(service_id),
            int(msg_seq),
            int(client_id),
            int(server_id),
            data,
            [F.zerocopy_to_dgl_ndarray(tsor) for tsor in tensors])

    @property
    def service_id(self):
        """Get service ID."""
        return _CAPI_DGLRPCMessageGetServiceId(self)

    @property
    def msg_seq(self):
        """Get message sequence number."""
        return _CAPI_DGLRPCMessageGetMsgSeq(self)

    @property
    def client_id(self):
        """Get client ID."""
        return _CAPI_DGLRPCMessageGetClientId(self)

    @property
    def server_id(self):
        """Get server ID."""
        return _CAPI_DGLRPCMessageGetServerId(self)

    @property
    def data(self):
        """Get payload buffer."""
        return _CAPI_DGLRPCMessageGetData(self)

    @property
    def tensors(self):
        """Get tensor payloads."""
        rst = _CAPI_DGLRPCMessageGetTensors(self)
        return [F.zerocopy_from_dgl_ndarray(tsor.data) for tsor in rst]

def send_request(target, request):
    """Send one request to the target server.

    Serialize the given request object to an :class:`RPCMessage` and send it
    out.

    The operation is non-blocking -- it does not guarantee the payloads have
    reached the target or even have left the sender process. However,
    all the payloads (i.e., data and arrays) can be safely freed after this
    function returns.

    Parameters
    ----------
    target : int
        ID of target server.
    request : Request
        The request to send.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    service_id = request.service_id
    msg_seq = incr_msg_seq()
    client_id = get_rank()
    server_id = target
    data, tensors = serialize_to_payload(request)
    msg = RPCMessage(service_id, msg_seq, client_id, server_id, data, tensors)
    send_rpc_message(msg)

def send_response(target, response):
    """Send one response to the target client.

    Serialize the given response object to an :class:`RPCMessage` and send it
    out.

    The operation is non-blocking -- it does not guarantee the payloads have
    reached the target or even have left the sender process. However,
    all the payloads (i.e., data and arrays) can be safely freed after this
    function returns.

    Parameters
    ----------
    target : int
        ID of target client.
    response : Response
        The response to send.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    service_id = response.service_id
    msg_seq = get_msg_seq()
    client_id = target
    server_id = get_rank()
    data, tensors = serialize_to_payload(response)
    msg = RPCMessage(service_id, msg_seq, client_id, server_id, data, tensors)
    send_rpc_message(msg)

def recv_request(timeout=0):
    """Receive one request.

    Receive one :class:`RPCMessage` and de-serialize it into a proper Request object.

    The operation is blocking -- it returns when it receives any message
    or it times out.

    Parameters
    ----------
    timeout : int, optional
        The timeout value in milliseconds. If zero, wait indefinitely.

    Returns
    -------
    req : request
        One request received from the target, or None if it times out.
    client_id : int
        Client' ID received from the target.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    # TODO(chao): handle timeout
    msg = recv_rpc_message(timeout)
    if msg is None:
        return None
    set_msg_seq(msg.msg_seq)
    req_cls, _ = SERVICE_ID_TO_PROPERTY[msg.service_id]
    if req_cls is None:
        raise DGLError('Got request message from service ID {}, '
                       'but no request class is registered.'.format(msg.service_id))
    req = deserialize_from_payload(req_cls, msg.data, msg.tensors)
    if msg.server_id != get_rank():
        raise DGLError('Got request sent to server {}, '
                       'different from my rank {}!'.format(msg.server_id, get_rank()))
    return req, msg.client_id

def recv_response(timeout=0):
    """Receive one response.

    Receive one :class:`RPCMessage` and de-serialize it into a proper Response object.

    The operation is blocking -- it returns when it receives any message
    or it times out.

    Parameters
    ----------
    timeout : int, optional
        The timeout value in milliseconds. If zero, wait indefinitely.

    Returns
    -------
    res : Response
        One response received from the target, or None if it times out.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    # TODO(chao): handle timeout
    print("aaaaa")
    msg = recv_rpc_message(timeout)
    if msg is None:
        return None
    _, res_cls = SERVICE_ID_TO_PROPERTY[msg.service_id]
    print("bbbbb")
    if res_cls is None:
        raise DGLError('Got response message from service ID {}, '
                       'but no response class is registered.'.format(msg.service_id))
    res = deserialize_from_payload(res_cls, msg.data, msg.tensors)
    print("ccccc")
    if msg.client_id != get_rank():
        raise DGLError('Got reponse of request sent by client {}, '
                       'different from my rank {}!'.format(msg.client_id, get_rank()))
    print("ddddd")
    return res

def remote_call(target_and_requests, timeout=0):
    """Invoke registered services on remote servers and collect responses.

    The operation is blocking -- it returns when it receives all responses
    or it times out.

    If the target server state is available locally, it invokes local computation
    to calculate the response.

    Parameters
    ----------
    target_and_requests : list[(int, Request)]
        A list of requests and the server they should be sent to.
    timeout : int, optional
        The timeout value in milliseconds. If zero, wait indefinitely.

    Returns
    -------
    list[Response]
        Responses for each target-request pair. If the request does not have
        response, None is placed.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    # TODO(chao): handle timeout
    all_res = [None] * len(target_and_requests)
    msgseq2pos = {}
    num_res = 0
    myrank = get_rank()
    for pos, (target, request) in enumerate(target_and_requests):
        # send request
        service_id = request.service_id
        msg_seq = incr_msg_seq()
        client_id = get_rank()
        server_id = target
        data, tensors = serialize_to_payload(request)
        msg = RPCMessage(service_id, msg_seq, client_id, server_id, data, tensors)
        send_rpc_message(msg)
        # check if has response
        res_cls = get_service_property(service_id)[1]
        if res_cls is not None:
            num_res += 1
            msgseq2pos[msg_seq] = pos
    while num_res != 0:
        # recv response
        msg = recv_rpc_message(timeout)
        num_res -= 1
        _, res_cls = SERVICE_ID_TO_PROPERTY[msg.service_id]
        if res_cls is None:
            raise DGLError('Got response message from service ID {}, '
                           'but no response class is registered.'.format(msg.service_id))
        res = deserialize_from_payload(res_cls, msg.data, msg.tensors)
        if msg.client_id != myrank:
            raise DGLError('Got reponse of request sent by client {}, '
                           'different from my rank {}!'.format(msg.client_id, myrank))
        # set response
        all_res[msgseq2pos[msg.msg_seq]] = res
    return all_res

def send_rpc_message(msg):
    """Send one message to the target server.

    The operation is non-blocking -- it does not guarantee the payloads have
    reached the target or even have left the sender process. However,
    all the payloads (i.e., data and arrays) can be safely freed after this
    function returns.

    The data buffer in the requst will be copied to internal buffer for actual
    transmission, while no memory copy for tensor payloads (a.k.a. zero-copy).
    The underlying sending threads will hold references to the tensors until
    the contents have been transmitted.

    Parameters
    ----------
    msg : RPCMessage
        The message to send.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    _CAPI_DGLRPCSendRPCMessage(msg)

def recv_rpc_message(timeout=0):
    """Receive one message.

    The operation is blocking -- it returns when it receives any message
    or it times out.

    Parameters
    ----------
    timeout : int, optional
        The timeout value in milliseconds. If zero, wait indefinitely.

    Returns
    -------
    msg : RPCMessage
        One rpc message received from the target, or None if it times out.

    Raises
    ------
    ConnectionError if there is any problem with the connection.
    """
    msg = _CAPI_DGLRPCCreateEmptyRPCMessage()
    _CAPI_DGLRPCRecvRPCMessage(timeout, msg)
    return msg

def finalize_server():
    """Finalize resources of current server
    """
    finalize_sender()
    finalize_receiver()
    print("Server (%d) shutdown." % get_rank())

############### Some basic services will be defined here #############

CLIENT_REGISTER = 22451

class ClientRegisterRequest(Request):
    """This request will send client's ip to server.

    Parameters
    ----------
    ip_addr : str
        client's IP address
    """
    def __init__(self, ip_addr):
        self.ip_addr = ip_addr

    def __getstate__(self):
        return self.ip_addr

    def __setstate__(self, state):
        self.ip_addr = state

    def process_request(self, server_state):
        return None # do nothing

class ClientRegisterResponse(Response):
    """This response will send assigned ID to client.

    Parameters
    ----------
    ID : int
        client's ID
    """
    def __init__(self, client_id):
        self.client_id = client_id

    def __getstate__(self):
        return self.client_id

    def __setstate__(self, state):
        self.client_id = state


SHUT_DOWN_SERVER = 22452

class ShutDownRequest(Request):
    """Client send this request to shut-down a server.

    This request has no response.

    Parameters
    ----------
    client_id : int
        client's ID
    """
    def __init__(self, client_id):
        self.client_id = client_id

    def __getstate__(self):
        return self.client_id

    def __setstate__(self, state):
        self.client_id = state

    def process_request(self, server_state):
        assert self.client_id == 0
        finalize_server()
        exit()


_init_api("dgl.distributed.rpc")
