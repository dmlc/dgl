"""RPC components."""
import abc
import pickle

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from ..base import DGLError, dgl_warning
from .. import backend as F

REQUEST_CLASS_TO_SERVICE_ID = {}
RESPONSE_CLASS_TO_SERVICE_ID = {}
SERVICE_ID_TO_PROPERTY = {}

def get_rank():
    """Get the rank of this process.

    If the process is a client, this is equal to client ID. Otherwise, the process
    is a server and this is equal to server ID.

    Returns
    -------
    int
        Rank value
    """
    pass

def incr_msg_seq():
    """Increment the message sequence number and return the old one.

    Returns
    -------
    long
        Message sequence number
    """
    pass

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
    # TODO: C registry

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

        Must be inherited by subclasses.
        """
        pass

    @abc.abstractmethod
    def __setstate__(self, state):
        """Construct the request object from serialized states.

        Must be inherited by subclasses.
        """
        pass

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
        pass

    @property
    def service_id(self):
        """Get service ID."""
        cls = self.__class__
        sid = REQUEST_CLASS_TO_SERVICE_ID.get(cls, None)
        if sid is None:
            raise DGLError('Request class {} has not been registered as a service.'.format(cls))
        return sid

class Response:
    """Base response class."""

    @abc.abstractmethod
    def __getstate__(self):
        """Get serializable states.

        Must be inherited by subclasses.
        """
        pass

    @abc.abstractmethod
    def __setstate__(self, state):
        """Construct the response object from serialized states.

        Must be inherited by subclasses.
        """
        pass

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
    for i, st in enumerate(state):
        if F.is_tensor(st):
            array_state.append(st)
        else:
            nonarray_state.append(st)
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
    for i, st in zip(pos, nonarray_state):
        state[i] = st
    if len(tensors) != 0:
        j = 0
        for i in range(len(state)):
            if state[i] is None:
                state[i] = tensors[j]
                j += 1
    if len(state) == 1:
        state = state[0]
    else:
        state = tuple(state)
    obj = cls.__new__()
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
            _CAPI_DGLCreateRPCMessage,
            int(service_id),
            int(msg_seq),
            int(client_id),
            int(server_id),
            data,
            [F.zerocopy_to_dgl_ndarray(tsor) for tsor in tensors])

    @property
    def service_id(self):
        """Get service ID."""
        return _CAPI_DGLRPCMessageGetReqType(self)

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
        rst = _CAPI_DGLRPCMessageGetData(self)
        return [F.zerocopy_from_dgl_ndarray(tsor) for tsor in rst]

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
        The server ID.
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
    msg = recv_rpc_message(timeout)
    _, res_cls = SERVICE_ID_TO_PROPERTY[msg.service_id]
    if res_cls is None:
        raise DGLError('Got response message from service ID {}, '
                       'but no response class is registered.'.format(msg.service_id))
    res = deserialize_from_payload(res_cls, msg.data, msg.tensors)
    if res.client_id != get_rank():
        raise DGLError('Got reponse of request sent by client {}, '
                       'different from my rank {}!'.format(res.client_id, get_rank()))
    return res

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
    pass

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
    pass

_init_api("dgl.distributed.rpc")
