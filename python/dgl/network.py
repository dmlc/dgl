"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from torch.utils import dlpack

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils
from . import ndarray


import time
import signal
from collections import namedtuple

_init_api("dgl.network")

_CONTROL_NODEFLOW = 0
_CONTROL_END_SIGNAL = 1

_PUSH_MSG = 2
_PULL_MSG = 3
_PULL_BACK_MSG = 4

_WAIT_TIME_SEC = 3  # 3 seconds for socket sync

KVStoreMsg = namedtuple("KVStoreMsg", "type rank name id data")
"""Message of DGL kvstore

Data Field
----------
type : MessageType
    Type of DGL kvstore message (PUSH, PULL, PULL_BACK)
rank : int
    sender's ID
name : str
    data name
id : tensor (mx.ndarray or torch.tensor)
    a vector storing the IDs
data : tensor (mx.ndarray or torch.tensor)
    a data matrix with the same row size of id
"""

def SocketSync():
    time.sleep(_WAIT_TIME_SEC)

def keyboardInterruptHandler(signal, frame):
    """Users can use [Ctl + C] to exit loop service
    """
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up DGL ...".format(signal))
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

def _create_sender():
    """Create a Sender communicator via C api
    """
    return _CAPI_DGLSenderCreate()

def _create_receiver():
    """Create a Receiver communicator via C api
    """
    return _CAPI_DGLReceiverCreate()

def _finalize_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLFinalizeSender(sender)

def _finalize_receiver(receiver):
    """Finalize Receiver Communicator
    """
    _CAPI_DGLFinalizeReceiver(receiver)

def _add_receiver_addr(sender, ip_addr, port, recv_id):
    """Add Receiver IP address to namebook

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    ip_addr : str
        IP address of Receiver
    port : int
        listen of Receiver
    recv_id : int
        Receiver ID
    """
    _CAPI_DGLSenderAddReceiver(sender, ip_addr, int(port), int(recv_id))

def _sender_connect(sender):
    """Connect to all the Receiver

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLSenderConnect(sender)

def _receiver_wait(receiver, ip_addr, port, num_sender):
    """Wait all Sender to connect..

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    ip_addr : str
        IP address of Receiver
    port : int
        port of Receiver
    num_sender : int
        total number of Sender
    """
    _CAPI_DGLReceiverWait(receiver, ip_addr, int(port), int(num_sender))

def _send_nodeflow(sender, nodeflow, recv_id):
    """Send sampled subgraph (Nodeflow) to remote Receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    nodeflow : NodeFlow
        NodeFlow object
    recv_id : int
        Receiver ID
    """
    graph_handle = nodeflow._graph._handle
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendSubgraph(sender,
                             int(recv_id),
                             graph_handle,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _send_end_signal(sender, recv_id):
    """Send an epoch-end signal to remote Receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    recv_id : int
        Receiver ID
    """
    _CAPI_SenderSendEndSignal(sender, int(recv_id))

def _recv_nodeflow(receiver, graph):
    """Receive sampled subgraph (NodeFlow) from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle
    graph : DGLGraph
        The parent graph

    Returns
    -------
    NodeFlow
        Sampled NodeFlow object
    """
    res = _CAPI_ReceiverRecvSubgraph(receiver)
    if isinstance(res, int):
        if res == _CONTROL_END_SIGNAL:
            return _CONTROL_END_SIGNAL
        else:
            raise RuntimeError('Got unexpected control code {}'.format(res))
    else:
        hdl = unwrap_to_ptr_list(res)
        return NodeFlow(graph, hdl[0])

def _send_kv_msg(sender, msg, recv_id):
    """Send kvstore message.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    msg : KVStoreMsg
        kvstore message
    recv_id : int
        receiver's ID
    """
    ID = ndarray.from_dlpack(dlpack.to_dlpack(msg.id.contiguous()))
    data = ndarray.from_dlpack(dlpack.to_dlpack(msg.data.contiguous()))
    if msg.type == _PUSH_MSG:
        _CAPI_SenderSendKVMsg(sender, int(recv_id), msg.type, msg.rank, msg.name, ID, data)
    elif msg.type == _PULL_MSG:  # PULL
        _CAPI_SenderSendKVMsg(sender, int(recv_id), msg.type, msg.rank, msg.name, ID)
    else:
        raise RuntimeError('Unknown message type: %d' % msg.type)

def _recv_kv_msg(receiver):
    """Receive kvstore message

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C Receiver handle

    Return
    ------
    KVStoreMsg
        kvstore message
    """
    res = _CAPI_ReceiverRecvKVMsg(receiver)
    hdl = unwrap_to_ptr_list(res)
    msg_type = _CAPI_ReceiverGetKVMsgType(hdl[0])
    rank = _CAPI_ReceiverGetKVMsgRank(hdl[0])
    name = _CAPI_ReceiverGetKVMsgName(hdl[0])
    ID = dlpack.from_dlpack(_CAPI_ReceiverGetKVMsgID(hdl[0]).to_dlpack())
    if msg_type == _PUSH_MSG:
        data = dlpack.from_dlpack(_CAPI_ReceiverGetKVMsgData(hdl[0]).to_dlpack())
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=name,
            id=ID,
            data=data)
        return msg
    elif msg_type == _PULL_MSG:
        msg = KVStoreMsg(
            type=msg_type,
            rank=rank,
            name=name,
            id=ID)
        return msg
    else:
        raise RuntimeError('Unknown message type: %d' % msg.type)
