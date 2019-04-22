"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils

_init_api("dgl.network")

_CONTROL_NODEFLOW = 0
_CONTROL_END_SIGNAL = 1

def _create_sender():
    """Create a Sender communicator via C api
    """
    return _CAPI_DGLSenderCreate()

def _finalize_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLFinalizeSender(sender)

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
    _CAPI_DGLSenderAddReceiver(sender, ip_addr, port, recv_id)

def _sender_connect(sender):
    """Connect to all the Receiver

    Parameters
    ----------
    sender : ctypes.c_void_p
        C Sender handle
    """
    _CAPI_DGLSenderConnect(sender)

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
                             recv_id,
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
    _CAPI_SenderSendEndSignal(sender, recv_id)

def _create_receiver():
    """Create a Receiver communicator via C api
    """
    return _CAPI_DGLReceiverCreate()

def _finalize_receiver(receiver):
    """Finalize Receiver Communicator
    """
    _CAPI_DGLFinalizeReceiver(receiver)

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
    _CAPI_DGLReceiverWait(receiver, ip_addr, port, num_sender)

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
