"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils

_init_api("dgl.network")

def _create_sender():
    """Create a sender communicator via C socket.
    """
    return _CAPI_DGLSenderCreate()

def _finalize_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    """
    _CAPI_DGLFinalizeSender(sender)

def _add_receiver_addr(sender, ip, port, recv_id):
    """Add receiver address to receiver list

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    ip : str
        IP address of receiver
    port : int
        listen of receiver
    recv_id : int
        Receiver ID
    """
    _CAPI_DGLSenderAddReceiver(sender, ip, port, recv_id)

def _sender_connect(sender):
    """Connect to receiver

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    """
    _CAPI_DGLSenderConnect(sender)

def _send_nodeflow(sender, nodeflow, recv_id):
    """Send sampled subgraph (Nodeflow) to remote receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow : NodeFlow
        NodeFlow object
    recv_id : int
        Receiver ID
    """
    graph_handle = nodeflow._graph._handle
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    # Can we convert NDArray to tensor directly, instead of using toindex()?
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendSubgraph(sender,
                             recv_id,
                             graph_handle,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _create_receiver():
    """Create a receiver communicator via C socket.
    """
    return _CAPI_DGLReceiverCreate()

def _finalize_receiver(receiver):
    """Finalize Receiver Communicator
    """
    _CAPI_DGLFinalizeReceiver(receiver)

def _receiver_wait(receiver, ip, port, num_sender):
    """Wait all Sender connect

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    ip : str
        IP address of receiver
    port : int
        port of receiver
    num_sender : int
        number of sender
    """
    _CAPI_DGLReceiverWait(receiver, ip, port, num_sender)

def _recv_nodeflow(receiver, graph):
    """Receive sampled subgraph (NodeFlow) from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    graph : DGLGraph
        The parent graph

    Returns
    -------
    NodeFlow
        Sampled NodeFlow object
    """
    # hdl is a list of ptr
    hdl = unwrap_to_ptr_list(_CAPI_ReceiverRecvSubgraph(receiver))
    return NodeFlow(graph, hdl[0])
