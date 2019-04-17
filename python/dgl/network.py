"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from . import ndarray as nd
from . import backend as F

_init_api("dgl.network")

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
    _CAPI_SenderSendSubgraph(
        sender,
        recv_id,
        nodeflow._graph._handle,
        nd.zerocopy_from_numpy(nodeflow._layer_offsets),
        nd.zerocopy_from_numpy(nodeflow._block_offsets),
        nodeflow._node_mapping.todgltensor(),
        nodeflow._edge_mapping.todgltensor()
        if nodeflow._edge_mapping_available else None,
        nodeflow._node_data_name,
        nd.from_dlpack(F.zerocopy_to_dlpack(nodeflow._node_data))
        if nodeflow._node_data_available else None,
        nodeflow._edge_data_name,
        nd.from_dlpack(F.zerocopy_to_dlpack(nodeflow._edge_data))
        if nodeflow._edge_data_available else None)

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
    # hdl is a list of ptr
    hdl = _CAPI_ReceiverRecvSubgraph(receiver)
    return NodeFlow(graph, hdl)
