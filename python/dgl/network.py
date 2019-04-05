"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from . import ndarray as nd
from . import backend as F

_init_api("dgl.network")

def _create_sender(ip_addr, port):
    """Create a sender communicator via C socket.

    Parameters
    ----------
    ip_addr : str
        ip address of remote trainer
    port : int
        port of remote trainer
    """
    return _CAPI_DGLSenderCreate(ip_addr, port)

def _create_receiver(ip_addr, port, num_sender):
    """Create a receiver communicator via C socket.

    Parameters
    ----------
    ip_addr : str
        ip address of remote trainer
    port : int
        listen port of remote trainer
    num_sender : int
        total number of sampler nodes
    """
    return _CAPI_DGLReceiverCreate(ip_addr, port, num_sender)

def _send_subgraph(sender, nodeflow):
    """Send sampled subgraph (Nodeflow) to remote trainer.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow : NodeFlow
        NodeFlow object
    """
    _CAPI_SenderSendSubgraph(
        sender,
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

def _recv_subgraph(receiver, graph):
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
    hdl = _CAPI_ReceiverRecvSubgraph(receiver)
    return NodeFlow(graph, hdl)

def _finalize_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    """
    _CAPI_DGLFinalizeCommunicator(sender)

def _finalize_receiver(receiver):
    """Finalize Receiver communicator

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    """
    _CAPI_DGLFinalizeCommunicator(receiver)
