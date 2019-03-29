"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils

_init_api("dgl.network")

############################# Distributed Sampler #############################

def _create_sampler_sender(ip_addr, port):
    """Create a sender communicator via C socket.

    Parameters
    ----------
    ip_addr : str
        ip address of remote trainer
    port : int
        port of remote trainer
    """
    return _CAPI_DGLSenderCreate(ip_addr, port)

def _create_sampler_receiver(ip_addr, port, num_sender):
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
    graph_handle = nodeflow._graph._handle
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    # Can we convert NDArray to tensor directly, instead of using toindex()?
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendSubgraph(sender,
                             graph_handle,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _batch_send_subgraph(sender, nodeflow_list):
    """Send a batch of sampled subgraph (Nodeflow) to remote trainer.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow_list : list
        a list of NodeFlow objects
    """
    assert(sender)
    assert(nodeflow_list)
    raiseNotImplementedError("_batch_send_subgraph: not implemented!")

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
    hdl = unwrap_to_ptr_list(_CAPI_ReceiverRecvSubgraph(receiver))
    return NodeFlow(graph, hdl[0])

def _batch_recv_subgraph(receiver, graph):
    """Receive a batch of sampled subgraph from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    graph : DGLGraph
        The parent graph

    Returns
    -------
    list
        a list of Sampled NodeFlow objects
    """
    assert(receiver)
    assert(graph)
    raiseNotImplementedError("_batch_recv_subgraph: not implemented!")

def _finalize_sampler_sender(sender):
    """Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    """
    _CAPI_DGLFinalizeCommunicator(sender)

def _finalize_sampler_receiver(receiver):
    """Finalize Receiver communicator

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    """
    _CAPI_DGLFinalizeCommunicator(receiver)
