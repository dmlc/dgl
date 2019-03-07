"""Distributed sampler infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils

_init_api("dgl.network")

def _create_sampler_sender(ip_addr, port):
    """ Create a sampler sender communicator via C socket.

    Parameters
    ----------
    ip_addr : str
        ip address of remote trainer.
    port : int
        listen port of remote trainer.
    """
    return _CAPI_DGLSenderCreate(ip_addr, port)

def _create_sampler_receiver(ip_addr, port, num_sender, queue_size):
    """ Create a sampler receiver communicator via C socket.

    Parameters
    ----------
    ip_addr : str
        ip address of remote trainer.
    port : int
        listen port of remote trainer.
    num_sender : int
        total number of sampler nodes.
    queue_size : int
        size (bytes) of message queue buffer.
    """
    return _CAPI_DGLReceiverCreate(ip_addr, port, num_sender, queue_size)

def _send_subgraph(sender, nodeflow):
    """ Send sampled subgraph to remote trainer.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow : NodeFlow
        sampled NodeFlow object
    """
    graph_handle = nodeflow._graph._handle
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendSubgraph(sender,
                             graph_handle,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _batch_send_subgraph(sender, nodeflow_list):
    """ Send a batch of sampled subgraph to remote trainer.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow_list : list
        a list of NodeFlow object
    """
    print(sender)
    print(nodeflow_list)
    raiseNotImplementedError("_batch_send_subgraph: not implemented!")

def _recv_subgraph(receiver):
    """ Receive sampled subgraph from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle

    Returns
    -------
    NodeFlowIndex
        a NodeFlowIndex object
    """
    hdl = unwrap_to_ptr_list(_CAPI_ReceiverRecvSubgraph(receiver))
    # Note that, for distributed sampler
    # we should set parent graph to None
    return NodeFlow(None, hdl[0])

def _batch_recv_subgraph(receiver):
    """ Receive a batch of sampled subgraph from remote sampler.

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle

    Returns
    -------
    list
        a list of NodeFlowIndex object
    """
    print(receiver)
    raiseNotImplementedError("_batch_recv_subgraph: not implemented!")

def _finalize_sampler_sender(sender):
    """ Finalize Sender communicator

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    """
    _CAPI_DGLFinalizeCommunicator(sender)

def _finalize_sampler_receiver(receiver):
    """ Finalize Receiver communicator

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    """
    _CAPI_DGLFinalizeCommunicator(receiver)
