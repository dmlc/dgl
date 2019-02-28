"""Distributed sampler infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .graph_index import NodeFlowIndex
from . import utils

_init_api("dgl.network")

def _create_sampler_sender(IP, port):
    """ Create a sampler sender communicator via C socket.

    Parameters
    ----------
    IP : str
        ip address of remote trainer.
    port : int
        listen port of remote trainer.
    """
    return _CAPI_DGLSenderCreate(IP, port)

def _create_sampler_receiver(IP, port, num_sender, queue_size):
    """ Create a sampler receiver communicator via C socket.

    Parameters
    ----------
    IP : str
        ip address of remote trainer.
    port : int
        listen port of remote trainer.
    num_sender : int
        total number of sampler nodes.
    queue_size : int
        size (bytes) of message queue buffer.
    """
    return _CAPI_DGLReceiverCreate(IP, port, num_sender, queue_size)

def _send_subgraph(sender, nodeflow):
    """ Send sampled subgraph to remote trainer.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    nodeflow : NodeFlow
        sampled NodeFlow object
    """
    graph_index = nodeflow._graph._handle
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    layers_offsets = nodeflow._graph._layers.todgltensor()
    flows_offsets = nodeflow._graph._flows.todgltensor()
    _CAPI_SenderSendSubgraph(sender,
                             graph_index,
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
    rst = _CAPI_ReceiverRecvSubgraph(receiver)
    # Note that, for distributed sampler
    # we should set parent graph to None
    nodeflow_idx = NodeFlowIndex(rst(0),   # graph index handle
                                 None,     # parent graph index
                                 utils.toindex(rst(1)),  # node_mapping
                                 utils.toindex(rst(2)),  # edge_mapping
                                 utils.toindex(rst(3)),  # layers_offsets
                                 utils.toindex(rst(4)))  # flows_offsets
    return nodeflow_idx

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
