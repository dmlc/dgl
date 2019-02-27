"""Distributed sampler infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .graph_index import NodeFlowIndex
from . import utils

_init_api("dgl.network")

def _create_sender(ip_addr, port):
    """ Create a sender communicator via C socket

    Parameter:
    -----------
    ip : ip address of remote machine
    port : port of remote machine
    """
    return _CAPI_DGLSenderCreate(ip_addr, port)

def _create_receiver(ip_addr, port, num_sender, queue_size):
    """ Create a receiver communicator via C socket

    Parameter:
    ----------
    ip : ip address of remote machine
    port : port of remote machine
    num_sender : total number of sender nodes
    queue_size : size (bytes) of message queue buffer
    """
    return _CAPI_DGLReceiverCreate(ip_addr, port, num_sender, queue_size)

def _send_subgraph(sender, nodeflow):
    """ Send sampled subgraph to remote trainer

    Parameter:
    ----------
    sender : C sender handle
    nodeflow : a NodeFlow object
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
    """ Send a batch of sampled subgraph to remote trainer

    Parameter:
    ----------
    sender : C sender handle
    nodeflow_list : a list of nodeflow object
    """
    graph_index_list = []
    node_mapping_list = []
    edge_mapping_list = []
    layers_offsets_list = []
    flows_offsets_list = []
    for nodeflow in nodeflow_list:
        graph_index_list.append(nodeflow._graph._handle)
        node_mapping_list.append(nodeflow._node_mapping.todgltensor())
        edge_mapping_list.append(nodeflow._edge_mapping.todgltensor())
        layers_offsets_list.append(nodeflow._graph._layers.todgltensor())
        flows_offsets_list.append(nodeflow._graph._flows.todgltensor())
    _CAPI_SenderBatchSendSubgraph(sender,
                                  graph_index_list,
                                  node_mapping_list,
                                  edge_mapping_list,
                                  layers_offsets_list,
                                  flows_offsets_list)

def _recv_subgraph(receiver):
    """ Receive sampled subgraph from remote sampler

    Parameter
    ----------
    receiver : C receiver handle

    Return
    -------
    nodeflow_idx : a NodeFlowIndex object
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
    """ Receive a batch of sampled subgraph from remote sampler

    Parameter
    ----------
    receiver :  C receiver handle

    Return
    -------
    nodeflow_idx_list : a list of NodeFlowIndex object
    """
    rst = _CAPI_ReceiverBatchRecvSubgraph(receiver)
    # Note that, for distributed sampler
    # we should set parent graph to None
    nodeflow_idx = NodeFlowIndex(rst(0),   # graph index handle
                                 None,     # parent graph index
                                 utils.toindex(rst(1)),  # node_mapping
                                 utils.toindex(rst(2)),  # edge_mapping
                                 utils.toindex(rst(3)),  # layers_offsets
                                 utils.toindex(rst(4)))  # flows_offsets
    return nodeflow_idx

def _finalize_sender(sender):
    """ Finalize Sender communicator

    Parameter
    ----------
    sender : C sender handle
    """
    _CAPI_DGLFinalizeCommunicator(sender)

def _finalize_receiver(receiver):
    """ Finalize Receiver communicator

    Parameter
    ----------
    receiver : C receiver handle
    """
    _CAPI_DGLFinalizeCommunicator(receiver)
