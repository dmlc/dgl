"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from . import utils

_init_api("dgl.network")


################################ Common Network Components ##################################

def _create_sender(net_type):
    """Create a Sender communicator via C api

    Parameters
    ----------
    net_type : str
        'socket' or 'mpi'
    """
    assert net_type in ('socket', 'mpi'), 'Unknown network type.'
    return _CAPI_DGLSenderCreate(net_type)

def _create_receiver(net_type):
    """Create a Receiver communicator via C api

    Parameters
    ----------
    net_type : str
        'socket' or 'mpi'
    """
    assert net_type in ('socket', 'mpi'), 'Unknown network type.'
    return _CAPI_DGLReceiverCreate(net_type)

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
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
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
    assert num_sender >= 0, 'num_sender cannot be a negative number.'
    _CAPI_DGLReceiverWait(receiver, ip_addr, int(port), int(num_sender))


################################ Distributed Sampler Components ################################


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
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
    gidx = nodeflow._graph
    node_mapping = nodeflow._node_mapping.todgltensor()
    edge_mapping = nodeflow._edge_mapping.todgltensor()
    layers_offsets = utils.toindex(nodeflow._layer_offsets).todgltensor()
    flows_offsets = utils.toindex(nodeflow._block_offsets).todgltensor()
    _CAPI_SenderSendNodeFlow(sender,
                             int(recv_id),
                             gidx,
                             node_mapping,
                             edge_mapping,
                             layers_offsets,
                             flows_offsets)

def _send_sampler_end_signal(sender, recv_id):
    """Send an epoch-end signal to remote Receiver.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle
    recv_id : int
        Receiver ID
    """
    assert recv_id >= 0, 'recv_id cannot be a negative number.'
    _CAPI_SenderSendSamplerEndSignal(sender, int(recv_id))

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
    NodeFlow or an end-signal
    """
    res = _CAPI_ReceiverRecvNodeFlow(receiver)
    if isinstance(res, int):
        return res
    else:
        return NodeFlow(graph, res)
