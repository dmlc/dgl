"""DGL Distributed Training Infrastructure."""
from __future__ import absolute_import

import socket

from ._ffi.function import _init_api
from .nodeflow import NodeFlow
from .utils import unwrap_to_ptr_list
from . import utils

_init_api("dgl.network")

def get_ip():
    """Get local IP address
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def find_free_port():
    """Find free port
    """
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

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

def _send_local_ip_port(sender):
    """Send usable ip and port to remote.

    Parameters
    ----------
    sender : ctypes.c_void_p
        C sender handle

    Returns
    -------
    string
        local IP address
    int
        local port
    """
    ip = get_ip()
    port = find_free_port()
    _CAPI_DGLSendIPandPort(sender, ip, port)
    return ip, port

def _recv_ip_port(receiver):
    """Receive ip and port from remote GraphStore Client

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle
    """
    ip = _CAPI_DGLRecvIP(receiver)
    port = _CAPI_DGLRecvPort(receiver)
    return ip, port

def _recv_graph_store_msg(receiver):
    """Receive message of Graphstore

    Parameters
    ----------
    receiver : ctypes.c_void_p
        C receiver handle

    Return
    ------
    GraphStoreMsg
        message of GraphStore
    """
    pass

def _send_node_feats(sender, node_feats):
    """Send node features
    """
    pass

def _push(sender, feat_name, node_ids, node_feats):
    """Send feature name, node IDs, and node features to the 
    GraphStore server, which uses these data to updating GraphStore.

    Parameters
    ----------
    sender : ctypes.void.c_void_p
        C sender handle
    feat_name : string
        feature name
    node_ids : tensor
        A tensor of node IDs
    node_feats : tensor
        tensor of node feature
    """
    pass

def _pull(sender, feat_name, node_ids):
    """Request the feature values associated with corresponding 
    feature name and node IDs.

    Parameters
    ----------
    sender : ctypes.void.c_void_p
        C sender handle
    feat_name : string
        feature name
    node_ids : tensor
        A tensor of node IDs
    """
    pass
