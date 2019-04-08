# This file contains DGL distributed graph store APIs.
from ...network import _create_sender, _create_receiver
from ...network import _finalize_sender, _finalize_receiver
from ...network import _recv_ip_port, _send_local_ip_port
from ...network import _recv_graph_store_msg, _send_graph_store_msg
from ...network import _pull, _push

import time
import signal
from enum import Enum
from collections import namedtuple

_WAIT_TIME_SEC = 3  # 3 seconds for socket sync

def SocketSync():
    time.sleep(_WAIT_TIME_SEC)

def keyboardInterruptHandler(signal, frame):
    """Users use [Ctl + C] to exit the GraphStore service
    """
    print("KeyboardInterrupt (ID: {}) has been caught. Cleaning up DGL ...".format(signal))
    exit(0)

signal.signal(signal.SIGINT, keyboardInterruptHandler)

GraphStoreMsg = namedtuple("GraphStoreMsg", "type feat_name node_ids node_feats")

class MessageType(Enum):
    """For now, we only support two types of message
    """
    PUSH = 0
    PULL = 1

class GraphStore(object):
    """GraphStore is a distributed storage for DGL distributed training.

    In practice, we use GraphStore to hold large-scale node features or node embedding 
    in standalone machine with big memory capability. 

    GraphStore provides three APIs for GraphStore server:

      * init_server(ip, port, node_feats) initialize GraphStore server.

      * push_msg_handle(msg) user-defined handler for push message.

      * pull_msg_handle(msg) user-defined handler for pull message.

    GraphStore also provides three APIs for GraphStore client:

      * init_client(sever_ip, server_port) initialize GraphStore client.

      * pull(feat_name, node_ids) requests the feature values associated with corresponding 
        feature name and node IDs.

      * push(feat_name, node_ids, node_feats) sends feature name, node IDs, and node features 
        to the GraphStore server, which uses these data to updating GraphStore.

    Parameters
    ----------
    is_server : bool
        True for GraphStore server and False for GraphStore client
    """
    def __init__(self, is_server):
        self._is_server = is_server

    def __del__(self):
        """Finalize GraphStore
        """
        _finalize_sender(self._sender)
        SocketSync()
        _finalize_receiver(self._receiver)

    def init_server(self, ip, port, tensor_store):
        """Initialize GraphStore server.

        init_server() will start a receiver to wait client's connection. 
        After that, GraphStore will create a sender to connect back to the client.

        Parameters
        ----------
        ip : string
            IP address of GraphStore server
        port : int
            listening port of GraphStore server
        tensor_store: dictionary of tensor
            A dictionary of tensor. The key of the dict is string, representing 
            the feature name. The value of the dict is tensor (torch.tensor() or mxnet.ndarray()), 
            representing the node feature or node embedding.
        """
        if self._is_server == False:
            raise RuntimeError('init_store() can only be used for GraphStore server.')
        self._ip = ip
        self._port = port
        self._tensor_store = tensor_store
        self._receiver = _create_receiver(self._ip, self._port, 1)
        wk_ip, wk_port = _recv_ip_port(self._receiver)
        SocketSync()
        self._sender = _create_sender(wk_ip, wk_port)
        # Solve message in a loop (use [Ctl + C] to exit)
        while True:
            msg = _recv_graph_store_msg(self._receiver)
            if msg.type == MessageType.PUSH:
                self.push_msg_handle(msg.feat_name, msg.node_ids, msg.node_feats)
            elif msg.type == MessageType.PULL:
                node_feats = self.pull_msg_handle(msg.feat_name, msg.node_ids)
                _send_node_feats(self._sender, node_feats)
            else:
                raise RuntimeError('Unknow message type.')

    def push_msg_handle(self, feat_name, node_ids, node_feats):
        """User-defined handler for push message

        Parameters
        ----------
        feat_name : string
            feature name
        node_ids : tensor
            A tensor of node IDs
        node_feats : tensor
            tensor of node feature
        """
        pass

    def pull_msg_handle(self, feat_name, node_ids):
        """User-defined handler for pull message

        Parameters
        ----------
        feat_name : string
            feature name
        node_ids : tensor
            A tensor of node IDs

        Return
        ------
        tensor
            node features
        """
        pass

    def init_client(self, server_ip, server_port):
        """Initialize client and connect to GraphStore server.

        Parameters
        ----------
        server_ip : string
            IP address for remote GraphStore server
        server_port : int
            listening port for remote GraphStore server
        """
        if self._is_server == True:
            raise RuntimeError('init_client() can only be used for client node.')
        self._server_ip = server_ip
        self._server_port = server_port
        self._sender = _create_sender(self._server_ip, self._server_port)
        # Auto find useable ip and port
        self._local_ip, self._local_port = _send_local_ip_port(self._sender)
        self._receiver = _create_receiver(self._local_ip, self._local_port, 1)

    def pull(self, feat_name, node_ids):
        """Request the feature values associated with corresponding 
        feature name and node IDs.

        Parameters
        ----------
        feat_name : string
            feature name
        node_ids : tensor
            node IDs
        """
        if self._is_server == True:
            raise RuntimeError('pull() can only be used for client node.')
        node_feats = _pull(self._sender, feat_name, node_ids)
        return node_feats

    def push(self, feat_name, node_ids, node_feats):
        """Send feature name, node IDs, and node features to the 
        GraphStore server, which uses these data to updating GraphStore.

        Parameters
        ----------
        feat_name : string
            feature name
        node_ids : tensor
            node IDs
        node_feats : tensor
            node features
        """
        if self._is_server == True:
            raise RuntimeError('push() can only be used for client node.')
        _push(self._sender, feat_name, node_ids, node_feats)
