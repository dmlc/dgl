# This file contains distributed samplers.
from ...node_flow import NodeFlow
from ...network import _create_sender, _create_receiver
from ...network import _send_subgraph, _recv_subgraph
from ...network import _finalize_sender, _finalize_receiver

class Sender(object):
    """The Sender class for distributed sampler.

    Users use the Sender class to send sampled subgraph 
    to remote trainer machine.

    Parameters
    ----------
    ip : ip address of remote machine
    port : listen port of remote machine
    """
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sender = _create_sender(ip, port)

    def Send(self, nodeflow):
        """Send nodeflow to remote receiver

        Parameter
        ---------
        nodeflow : a NodeFlow object
        """
        _send_subgraph(self._sender, nodeflow)

    def BatchSend(self, nodeflow_list):
        """Send a batch of nodeflow to remote receiver

        Parameter
        ---------
        nodeflow_list : a list of nodeflow object
        """
        _batch_send_subgraph(self._sender, nodeflow_list)

    def Finalize(self):
        """Finalize Sender
        """
        _finalize_sender(self._sender)

class Receiver(object):
    """The Receiver class for distributed sampler.

    Users use the Receiver class to receive sampled subgraph 
    from the remote sampler machine.

    Parameters
    ----------
    ip : ip address of remote machine
    port : port of remote machine
    num_sender : total number of sender nodes
    queue_size : size (bytes) of message queue 
    """
    def __init__(self, ip, port, num_sender, queue_size):
        self._ip = ip
        self._port = port
        self._num_sender = num_sender
        self._queue_size = queue_size
        self._receiver = _create_receiver(ip, port, num_sender, queue_size)

    def Receive(self):
        """ Receive data from sender and construct sampled subgraph.
        Note that, in distributed sampler, the parent graph of NodeFlow 
        in will be set to None object.
        """
        sgi = _recv_subgraph(self._receiver)
        return NodeFlow(None, sgi)

    def BatchReceive(self):
        """ Receive data from sender and construct a batch of sampled subgraph.
        Note that, in distributed sampler, the parent graph of NodeFlow 
        in will be set to None object.
        """
        sgi_list = _batch_recv_subgraph(self._receiver)
        nodeflow_list = []
        for sgi in sgi_list:
            nodeflow_list.append(NodeFlow(None, sgi))
        return nodeflow_list

    def Finalize(self):
        """Finalize Receiver
        """
        _finalize_receiver(self._receiver)