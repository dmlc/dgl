# This file contains distributed samplers.
from ...node_flow import NodeFlow
from ...graph_index import CreateSender, CreateReceiver, SendSubgraph, RecvSubgraph

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
        self._sender = CreateSender(ip, port)

    def Send(self, nodeflow):
        """ Send nodeflow to remote receiver

        Parameter
        ---------
        nodeflow : a NodeFlow object
        """
        SendSubgraph(self._sender, nodeflow)

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
        self._receiver = CreateReceiver(ip, port, num_sender, queue_size)

    def Receive(self):
        """ Receive data from sender and construct sampled subgraph.
        Note that, in distributed sampler, the parent graph of NodeFlow 
        in will be set to None object.
        """
        sgi = RecvSubgraph(self._receiver)
        return NodeFlow(None, sgi)