# This file contains distributed samplers.
from ...node_flow import NodeFlow
from ...network import _send_subgraph, _recv_subgraph
from ...network import _batch_send_subgraph, _batch_recv_subgraph
from ...network import _create_sampler_sender, _create_sampler_receiver
from ...network import _finalize_sampler_sender, _finalize_sampler_receiver

class SamplerSender(object):
    """The SamplerSender class for distributed sampler.

    Users use the this class to send sampled subgraph to remote trainer.

    Parameters
    ----------
    ip : str
        ip address of remote trainer machine.
    port : int
        listen port of remote trainer machine.
    """
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sender = _create_sampler_sender(ip, port)

    def __del__(self):
        """Finalize Sender
        """
        _finalize_sampler_sender(self._sender)

    def Send(self, nodeflow):
        """Send sampled NodeFlow to remote trainer.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow object.
        """
        _send_subgraph(self._sender, nodeflow)

    def BatchSend(self, nodeflow_list):
        """Send a batch of sampled NodeFlow to remote trainer.

        Parameters
        ----------
        nodeflow_list : list
            a list of NodeFlow object.
        """
        _batch_send_subgraph(self._sender, nodeflow_list)

class SamplerReceiver(object):
    """The SamplerReceiver class for distributed sampler.

    Users use this class to receive sampled subgraph from remote sampler.

    Parameters
    ----------
    ip : str
        ip address of trainer machine.
    port : int
        listen port of trainer machine.
    num_sender : int
        total number of sampler nodes, use 1 by default.
    queue_size : int
        size (bytes) of message queue, use 500 MB by default.
    """
    def __init__(self, ip, port, num_sender=1, queue_size=500*1024*1024):
        self._ip = ip
        self._port = port
        self._num_sender = num_sender
        self._queue_size = queue_size
        self._receiver = _create_sampler_receiver(ip, port, num_sender, queue_size)

    def __del__(self):
        """Finalize Receiver
        """
        _finalize_sampler_receiver(self._receiver)

    def Receive(self):
        """Receive data from sampler node and construct sampled subgraph.
        """
        # Receive a NodeFlowIndex object
        sgi = _recv_subgraph(self._receiver)
        # Note that the parent node will be set 
        # to None in distributed sampler
        return NodeFlow(None, sgi)

    def BatchReceive(self):
        """ Receive data from sender and construct a batch of sampled subgraph.
        """
        # Receive a list of NodeFlowIndex object
        sgi_list = _batch_recv_subgraph(self._receiver)
        nodeflow_list = []
        for sgi in sgi_list:
            # Note that the parent node will be set 
            # to None in distributed sampler
            nodeflow_list.append(NodeFlow(None, sgi))

        return nodeflow_list
