# This file contains DGL distributed samplers APIs.
from ...network import _send_subgraph, _recv_subgraph
from ...network import _batch_send_subgraph, _batch_recv_subgraph
from ...network import _create_sampler_sender, _create_sampler_receiver
from ...network import _finalize_sampler_sender, _finalize_sampler_receiver

class SamplerSender(object):
    """Sender of DGL distributed sampler.

    Users use SamplerSender class to send sampled 
    subgraph (NodeFlow) to remote trainer. Note that, SamplerSender
    class will try to connect to SamplerReceiver in a loop until the
    SamplerReceiver started.

    Parameters
    ----------
    ip : str
        ip address of remote trainer machine
    port : int
        port of remote trainer machine
    """
    def __init__(self, ip, port):
        self._ip = ip
        self._port = port
        self._sender = _create_sampler_sender(ip, port)

    def __del__(self):
        """Finalize Sender
        """
        # _finalize_sampler_sender will send a special message
        # to tell the remote trainer machine that it has finished its job.
        _finalize_sampler_sender(self._sender)

    def Send(self, nodeflow):
        """Send sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow object
        """
        _send_subgraph(self._sender, nodeflow)

    def BatchSend(self, nodeflow_list):
        """Send a batch of sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow_list : list
            A list of NodeFlow objects
        """
        _batch_send_subgraph(self._sender, nodeflow_list)

class SamplerReceiver(object):
    """Receiver of DGL distributed sampler.

    Users use SamplerReceiver class to receive sampled 
    subgraph (NodeFlow) from remote samplers. Note that SamplerReceiver 
    can receive messages from multiple senders concurrently, by given 
    the num_sender parameter, and only when all senders connect to SamplerReceiver,
    the SamplerReceiver can start its job.

    Parameters
    ----------
    ip : str
        ip address of current trainer machine
    port : int
        port of current trainer machine
    num_sender : int
        total number of sampler nodes, use 1 by default
    """
    def __init__(self, ip, port, num_sender=1):
        self._ip = ip
        self._port = port
        self._num_sender = num_sender
        self._receiver = _create_sampler_receiver(ip, port, num_sender)

    def __del__(self):
        """Finalize Receiver

        _finalize_sampler_receiver method will clean up the 
        back-end threads started by the SamplerReceiver.
        """
        _finalize_sampler_receiver(self._receiver)

    def Receive(self, graph):
        """Receive a NodeFlow object from remote sampler.

        Parameters
        ----------
        graph : DGLGraph
            The parent graph

        Returns
        -------
        NodeFlow
            received NodeFlow object
        """
        return _recv_subgraph(self._receiver, graph)

    def BatchReceive(self, graph):
        """Receive a batch of NodeFlow objects from remote sampler.

        Parameters
        ----------
        graph : DGLGraph
            The parent graph

        Returns
        -------
        list
            A list of received NodeFlow object
        """
        return _batch_recv_subgraph(self._receiver, graph)
