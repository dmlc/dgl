# This file contains DGL distributed samplers APIs.
from ...network import _send_subgraph, _recv_subgraph
from ...network import _create_sampler_sender, _create_sampler_receiver
from ...network import _finalize_sampler_sender, _finalize_sampler_receiver

from multiprocessing import Pool
from abc import ABCMeta, abstractmethod

class SamplerPool(object):
    """SamplerPool is an abstract class, in which the worker method 
    should be implemented by users. SamplerPool will fork() N (N = num_worker)
    child processes, and each process will perform worker() method independently.
    Note that, the fork() API will use shared memory for N process and the OS will
    perfrom copy-on-write only when developers write that piece of memory.

    Users can use this class like this:

      class MySamplerPool(SamplerPool):

        def worker(self):
            # Do anything here #

      if __name__ == '__main__':
        pool = MySamplerPool()
        pool.start(5) # Start 5 processes

    Parameters
    ----------
    num_worker : int
        number of worker (child process)
    """
    __metaclass__ = ABCMeta

    def start(self, num_worker):
        p = Pool()
        for i in range(num_worker):
            print("Start child process %d ..." % i)
            p.apply_async(self.worker)
        # Waiting for all subprocesses done ...
        p.close()
        p.join()

    @abstractmethod
    def worker(self):
        pass

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

    def send(self, nodeflow):
        """Send sampled subgraph (NodeFlow) to remote trainer.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow object
        """
        _send_subgraph(self._sender, nodeflow)

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

    def recv(self, graph):
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
