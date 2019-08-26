# This file contains DGL distributed samplers APIs.
from ...network import _send_nodeflow, _recv_nodeflow
from ...network import _create_sender, _create_receiver
from ...network import _finalize_sender, _finalize_receiver
from ...network import _add_receiver_addr, _sender_connect
from ...network import _receiver_wait, _send_sampler_end_signal

from multiprocessing import Pool
from abc import ABCMeta, abstractmethod

class SamplerPool(object):
    """SamplerPool is an abstract class, in which the worker() method 
    should be implemented by users. SamplerPool will fork() N (N = num_worker)
    child processes, and each process will perform worker() method independently.
    Note that, the fork() API uses shared memory for N processes and the OS will
    perfrom copy-on-write on that only when developers write that piece of memory. 
    So fork N processes and load N copies of graph will not increase the memory overhead.

    For example, users can use this class like this:

      class MySamplerPool(SamplerPool):

          def worker(self):
              # Do anything here #

      if __name__ == '__main__':
          ...
          args = parser.parse_args()
          pool = MySamplerPool()
          pool.start(args.num_sender, args)
    """
    __metaclass__ = ABCMeta

    def start(self, num_worker, args):
        """Start sampler pool

        Parameters
        ----------
        num_worker : int
            number of child process
        args : arguments
            any arguments passed by user
        """
        p = Pool()
        for i in range(num_worker):
            print("Start child sampler process %d ..." % i)
            p.apply_async(self.worker, args=(args,))
        # Waiting for all subprocesses done ...
        p.close()
        p.join()

    @abstractmethod
    def worker(self, args):
        """User-defined function for worker

        Parameters
        ----------
        args : arguments
            any arguments passed by user 
        """
        pass

class SamplerSender(object):
    """SamplerSender for DGL distributed training.

    Users use SamplerSender to send sampled subgraphs (NodeFlow) 
    to remote SamplerReceiver. Note that, a SamplerSender can connect 
    to multiple SamplerReceiver currently. The underlying implementation 
    will send different subgraphs to different SamplerReceiver in parallel 
    via multi-threading.

    Parameters
    ----------
    namebook : dict
        IP address namebook of SamplerReceiver, where the
        key is recevier's ID (start from 0) and value is receiver's address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }

    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, namebook, net_type='socket'):
        assert len(namebook) > 0, 'namebook cannot be empty.'
        assert net_type in ('socket', 'mpi'), 'Unknown network type.'
        self._namebook = namebook
        self._sender = _create_sender(net_type)
        for ID, addr in self._namebook.items():
            ip_port = addr.split(':')
            assert len(ip_port) == 2, 'Uncorrect format of IP address.'
            _add_receiver_addr(self._sender, ip_port[0], int(ip_port[1]), ID)
        _sender_connect(self._sender)

    def __del__(self):
        """Finalize Sender
        """
        _finalize_sender(self._sender)

    def send(self, nodeflow, recv_id):
        """Send sampled subgraph (NodeFlow) to remote trainer. Note that, 
        the send() API is non-blocking and it returns immediately if the 
        underlying message queue is not full.

        Parameters
        ----------
        nodeflow : NodeFlow
            sampled NodeFlow
        recv_id : int
            receiver's ID
        """
        assert recv_id >= 0, 'recv_id cannot be a negative number.'
        _send_nodeflow(self._sender, nodeflow, recv_id)

    def batch_send(self, nf_list, id_list):
        """Send a batch of subgraphs (Nodeflow) to remote trainer. Note that, 
        the batch_send() API is non-blocking and it returns immediately if the 
        underlying message queue is not full.

        Parameters
        ----------
        nf_list : list
            a list of NodeFlow object
        id_list : list
            a list of recv_id
        """
        assert len(nf_list) > 0, 'nf_list cannot be empty.'
        assert len(nf_list) == len(id_list), 'The length of nf_list must be equal to id_list.'
        for i in range(len(nf_list)):
            assert id_list[i] >= 0, 'recv_id cannot be a negative number.'
            _send_nodeflow(self._sender, nf_list[i], id_list[i])

    def signal(self, recv_id):
        """When the samplling of each epoch is finished, users can 
        invoke this API to tell SamplerReceiver that sampler has finished its job.

        Parameters
        ----------
        recv_id : int
            receiver's ID
        """
        assert recv_id >= 0, 'recv_id cannot be a negative number.'
        _send_sampler_end_signal(self._sender, recv_id)

class SamplerReceiver(object):
    """SamplerReceiver for DGL distributed training.

    Users use SamplerReceiver to receive sampled subgraphs (NodeFlow) 
    from remote SamplerSender. Note that SamplerReceiver can receive messages 
    from multiple SamplerSenders concurrently by given the num_sender parameter. 
    Only when all SamplerSenders connected to SamplerReceiver successfully, 
    SamplerReceiver can start its job.

    Parameters
    ----------
    graph : DGLGraph
        The parent graph
    addr : str
        address of SamplerReceiver, e.g., '127.0.0.1:50051'
    num_sender : int
        total number of SamplerSender
    net_type : str
        networking type, e.g., 'socket' (default) or 'mpi'.
    """
    def __init__(self, graph, addr, num_sender, net_type='socket'):
        assert num_sender > 0, 'num_sender must be large than zero.'
        assert net_type in ('socket', 'mpi'), 'Unknown network type.'
        self._graph = graph
        self._addr = addr
        self._num_sender = num_sender
        self._tmp_count = 0
        self._receiver = _create_receiver(net_type)
        ip_port = addr.split(':')
        assert len(ip_port) == 2, 'Uncorrect format of IP address.'
        _receiver_wait(self._receiver, ip_port[0], int(ip_port[1]), num_sender);

    def __del__(self):
        """Finalize Receiver
        """
        _finalize_receiver(self._receiver)

    def __iter__(self):
        """Sampler iterator
        """
        return self

    def __next__(self):
        """Return sampled NodeFlow object
        """
        while True:
            res = _recv_nodeflow(self._receiver, self._graph)
            if isinstance(res, int):  # recv an end-signal
                self._tmp_count += 1
                if self._tmp_count == self._num_sender:
                    self._tmp_count = 0
                    raise StopIteration
            else:
                return res  # recv a nodeflow
