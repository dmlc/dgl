# This file contains subgraph samplers.

import sys
import numpy as np
import threading
import random
import traceback

from ... import utils
from ...node_flow import NodeFlow
from ... import backend as F
try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['NeighborSampler']

class NSSubgraphLoader(object):
    def __init__(self, g, batch_size, expand_factor, num_hops=1,
                 neighbor_type='in', node_prob=None, seed_nodes=None,
                 shuffle=False, num_workers=1, return_seed_id=False):
        self._g = g
        if not g._graph.is_readonly():
            raise NotImplementedError("subgraph loader only support read-only graphs.")
        self._batch_size = batch_size
        self._expand_factor = expand_factor
        self._num_hops = num_hops
        self._node_prob = node_prob
        self._return_seed_id = return_seed_id
        if self._node_prob is not None:
            assert self._node_prob.shape[0] == g.number_of_nodes(), \
                    "We need to know the sampling probability of every node"
        if seed_nodes is None:
            self._seed_nodes = F.arange(0, g.number_of_nodes())
        else:
            self._seed_nodes = seed_nodes
        if shuffle:
            self._seed_nodes = F.rand_shuffle(self._seed_nodes)
        self._num_workers = num_workers
        self._neighbor_type = neighbor_type
        self._subgraphs = []
        self._seed_ids = []
        self._subgraph_idx = 0

    def _prefetch(self):
        seed_ids = []
        num_nodes = len(self._seed_nodes)
        for i in range(self._num_workers):
            start = self._subgraph_idx * self._batch_size
            # if we have visited all nodes, don't do anything.
            if start >= num_nodes:
                break
            end = min((self._subgraph_idx + 1) * self._batch_size, num_nodes)
            seed_ids.append(utils.toindex(self._seed_nodes[start:end]))
            self._subgraph_idx += 1
        sgi = self._g._graph.neighbor_sampling(seed_ids, self._expand_factor,
                                               self._num_hops, self._neighbor_type,
                                               self._node_prob)
        subgraphs = [NodeFlow(self._g, i) for i in sgi]
        self._subgraphs.extend(subgraphs)
        if self._return_seed_id:
            self._seed_ids.extend(seed_ids)

    def __iter__(self):
        return self

    def __next__(self):
        # If we don't have prefetched subgraphs, let's prefetch them.
        if len(self._subgraphs) == 0:
            self._prefetch()
        # At this point, if we still don't have subgraphs, we must have
        # iterate all subgraphs and we should stop the iterator now.
        if len(self._subgraphs) == 0:
            raise StopIteration
        aux_infos = {}
        if self._return_seed_id:
            aux_infos['seeds'] = self._seed_ids.pop(0).tousertensor()
        return self._subgraphs.pop(0), aux_infos

class _Prefetcher(object):
    """Internal shared prefetcher logic. It can be sub-classed by a Thread-based implementation
    or Process-based implementation."""
    _dataq = None  # Data queue transmits prefetched elements
    _controlq = None  # Control queue to instruct thread / process shutdown
    _errorq = None  # Error queue to transmit exceptions from worker to master

    _checked_start = False  # True once startup has been checkd by _check_start

    def __init__(self, loader, num_prefetch):
        super(_Prefetcher, self).__init__()
        self.loader = loader
        assert num_prefetch > 0, 'Unbounded Prefetcher is unsupported.'
        self.num_prefetch = num_prefetch

    def run(self):
        """Method representing the process’s activity."""
        # Startup - Master waits for this
        try:
            loader_iter = iter(self.loader)
            self._errorq.put(None)
        except Exception as e:  # pylint: disable=broad-except
            tb = traceback.format_exc()
            self._errorq.put((e, tb))

        while True:
            try:  # Check control queue
                c = self._controlq.get(False)
                if c is None:
                    break
                else:
                    raise RuntimeError('Got unexpected control code {}'.format(repr(c)))
            except queue.Empty:
                pass
            except RuntimeError as e:
                tb = traceback.format_exc()
                self._errorq.put((e, tb))
                self._dataq.put(None)

            try:
                data = next(loader_iter)
                error = None
            except Exception as e:  # pylint: disable=broad-except
                tb = traceback.format_exc()
                error = (e, tb)
                data = None
            finally:
                self._errorq.put(error)
                self._dataq.put(data)

    def __next__(self):
        next_item = self._dataq.get()
        next_error = self._errorq.get()

        if next_error is None:
            return next_item
        else:
            self._controlq.put(None)
            if isinstance(next_error[0], StopIteration):
                raise StopIteration
            else:
                return self._reraise(*next_error)

    def _reraise(self, e, tb):
        print('Reraising exception from Prefetcher', file=sys.stderr)
        print(tb, file=sys.stderr)
        raise e

    def _check_start(self):
        assert not self._checked_start
        self._checked_start = True
        next_error = self._errorq.get(block=True)
        if next_error is not None:
            self._reraise(*next_error)

    def next(self):
        return self.__next__()


class _ThreadPrefetcher(_Prefetcher, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(_ThreadPrefetcher, self).__init__(*args, **kwargs)
        self._dataq = queue.Queue(self.num_prefetch)
        self._controlq = queue.Queue()
        self._errorq = queue.Queue(self.num_prefetch)
        self.daemon = True
        self.start()
        self._check_start()

class _PrefetchingLoader(object):
    """Prefetcher for a Loader in a separate Thread or Process.
    This iterator will create another thread or process to perform
    ``iter_next`` and then store the data in memory. It potentially accelerates
    the data read, at the cost of more memory usage.

    Parameters
    ----------
    loader : an iterator
        Source loader.
    num_prefetch : int, default 1
        Number of elements to prefetch from the loader. Must be greater 0.
    """

    def __init__(self, loader, num_prefetch=1):
        self._loader = loader
        self._num_prefetch = num_prefetch
        if num_prefetch < 1:
            raise ValueError('num_prefetch must be greater 0.')

    def __iter__(self):
        return _ThreadPrefetcher(self._loader, self._num_prefetch)

def NeighborSampler(g, batch_size, expand_factor, num_hops=1,
                    neighbor_type='in', node_prob=None, seed_nodes=None,
                    shuffle=False, num_workers=1,
                    return_seed_id=False, prefetch=False):
    '''Create a sampler that samples neighborhood.

    This creates a subgraph data loader that samples subgraphs from the input graph
    with neighbor sampling. This sampling method is implemented in C and can perform
    sampling very efficiently.
    
    A subgraph grows from a seed vertex. It contains sampled neighbors
    of the seed vertex as well as the edges that connect neighbor nodes with
    seed nodes. When the number of hops is k (>1), the neighbors are sampled
    from the k-hop neighborhood. In this case, the sampled edges are the ones
    that connect the source nodes and the sampled neighbor nodes of the source
    nodes.

    The subgraph loader returns a list of subgraphs and a dictionary of additional
    information about the subgraphs. The size of the subgraph list is the number of workers.

    The dictionary contains:

    - seeds: a list of 1D tensors of seed Ids, if return_seed_id is True.

    Parameters
    ----------
    g: the DGLGraph where we sample subgraphs.
    batch_size: The number of subgraphs in a batch.
    expand_factor: the number of neighbors sampled from the neighbor list
        of a vertex. The value of this parameter can be
        an integer: indicates the number of neighbors sampled from a neighbor list.
        a floating-point: indicates the ratio of the sampled neighbors in a neighbor list.
        string: indicates some common ways of calculating the number of sampled neighbors,
        e.g., 'sqrt(deg)'.
    num_hops: The size of the neighborhood where we sample vertices.
    neighbor_type: indicates the neighbors on different types of edges.
        "in" means the neighbors on the in-edges, "out" means the neighbors on
        the out-edges and "both" means neighbors on both types of edges.
    node_prob: the probability that a neighbor node is sampled.
        1D Tensor. None means uniform sampling. Otherwise, the number of elements
        should be the same as the number of vertices in the graph.
    seed_nodes: a list of nodes where we sample subgraphs from.
        If it's None, the seed vertices are all vertices in the graph.
    shuffle: indicates the sampled subgraphs are shuffled.
    num_workers: the number of worker threads that sample subgraphs in parallel.
    return_seed_id: indicates whether to return seed ids along with the subgraphs.
        The seed Ids are in the parent graph.
    prefetch : bool, default False
        Whether to prefetch the samples in the next batch.

    Returns
    -------
    A subgraph iterator
        The iterator returns a list of batched subgraphs and a dictionary of additional
        information about the subgraphs.
    '''
    loader = NSSubgraphLoader(g, batch_size, expand_factor, num_hops, neighbor_type, node_prob,
                              seed_nodes, shuffle, num_workers, return_seed_id)
    if not prefetch:
        return loader
    else:
        return _PrefetchingLoader(loader, num_prefetch=num_workers*2)
