# This file contains NodeFlow samplers.

import sys
import numpy as np
import threading
import random
import traceback

from ..._ffi.function import _init_api
from ... import utils
from ...node_flow import NodeFlow
from ...graph_index import NodeFlowIndex
from ... import backend as F

try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['NeighborSampler']

class NSSubgraphLoader(object):
    def __init__(self, g, batch_size, expand_factor, num_hops=1,
                 neighbor_type='in', node_prob=None, seed_nodes=None,
                 shuffle=False, num_workers=1, add_self_loop=False):
        self._g = g
        if not g._graph.is_readonly():
            raise NotImplementedError("NodeFlow loader only support read-only graphs.")
        self._batch_size = batch_size
        self._expand_factor = expand_factor
        self._num_hops = num_hops
        self._node_prob = node_prob
        self._add_self_loop = add_self_loop
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
        self._nflows = []
        self._seed_ids = []
        self._nflow_idx = 0

    def _prefetch(self):
        seed_ids = []
        num_nodes = len(self._seed_nodes)
        for i in range(self._num_workers):
            start = self._nflow_idx * self._batch_size
            # if we have visited all nodes, don't do anything.
            if start >= num_nodes:
                break
            end = min((self._nflow_idx + 1) * self._batch_size, num_nodes)
            seed_ids.append(utils.toindex(self._seed_nodes[start:end]))
            self._nflow_idx += 1
        sgi = _neighbor_sampling(self._g._graph, seed_ids, self._expand_factor,
                                 self._num_hops, self._neighbor_type,
                                 self._node_prob, self._add_self_loop)
        nflows = [NodeFlow(self._g, i) for i in sgi]
        self._nflows.extend(nflows)

    def __iter__(self):
        return self

    def __next__(self):
        # If we don't have prefetched NodeFlows, let's prefetch them.
        if len(self._nflows) == 0:
            self._prefetch()
        # At this point, if we still don't have NodeFlows, we must have
        # iterate all NodeFlows and we should stop the iterator now.
        if len(self._nflows) == 0:
            raise StopIteration
        return self._nflows.pop(0)

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
                    shuffle=False, num_workers=1, prefetch=False, add_self_loop=False):
    '''Create a sampler that samples neighborhood.

    It returns a generator of :class:`~dgl.NodeFlow`. This can be viewed as
    an analogy of *mini-batch training* on graph data -- the given graph represents
    the whole dataset and the returned generator produces mini-batches (in the form
    of :class:`~dgl.NodeFlow` objects).
    
    A NodeFlow grows from sampled nodes. It first samples a set of nodes from the given
    ``seed_nodes`` (or all the nodes if not given), then samples their neighbors
    and extracts the subgraph. If the number of hops is :math:`k(>1)`, the process is repeated
    recursively, with the neighbor nodes just sampled become the new seed nodes.
    The result is a graph we defined as :class:`~dgl.NodeFlow` that contains :math:`k+1`
    layers. The last layer is the initial seed nodes. The sampled neighbor nodes in
    layer :math:`i+1` are in layer :math:`i`. All the edges are from nodes
    in layer :math:`i` to layer :math:`i+1`.

    TODO(minjie): give a figure here.
    
    As an analogy to mini-batch training, the ``batch_size`` here is equal to the number
    of the initial seed nodes (number of nodes in the last layer).
    The number of nodeflow objects (the number of batches) is calculated by
    ``len(seed_nodes) // batch_size`` (if ``seed_nodes`` is None, then it is equal
    to the set of all nodes in the graph).

    Parameters
    ----------
    g : DGLGraph
        The DGLGraph where we sample NodeFlows.
    batch_size : int
        The batch size (i.e, the number of nodes in the last layer)
    expand_factor : int, float, str
        The number of neighbors sampled from the neighbor list of a vertex.
        The value of this parameter can be:

        * int: indicates the number of neighbors sampled from a neighbor list.
        * float: indicates the ratio of the sampled neighbors in a neighbor list.
        * str: indicates some common ways of calculating the number of sampled neighbors,
          e.g., ``sqrt(deg)``.

    num_hops : int, optional
        The number of hops to sample (i.e, the number of layers in the NodeFlow).
        Default: 1
    neighbor_type: str, optional
        Indicates the neighbors on different types of edges.

        * "in": the neighbors on the in-edges.
        * "out": the neighbors on the out-edges.
        * "both": the neighbors on both types of edges.

        Default: "in"
    node_prob : Tensor, optional
        A 1D tensor for the probability that a neighbor node is sampled.
        None means uniform sampling. Otherwise, the number of elements
        should be equal to the number of vertices in the graph.
        Default: None
    seed_nodes : Tensor, optional
        A 1D tensor  list of nodes where we sample NodeFlows from.
        If None, the seed vertices are all the vertices in the graph.
        Default: None
    shuffle : bool, optional
        Indicates the sampled NodeFlows are shuffled. Default: False
    num_workers : int, optional
        The number of worker threads that sample NodeFlows in parallel. Default: 1
    prefetch : bool, optional
        If true, prefetch the samples in the next batch. Default: False
    add_self_loop : bool, optional
        If true, add self loop to the sampled NodeFlow.
        The edge IDs of the self loop edges are -1. Default: False

    Returns
    -------
    generator
        The generator of NodeFlows.
    '''
    loader = NSSubgraphLoader(g, batch_size, expand_factor, num_hops, neighbor_type, node_prob,
                              seed_nodes, shuffle, num_workers, add_self_loop)
    if not prefetch:
        return loader
    else:
        return _PrefetchingLoader(loader, num_prefetch=num_workers*2)

def _neighbor_sampling(gidx, seed_ids, expand_factor, num_hops, neighbor_type,
                      node_prob, add_self_loop=False):
    """Neighborhood sampling"""
    if len(seed_ids) == 0:
        return []

    seed_ids = [v.todgltensor() for v in seed_ids]
    num_subgs = len(seed_ids)
    if node_prob is None:
        rst = _uniform_sampling(gidx, seed_ids, neighbor_type, num_hops,
                                expand_factor, add_self_loop)
    else:
        rst = _nonuniform_sampling(gidx, node_prob, seed_ids, neighbor_type, num_hops,
                                   expand_factor)

    return [NodeFlowIndex(rst(i), gidx, utils.toindex(rst(num_subgs + i)),
                          utils.toindex(rst(num_subgs * 2 + i)),
                          utils.toindex(rst(num_subgs * 3 + i)),
                          utils.toindex(rst(num_subgs * 4 + i))) for i in range(num_subgs)]

_init_api('dgl.sampling', __name__)

# TODO(zhengda): we'll support variable-length inputs.
_NEIGHBOR_SAMPLING_APIS = {
    1: _CAPI_DGLGraphUniformSampling,
    2: _CAPI_DGLGraphUniformSampling2,
    4: _CAPI_DGLGraphUniformSampling4,
    8: _CAPI_DGLGraphUniformSampling8,
    16: _CAPI_DGLGraphUniformSampling16,
    32: _CAPI_DGLGraphUniformSampling32,
    64: _CAPI_DGLGraphUniformSampling64,
    128: _CAPI_DGLGraphUniformSampling128,
}

_EMPTY_ARRAYS = [utils.toindex(F.ones(shape=(0), dtype=F.int64, ctx=F.cpu()))]

def _uniform_sampling(gidx, seed_ids, neigh_type, num_hops, expand_factor, add_self_loop):
    num_seeds = len(seed_ids)
    empty_ids = []
    if len(seed_ids) > 1 and len(seed_ids) not in _NEIGHBOR_SAMPLING_APIS.keys():
        remain = 2**int(math.ceil(math.log2(len(dgl_ids)))) - len(dgl_ids)
        empty_ids = _EMPTY_ARRAYS[0:remain]
        seed_ids.extend([empty.todgltensor() for empty in empty_ids])
    assert len(seed_ids) in _NEIGHBOR_SAMPLING_APIS.keys()
    return _NEIGHBOR_SAMPLING_APIS[len(seed_ids)](gidx._handle, *seed_ids, neigh_type,
                                                  num_hops, expand_factor, num_seeds,
                                                  add_self_loop)
