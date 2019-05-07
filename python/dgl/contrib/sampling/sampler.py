"""This file contains NodeFlow samplers."""

import sys
import numpy as np
import threading
from numbers import Integral
import traceback

from ..._ffi.function import _init_api
from ... import utils
from ...nodeflow import NodeFlow
from ... import backend as F
from ...utils import unwrap_to_ptr_list

try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['NeighborSampler', 'LayerSampler']

class NodeFlowSamplerIter(object):
    def __init__(self, sampler):
        super(NodeFlowSamplerIter, self).__init__()
        self._sampler = sampler
        self._nflows = []
        self._nflow_idx = 0

    def prefetch(self):
        nflows = self._sampler.fetch(self._nflow_idx)
        self._nflows.extend(nflows)
        self._nflow_idx += len(nflows)

    def __next__(self):
        if len(self._nflows) == 0:
            self.prefetch()
        if len(self._nflows) == 0:
            raise StopIteration
        return self._nflows.pop(0)

class PrefetchingWrapper(object):
    """Internal shared prefetcher logic. It can be sub-classed by a Thread-based implementation
    or Process-based implementation."""
    _dataq = None  # Data queue transmits prefetched elements
    _controlq = None  # Control queue to instruct thread / process shutdown
    _errorq = None  # Error queue to transmit exceptions from worker to master

    _checked_start = False  # True once startup has been checkd by _check_start

    def __init__(self, sampler_iter, num_prefetch):
        super(PrefetchingWrapper, self).__init__()
        self.sampler_iter = sampler_iter
        assert num_prefetch > 0, 'Unbounded Prefetcher is unsupported.'
        self.num_prefetch = num_prefetch

    def run(self):
        """Method representing the process activity."""
        # Startup - Master waits for this
        try:
            loader_iter = self.sampler_iter
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

class ThreadPrefetchingWrapper(PrefetchingWrapper, threading.Thread):
    """Internal threaded prefetcher."""

    def __init__(self, *args, **kwargs):
        super(ThreadPrefetchingWrapper, self).__init__(*args, **kwargs)
        self._dataq = queue.Queue(self.num_prefetch)
        self._controlq = queue.Queue()
        self._errorq = queue.Queue(self.num_prefetch)
        self.daemon = True
        self.start()
        self._check_start()


class NodeFlowSampler(object):
    '''
    Base class that generates NodeFlows from a graph.

    Class properties
    ----------------
    immutable_only : bool
        Whether the sampler only works on immutable graphs.
        Subclasses can override this property.
    '''
    immutable_only = False

    def __init__(
            self,
            g,
            batch_size,
            seed_nodes,
            shuffle,
            num_prefetch,
            prefetching_wrapper_class):
        self._g = g
        if self.immutable_only and not g._graph.is_readonly():
            raise NotImplementedError("This loader only support read-only graphs.")

        self._batch_size = int(batch_size)

        if seed_nodes is None:
            self._seed_nodes = F.arange(0, g.number_of_nodes())
        else:
            self._seed_nodes = seed_nodes
        if shuffle:
            self._seed_nodes = F.rand_shuffle(self._seed_nodes)
        self._seed_nodes = utils.toindex(self._seed_nodes)

        if num_prefetch:
            self._prefetching_wrapper_class = prefetching_wrapper_class
        self._num_prefetch = num_prefetch

    def fetch(self, current_nodeflow_index):
        '''
        Method that returns the next "bunch" of NodeFlows.
        Each worker will return a single NodeFlow constructed from a single
        batch.

        Subclasses of NodeFlowSampler should override this method.

        Parameters
        ----------
        current_nodeflow_index : int
            How many NodeFlows the sampler has generated so far.

        Returns
        -------
        list[NodeFlow]
            Next "bunch" of nodeflows to be processed.
        '''
        raise NotImplementedError

    def __iter__(self):
        it = NodeFlowSamplerIter(self)
        if self._num_prefetch:
            return self._prefetching_wrapper_class(it, self._num_prefetch)
        else:
            return it

    @property
    def g(self):
        return self._g

    @property
    def seed_nodes(self):
        return self._seed_nodes

    @property
    def batch_size(self):
        return self._batch_size

class NeighborSampler(NodeFlowSampler):
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

        Note that no matter how large the expand_factor, the max number of sampled neighbors
        is the neighborhood size.
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
    '''

    immutable_only = True

    def __init__(
            self,
            g,
            batch_size,
            expand_factor=None,
            num_hops=1,
            neighbor_type='in',
            node_prob=None,
            seed_nodes=None,
            shuffle=False,
            num_workers=1,
            prefetch=False,
            add_self_loop=False):
        super(NeighborSampler, self).__init__(
                g, batch_size, seed_nodes, shuffle, num_workers * 2 if prefetch else 0,
                ThreadPrefetchingWrapper)

        assert node_prob is None, 'non-uniform node probability not supported'
        assert isinstance(expand_factor, Integral), 'non-int expand_factor not supported'

        self._expand_factor = expand_factor
        self._num_hops = num_hops
        self._add_self_loop = add_self_loop
        self._num_workers = num_workers
        self._neighbor_type = neighbor_type

    def fetch(self, current_nodeflow_index):
        handles = unwrap_to_ptr_list(_CAPI_UniformSampling(
            self.g.c_handle,
            self.seed_nodes.todgltensor(),
            current_nodeflow_index, # start batch id
            self.batch_size,        # batch size
            self._num_workers,      # num batches
            self._expand_factor,
            self._num_hops,
            self._neighbor_type,
            self._add_self_loop))
        nflows = [NodeFlow(self.g, hdl) for hdl in handles]
        return nflows


class LayerSampler(NodeFlowSampler):
    '''Create a sampler that samples neighborhood.

    This creates a NodeFlow loader that samples subgraphs from the input graph
    with layer-wise sampling. This sampling method is implemented in C and can perform
    sampling very efficiently.

    The NodeFlow loader returns a list of NodeFlows.
    The size of the NodeFlow list is the number of workers.

    Parameters
    ----------
    g: the DGLGraph where we sample NodeFlows.
    batch_size: The number of NodeFlows in a batch.
    layer_size: A list of layer sizes.
    node_prob: the probability that a neighbor node is sampled.
        Not implemented.
    seed_nodes: a list of nodes where we sample NodeFlows from.
        If it's None, the seed vertices are all vertices in the graph.
    shuffle: indicates the sampled NodeFlows are shuffled.
    num_workers: the number of worker threads that sample NodeFlows in parallel.
    prefetch : bool, default False
        Whether to prefetch the samples in the next batch.
    '''

    immutable_only = True

    def __init__(
            self,
            g,
            batch_size,
            layer_sizes,
            neighbor_type='in',
            node_prob=None,
            seed_nodes=None,
            shuffle=False,
            num_workers=1,
            prefetch=False):
        super(LayerSampler, self).__init__(
                g, batch_size, seed_nodes, shuffle, num_workers * 2 if prefetch else 0,
                ThreadPrefetchingWrapper)

        assert node_prob is None, 'non-uniform node probability not supported'

        self._num_workers = num_workers
        self._neighbor_type = neighbor_type
        self._layer_sizes = utils.toindex(layer_sizes)

    def fetch(self, current_nodeflow_index):
        handles = unwrap_to_ptr_list(_CAPI_LayerSampling(
            self.g.c_handle,
            self.seed_nodes.todgltensor(),
            current_nodeflow_index,  # start batch id
            self.batch_size,         # batch size
            self._num_workers,       # num batches
            self._layer_sizes.todgltensor(),
            self._neighbor_type))
        nflows = [NodeFlow(self.g, hdl) for hdl in handles]
        return nflows

def create_full_nodeflow(g, num_layers, add_self_loop=False):
    """Convert a full graph to NodeFlow to run a L-layer GNN model.

    Parameters
    ----------
    g : DGLGraph
        a DGL graph
    num_layers : int
        The number of layers
    add_self_loop : bool, default False
        Whether to add self loop to the sampled NodeFlow.
        If True, the edge IDs of the self loop edges are -1.

    Returns
    -------
    NodeFlow
        a NodeFlow with a specified number of layers.
    """
    batch_size = g.number_of_nodes()
    expand_factor = g.number_of_nodes()
    sampler = NeighborSampler(g, batch_size, expand_factor,
        num_layers, add_self_loop=add_self_loop)
    return next(iter(sampler))

_init_api('dgl.sampling', __name__)
