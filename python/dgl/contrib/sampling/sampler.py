"""This file contains NodeFlow samplers."""

import sys
import numpy as np
import threading
from numbers import Integral
import traceback

from ..._ffi.function import _init_api
from ..._ffi.object import register_object, ObjectBase
from ..._ffi.ndarray import empty
from ... import utils
from ..._deprecate.nodeflow import NodeFlow
from ... import backend as F
from ..._deprecate.graph import DGLGraph as DGLGraphStale
from ...base import NID, EID, dgl_warning

try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['NeighborSampler', 'LayerSampler', 'EdgeSampler']

class SamplerIter(object):
    def __init__(self, sampler):
        super(SamplerIter, self).__init__()
        self._sampler = sampler
        self._batches = []
        self._batch_idx = 0

    def prefetch(self):
        batches = self._sampler.fetch(self._batch_idx)
        self._batches.extend(batches)
        self._batch_idx += len(batches)

    def __next__(self):
        if len(self._batches) == 0:
            self.prefetch()
        if len(self._batches) == 0:
            raise StopIteration
        return self._batches.pop(0)

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
    '''Base class that generates NodeFlows from a graph.

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
        it = SamplerIter(self)
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
    r'''Create a sampler that samples neighborhood.

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

    .. image:: https://data.dgl.ai/tutorial/sampling/NodeFlow.png
    
    As an analogy to mini-batch training, the ``batch_size`` here is equal to the number
    of the initial seed nodes (number of nodes in the last layer).
    The number of nodeflow objects (the number of batches) is calculated by
    ``len(seed_nodes) // batch_size`` (if ``seed_nodes`` is None, then it is equal
    to the set of all nodes in the graph).

    Note: NeighborSampler currently only supprts immutable graphs.

    Parameters
    ----------
    g : DGLGraphStale
        The DGLGraphStale where we sample NodeFlows.
    batch_size : int
        The batch size (i.e, the number of nodes in the last layer)
    expand_factor : int
        The number of neighbors sampled from the neighbor list of a vertex.

        Note that no matter how large the expand_factor, the max number of sampled neighbors
        is the neighborhood size.
    num_hops : int, optional
        The number of hops to sample (i.e, the number of layers in the NodeFlow).
        Default: 1
    neighbor_type: str, optional
        Indicates the neighbors on different types of edges.

        * "in": the neighbors on the in-edges.
        * "out": the neighbors on the out-edges.

        Default: "in"
    transition_prob : str, optional
        A 1D tensor containing the (unnormalized) transition probability.

        The probability of a node v being sampled from a neighbor u is proportional to
        the edge weight, normalized by the sum over edge weights grouping by the
        destination node.

        In other words, given a node v, the probability of node u and edge (u, v)
        included in the NodeFlow layer preceding that of v is given by:

        .. math::

           p(u, v) = \frac{w_{u, v}}{\sum_{u', (u', v) \in E} w_{u', v}}

        If neighbor type is "out", then the probability is instead normalized by the sum
        grouping by source node:

        .. math::

           p(v, u) = \frac{w_{v, u}}{\sum_{u', (v, u') \in E} w_{v, u'}}

        If a str is given, the edge weight will be loaded from the edge feature column with
        the same name.  The feature column must be a scalar column in this case.

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
            transition_prob=None,
            seed_nodes=None,
            shuffle=False,
            num_workers=1,
            prefetch=False,
            add_self_loop=False):
        super(NeighborSampler, self).__init__(
                g, batch_size, seed_nodes, shuffle, num_workers * 2 if prefetch else 0,
                ThreadPrefetchingWrapper)
        dgl_warning('dgl.contrib.sampling.NeighborSampler is deprecated starting from v0.5.'
                    ' Please read our guide<link> for how to use the new sampling APIs.')

        assert g.is_readonly, "NeighborSampler doesn't support mutable graphs. " + \
                "Please turn it into an immutable graph with DGLGraphStale.readonly"
        assert isinstance(expand_factor, Integral), 'non-int expand_factor not supported'

        self._expand_factor = int(expand_factor)
        self._num_hops = int(num_hops)
        self._add_self_loop = add_self_loop
        self._num_workers = int(num_workers)
        self._neighbor_type = neighbor_type
        self._transition_prob = transition_prob

    def fetch(self, current_nodeflow_index):
        if self._transition_prob is None:
            prob = F.tensor([], F.float32)
        elif isinstance(self._transition_prob, str):
            prob = self.g.edata[self._transition_prob]
        else:
            prob = self._transition_prob

        nfobjs = _CAPI_NeighborSampling(
            self.g._graph,
            self.seed_nodes.todgltensor(),
            current_nodeflow_index, # start batch id
            self.batch_size,        # batch size
            self._num_workers,      # num batches
            self._expand_factor,
            self._num_hops,
            self._neighbor_type,
            self._add_self_loop,
            F.zerocopy_to_dgl_ndarray(prob))

        nflows = [NodeFlow(self.g, obj) for obj in nfobjs]
        return nflows


class LayerSampler(NodeFlowSampler):
    '''Create a sampler that samples neighborhood.

    This creates a NodeFlow loader that samples subgraphs from the input graph
    with layer-wise sampling. This sampling method is implemented in C and can perform
    sampling very efficiently.

    The NodeFlow loader returns a list of NodeFlows.
    The size of the NodeFlow list is the number of workers.

    Note: LayerSampler currently only supprts immutable graphs.

    Parameters
    ----------
    g : DGLGraphStale
        The DGLGraphStale where we sample NodeFlows.
    batch_size : int
        The batch size (i.e, the number of nodes in the last layer)
    layer_size: int
        A list of layer sizes.
    neighbor_type: str, optional
        Indicates the neighbors on different types of edges.

        * "in": the neighbors on the in-edges.
        * "out": the neighbors on the out-edges.

        Default: "in"
    node_prob : Tensor, optional
        A 1D tensor for the probability that a neighbor node is sampled.
        None means uniform sampling. Otherwise, the number of elements
        should be equal to the number of vertices in the graph.
        It's not implemented.
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

        assert g.is_readonly, "LayerSampler doesn't support mutable graphs. " + \
                "Please turn it into an immutable graph with DGLGraphStale.readonly"
        assert node_prob is None, 'non-uniform node probability not supported'

        self._num_workers = int(num_workers)
        self._neighbor_type = neighbor_type
        self._layer_sizes = utils.toindex(layer_sizes)

    def fetch(self, current_nodeflow_index):
        nfobjs = _CAPI_LayerSampling(
            self.g._graph,
            self.seed_nodes.todgltensor(),
            current_nodeflow_index,  # start batch id
            self.batch_size,         # batch size
            self._num_workers,       # num batches
            self._layer_sizes.todgltensor(),
            self._neighbor_type)
        nflows = [NodeFlow(self.g, obj) for obj in nfobjs]
        return nflows

class EdgeSubgraph(DGLGraphStale):
    ''' The subgraph sampled from an edge sampler.

    A user can access the head nodes and tail nodes of the subgraph directly.
    '''
    def __init__(self, parent, sgi, neg):
        super(EdgeSubgraph, self).__init__(graph_data=sgi.graph,
                                           readonly=True,
                                           parent=parent)
        self.ndata[NID] = sgi.induced_nodes.tousertensor()
        self.edata[EID] = sgi.induced_edges.tousertensor()
        self.sgi = sgi
        self.neg = neg
        self.head = None
        self.tail = None

    def set_head_tail(self):
        if self.head is None or self.tail is None:
            if self.neg:
                exist = _CAPI_GetEdgeSubgraphHead(self.sgi)
                self.head = utils.toindex(exist).tousertensor()

                exist = _CAPI_GetEdgeSubgraphTail(self.sgi)
                self.tail = utils.toindex(exist).tousertensor()
            else:
                head, tail = self.all_edges()
                self.head = F.unique(head)
                self.tail = F.unique(tail)

    @property
    def head_nid(self):
        ''' The unique Ids of the head nodes.
        '''
        self.set_head_tail()
        return self.head

    @property
    def tail_nid(self):
        ''' The unique Ids of the tail nodes.
        '''
        self.set_head_tail()
        return self.tail


class EdgeSampler(object):
    '''Edge sampler for link prediction.

    This samples edges from a given graph. The edges sampled for a batch are
    placed in a subgraph before returning. In many link prediction tasks,
    negative edges are required to train a model. A negative edge is constructed by
    corrupting an existing edge in the graph. The current implementation
    support two ways of corrupting an edge: corrupt the head node of
    an edge (by randomly selecting a node as the head node), or corrupt
    the tail node of an edge. When we corrupt the head node of an edge, we randomly
    sample a node from the entire graph as the head node. It's possible the constructed
    edge exists in the graph. By default, the implementation doesn't explicitly check
    if the sampled negative edge exists in a graph. However, a user can exclude
    positive edges from negative edges by specifying 'exclude_positive=True'.
    When negative edges are created, a batch of negative edges are also placed
    in a subgraph.

    Currently, `negative_mode` only supports:

    * 'head': the negative edges are generated by corrupting head nodes with uniformly randomly sampled nodes,

    * 'tail': the negative edges are generated by corrupting tail nodes with uniformly randomly sampled nodes,

    * 'chunk-head': the negative edges are generated for a chunk of positive edges. \
    It first groups positive edges into chunks and corrupts a chunk of edges together \
    by replacing a set of head nodes with the same set of nodes uniformly randomly sampled \
    from the graph.

    * 'chunk-tail': the negative edges are generated by corrupting a set \
    of tail nodes with the same set of nodes similar to 'chunk-head'.

    When we use chunked negative sampling, a chunk size needs to be specified. By default,
    the chunk size is the same as the number of negative edges.

    The sampler returns EdgeSubgraph, where a user can access the unique head nodes
    and tail nodes directly.

    This sampler allows to non-uniformly sample positive edges and negative edges. 
    For non-uniformly sampling positive edges, users need to provide an array of m 
    elements (m is the number of edges), i.e. edge_weight, each of which represents 
    the sampling probability of an edge. For non-uniformly sampling negative edges, 
    users need to provide an array of n elements, i.e. node_weight and the sampler 
    samples nodes based on the sampling probability to corrupt a positive edge. If 
    both edge_weight and node_weight are not provided, a uniformed sampler is used.
    if only edge_weight is provided, the sampler will take uniform sampling when 
    corrupt positive edges. 

    When the flag `return_false_neg` is turned on, the sampler will also check
    if the generated negative edges are true negative edges and will return
    a vector that indicates false negative edges. The vector is stored in
    the negative graph as `false_neg` edge data.

    When checking false negative edges, a user can provide edge relations
    for a knowledge graph. A negative edge is considered as a false negative
    edge only if the triple (source node, destination node and relation)
    matches one of the edges in the graph.

    This sampler samples positive edges without replacement by default, which means 
    it returns a fixed number of batches (i.e., num_edges/batch_size), and the 
    positive edges sampled will not be duplicated. However, one can explicitly 
    specify sampling with replacement (replacement = True), that the sampler treats 
    each sampling of a single positive edge as a standalone event. 

    To contorl how many samples the sampler can return, a reset parameter can be used.
    If it is set to true, the sampler will generate samples infinitely. For the sampler 
    with replacement, it will reshuffle the seed edges each time it consumes all the 
    edges and reset the replacement state. If it is set to false, the sampler will only 
    generate num_edges/batch_size samples.

    Note: If node_weight is extremely imbalanced, the sampler will take much longer 
    time to return a minibatch, as sampled negative nodes must not be duplicated for 
    one corruptted positive edge.

    Parameters
    ----------
    g : DGLGraphStale
        The DGLGraphStale where we sample edges.
    batch_size : int
        The batch size (i.e, the number of edges from the graph)
    seed_edges : tensor, optional
        A list of edges where we sample from.
    edge_weight : tensor, optional
        The weight of each edge which decide the change of certain edge being sampled.
    node_weight : tensor, optional
        The weight of each node which decide the change of certain node being sampled.
        Used in negative sampling. If not provided, uniform node sampling is used.
    shuffle : bool, optional
        whether randomly shuffle the list of edges where we sample from.
    num_workers : int, optional
        The number of workers to sample edges in parallel.
    prefetch : bool, optional
        If true, prefetch the samples in the next batch. Default: False
    replacement: bool, optional
        Whether the sampler samples edges with or without repalcement. Default False
    reset: bool, optional
        If true, the sampler will generate samples infinitely, and for the sampler with 
        replacement, it will reshuffle the edges each time it consumes all the edges and 
        reset the replacement state. 
        If false, the sampler will only generate num_edges/batch_size samples by default.
        Default: False.
    negative_mode : string, optional
        The method used to construct negative edges. Possible values are 'head', 'tail'.
    neg_sample_size : int, optional
        The number of negative edges to sample for each edge.
    chunk_size : int, optional
        The chunk size for chunked negative sampling.
    exclude_positive : int, optional
        Whether to exclude positive edges from the negative edges.
    return_false_neg: bool, optional
        Whether to calculate false negative edges and return them as edge data in negative graphs.
    relations: tensor, optional
        relations of the edges if this is a knowledge graph.

    Examples
    --------
    >>> for pos_g, neg_g in EdgeSampler(g, batch_size=10):
    >>>     print(pos_g.head_nid, pos_g.tail_nid)
    >>>     print(neg_g.head_nid, pos_g.tail_nid)
    >>>     print(neg_g.edata['false_neg'])

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
            seed_edges=None,
            edge_weight=None,
            node_weight=None,
            shuffle=False,
            num_workers=1,
            prefetch=False,
            replacement=False,
            reset=False,
            negative_mode="",
            neg_sample_size=0,
            exclude_positive=False,
            return_false_neg=False,
            relations=None,
            chunk_size=None):
        self._g = g
        if self.immutable_only and not g._graph.is_readonly():
            raise NotImplementedError("This loader only support read-only graphs.")

        if relations is None:
            relations = empty((0,), 'int64')
        else:
            relations = utils.toindex(relations)
            relations = relations.todgltensor()
            assert g.number_of_edges() == len(relations)
        self._relations = relations

        if batch_size < 0 or neg_sample_size < 0:
            raise Exception('Invalid arguments')

        self._return_false_neg = return_false_neg
        self._batch_size = int(batch_size)

        if seed_edges is None:
            self._seed_edges = F.arange(0, g.number_of_edges())
        else:
            self._seed_edges = seed_edges

        if shuffle:
            self._seed_edges = F.rand_shuffle(self._seed_edges)
        if edge_weight is None:
            self._is_uniform = True
        else:
            self._is_uniform = False
            self._edge_weight = F.zerocopy_to_dgl_ndarray(F.gather_row(edge_weight, self._seed_edges))
            if node_weight is None:
                self._node_weight = empty((0,), 'float32')
            else:
                self._node_weight = F.zerocopy_to_dgl_ndarray(node_weight)

        self._seed_edges = utils.toindex(self._seed_edges)

        if prefetch:
            self._prefetching_wrapper_class = ThreadPrefetchingWrapper
        self._num_prefetch = num_workers * 2 if prefetch else 0
        self._replacement = replacement
        self._reset = reset

        if chunk_size is None and negative_mode in ('chunk-head', 'chunk-tail'):
            chunk_size = neg_sample_size
        elif chunk_size is None:
            chunk_size = -1

        assert negative_mode in ('', 'head', 'tail', 'chunk-head', 'chunk-tail')

        self._num_workers = int(num_workers)
        self._negative_mode = negative_mode
        self._chunk_size = chunk_size
        self._neg_sample_size = neg_sample_size
        self._exclude_positive = exclude_positive
        if self._is_uniform:
            self._sampler = _CAPI_CreateUniformEdgeSampler(
                self.g._graph,
                self.seed_edges.todgltensor(),
                self._batch_size,       # batch size
                self._num_workers,      # num batches
                self._replacement,
                self._reset,
                self._negative_mode,
                self._neg_sample_size,
                self._exclude_positive,
                self._return_false_neg,
                self._relations,
                self._chunk_size)
        else:
            self._sampler = _CAPI_CreateWeightedEdgeSampler(
                self.g._graph,
                self._seed_edges.todgltensor(),
                self._edge_weight,
                self._node_weight,
                self._batch_size,       # batch size
                self._num_workers,      # num batches
                self._replacement,
                self._reset,
                self._negative_mode,
                self._neg_sample_size,
                self._exclude_positive,
                self._return_false_neg,
                self._relations,
                self._chunk_size)

    def fetch(self, current_index):
        '''
        It returns a list of subgraphs if it only samples positive edges.
        It returns a list of subgraph pairs if it samples both positive edges
        and negative edges.

        Parameters
        ----------
        current_index : int
            deprecated, not used actually.

        Returns
        -------
        list[GraphIndex] or list[(GraphIndex, GraphIndex)]
            Next "bunch" of edges to be processed. 
            If negative_mode is specified, a list of (pos_subg, neg_subg) pairs i
            s returned.
            If return_false_neg is specified as True, the true negative edges and 
            false negative edges in neg_subg is identified in neg_subg.edata['false_neg'].
        '''
        if self._is_uniform:
            subgs = _CAPI_FetchUniformEdgeSample(
                self._sampler)
        else:
            subgs = _CAPI_FetchWeightedEdgeSample(
                self._sampler)

        if len(subgs) == 0:
            return []

        if self._negative_mode == "":
            # If no negative subgraphs.
            return [self.g._create_subgraph(subg,
                                            subg.induced_nodes,
                                            subg.induced_edges) for subg in subgs]
        else:
            rets = []
            assert len(subgs) % 2 == 0
            num_pos = int(len(subgs) / 2)
            for i in range(num_pos):
                pos_subg = EdgeSubgraph(self.g, subgs[i], False)
                neg_subg = EdgeSubgraph(self.g, subgs[i + num_pos], True)
                if self._return_false_neg:
                    exist = _CAPI_GetNegEdgeExistence(subgs[i + num_pos])
                    neg_subg.edata['false_neg'] = utils.toindex(exist).tousertensor()
                rets.append((pos_subg, neg_subg))
            return rets

    def __iter__(self):
        it = SamplerIter(self)
        if self._is_uniform:
            _CAPI_ResetUniformEdgeSample(self._sampler)
        else:
            _CAPI_ResetWeightedEdgeSample(self._sampler)

        if self._num_prefetch:
            return self._prefetching_wrapper_class(it, self._num_prefetch)
        else:
            return it

    @property
    def g(self):
        return self._g

    @property
    def seed_edges(self):
        return self._seed_edges

    @property
    def batch_size(self):
        return self._batch_size

def create_full_nodeflow(g, num_layers, add_self_loop=False):
    """Convert a full graph to NodeFlow to run a L-layer GNN model.

    Parameters
    ----------
    g : DGLGraphStale
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
