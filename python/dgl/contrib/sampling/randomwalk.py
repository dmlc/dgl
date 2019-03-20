
from ... import utils
from ... import backend as F
from ..._ffi.function import _init_api
from ...utils import unwrap_to_ptr_list
from ...nodeflow import NodeFlow
from .sampler import NodeFlowSampler

__all__ = ['random_walk',
           'random_walk_with_restart',
           'bipartite_single_sided_random_walk_with_restart',
           ]


def random_walk(g, seeds, num_traces, num_hops):
    """Batch-generate random walk traces on given graph with the same length.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    seeds : Tensor
        The node ID tensor from which the random walk traces starts.
    num_traces : int
        Number of traces to generate for each seed.
    num_hops : int
        Number of hops for each trace.

    Returns
    -------
    traces : Tensor
        A 3-dimensional node ID tensor with shape

            (num_seeds, num_traces, num_hops + 1)

        traces[i, j, 0] are always starting nodes (i.e. seed[i]).
    """
    if len(seeds) == 0:
        return utils.toindex([]).tousertensor()
    seeds = utils.toindex(seeds).todgltensor()
    traces = _CAPI_RandomWalk(g._graph._handle, seeds, num_traces, num_hops)
    return F.zerocopy_from_dlpack(traces.to_dlpack())


def _split_traces(traces):
    """Splits the flattened RandomWalkTraces structure into list of list
    of tensors.

    Parameters
    ----------
    traces : PackedFunc object of RandomWalkTraces structure

    Returns
    -------
    traces : list[list[Tensor]]
        traces[i][j] is the j-th trace generated for i-th seed.
    """
    trace_counts = F.zerocopy_to_numpy(
            F.zerocopy_from_dlpack(traces(0).to_dlpack())).tolist()
    trace_lengths = F.zerocopy_from_dlpack(traces(1).to_dlpack())
    trace_vertices = F.zerocopy_from_dlpack(traces(2).to_dlpack())

    trace_vertices = F.split(
            trace_vertices, F.zerocopy_to_numpy(trace_lengths).tolist(), 0)

    traces = []
    s = 0
    for c in trace_counts:
        traces.append(trace_vertices[s:s+c])
        s += c

    return traces


def random_walk_with_restart(
        g, seeds, restart_prob, max_nodes_per_seed,
        max_visit_counts=0, max_frequent_visited_nodes=0):
    """Batch-generate random walk traces on given graph with restart probability.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    seeds : Tensor
        The node ID tensor from which the random walk traces starts.
    restart_prob : float
        Probability to stop a random walk after each step.
    max_nodes_per_seed : int
        Stop generating traces for a seed if the total number of nodes
        visited exceeds this number. [1]
    max_visit_counts : int, optional
    max_frequent_visited_nodes : int, optional
        Alternatively, stop generating traces for a seed if no less than
        ``max_frequent_visited_nodes`` are visited no less than
        ``max_visit_counts`` times.  [1]

    Returns
    -------
    traces : list[list[Tensor]]
        traces[i][j] is the j-th trace generated for i-th seed.

    Notes
    -----
    The traces does **not** include the seed nodes themselves.

    Reference
    ---------
    [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
    """
    if len(seeds) == 0:
        return []
    seeds = utils.toindex(seeds).todgltensor()
    traces = _CAPI_RandomWalkWithRestart(
            g._graph._handle, seeds, restart_prob, max_nodes_per_seed,
            max_visit_counts, max_frequent_visited_nodes)
    return _split_traces(traces)


def bipartite_single_sided_random_walk_with_restart(
        g, seeds, restart_prob, max_nodes_per_seed,
        max_visit_counts=0, max_frequent_visited_nodes=0):
    """Batch-generate random walk traces on given graph with restart probability.

    The graph must be a bipartite graph.

    A single random walk step involves two normal steps, so that the "visited"
    nodes always stay on the same side. [1]

    Parameters
    ----------
    g : DGLGraph
        The graph.
    seeds : Tensor
        The node ID tensor from which the random walk traces starts.
    restart_prob : float
        Probability to stop a random walk after each step.
    max_nodes_per_seed : int
        Stop generating traces for a seed if the total number of nodes
        visited exceeds this number. [1]
    max_visit_counts : int, optional
    max_frequent_visited_nodes : int, optional
        Alternatively, stop generating traces for a seed if no less than
        ``max_frequent_visited_nodes`` are visited no less than
        ``max_visit_counts`` times.  [1]

    Returns
    -------
    traces : list[list[Tensor]]
        traces[i][j] is the j-th trace generated for i-th seed.

    Notes
    -----
    The current implementation does not verify whether the graph is a
    bipartite graph.

    The traces does **not** include the seed nodes themselves.

    Reference
    ---------
    [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
    """
    if len(seeds) == 0:
        return []
    seeds = utils.toindex(seeds).todgltensor()
    traces = _CAPI_BipartiteSingleSidedRandomWalkWithRestart(
            g._graph._handle, seeds, restart_prob, max_nodes_per_seed,
            max_visit_counts, max_frequent_visited_nodes)
    return _split_traces(traces)


class BasePPRNeighborSampler(NodeFlowSampler):
    '''Base PPR neighbor sampling class
    '''
    capi = None

    def __init__(
            self,
            g,
            batch_size,
            num_hops,
            top_t,
            max_nodes_per_seed,
            seed_nodes=None,
            shuffle=False,
            restart_prob=0.1,
            max_visit_counts=0,
            max_frequent_visited_nodes=0,
            num_workers=1,
            prefetch=False):
        super(BasePPRNeighborSampler, self).__init__(
                g, batch_size, seed_nodes, shuffle, num_workers * 2 if prefetch else 0)

        self._restart_prob = restart_prob
        self._max_visit_counts = max_visit_counts
        self._max_frequent_visited_nodes = max_frequent_visited_nodes
        self._num_hops = num_hops
        self._top_t = top_t
        self._num_workers = num_workers
        self._max_nodes_per_seed = max_nodes_per_seed

    def fetch(self, current_nodeflow_index):
        handles = unwrap_to_ptr_list(self.capi(
            self.g.c_handle,
            self.seed_nodes.todgltensor(),
            current_nodeflow_index,
            self.batch_size,
            self._num_workers,
            self._restart_prob,
            self._max_nodes_per_seed,
            self._max_visit_counts,
            self._max_frequent_visited_nodes,
            self._num_hops,
            self._top_t))
        nflows = [NodeFlow(self.g, hdl, edata_key='ppr_weight') for hdl in handles]
        return nflows


_init_api('dgl.randomwalk', __name__)


class PPRBipartiteSingleSidedNeighborSampler(BasePPRNeighborSampler):
    '''Create a sampler that, given a node on a bipartite graph, samples the
    top-k most important neighborhood of that node, using visit counts of
    random walks with restart.

    Note: this only works on undirected bipartite graph.

    Parameters
    ----------
    g : DGLGraph
        The graph
    batch_size : int
        The batch size
    num_hops : int
        Number of random walk hops.
    top_t : int
        Number of most important neighbors to pick.
    max_nodes_per_seed : int
        Stop generating traces for a seed if total number of visited nodes
        exceed this amount.
    seed_nodes : Tensor, optional
        A 1D tensor list of nodes where we sample NodeFlows from.
        If None, the seed vertices are all the vertices in the graph.
        Default: None
    shuffle : bool, optional
        Indicates the sampled NodeFlows are shuffled.  Default: False
    max_visit_counts : int, optional
    max_frequent_visited_nodes : int, optional
        Alternatively, stop generating traces for a seed if no less than
        ``max_frequent_visited_nodes`` are visited no less than
        ``max_visit_counts`` times.  [1]
    num_workers : int, optional
        The number of worker threads that sample NodeFlows in parallel. Default: 1
    prefetch : bool, optional
        If true, prefetch the samples in the next batch. Default: False

    Notes
    -----
    The current implementation does not verify whether the graph is a
    bipartite graph.

    The importance of a neighbor to a node is stored on the edges of the NodeFlow
    with column name ``ppr_weight``.

    Reference
    ---------
    [1] Eksombatchai et al., 2017
    '''
    capi = _CAPI_PPRBipartiteSingleSidedNeighborSampling

class PPRNeighborSampler(BasePPRNeighborSampler):
    '''Create a sampler that samples the top-k most important neighborhood of
    that node, using visit counts of random walks with restart.

    Parameters
    ----------
    g : DGLGraph
        The graph
    batch_size : int
        The batch size
    num_hops : int
        Number of random walk hops.
    top_t : int
        Number of most important neighbors to pick.
    max_nodes_per_seed : int
        Stop generating traces for a seed if total number of visited nodes
        exceed this amount.
    seed_nodes : Tensor, optional
        A 1D tensor list of nodes where we sample NodeFlows from.
        If None, the seed vertices are all the vertices in the graph.
        Default: None
    shuffle : bool, optional
        Indicates the sampled NodeFlows are shuffled.  Default: False
    max_visit_counts : int, optional
    max_frequent_visited_nodes : int, optional
        Alternatively, stop generating traces for a seed if no less than
        ``max_frequent_visited_nodes`` are visited no less than
        ``max_visit_counts`` times.  [1]
    num_workers : int, optional
        The number of worker threads that sample NodeFlows in parallel. Default: 1
    prefetch : bool, optional
        If true, prefetch the samples in the next batch. Default: False

    Notes
    -----
    The importance of a neighbor to a node is stored on the edges of the NodeFlow
    with column name ``ppr_weight``.

    Reference
    ---------
    [1] Eksombatchai et al., 2017
    '''
    capi = _CAPI_PPRNeighborSampling
