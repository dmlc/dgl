
from ... import utils
from ... import backend as F
from ..._ffi.function import _init_api

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
    traces = _CAPI_DGLRandomWalk(g._graph,
            seeds, int(num_traces), int(num_hops))
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
    traces = _CAPI_DGLRandomWalkWithRestart(
            g._graph, seeds, restart_prob, int(max_nodes_per_seed),
            int(max_visit_counts), int(max_frequent_visited_nodes))
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
    The current implementation does not ensure that the graph is a bipartite
    graph.

    The traces does **not** include the seed nodes themselves.

    Reference
    ---------
    [1] Eksombatchai et al., 2017 https://arxiv.org/abs/1711.07601
    """
    if len(seeds) == 0:
        return []
    seeds = utils.toindex(seeds).todgltensor()
    traces = _CAPI_DGLBipartiteSingleSidedRandomWalkWithRestart(
            g._graph, seeds, restart_prob, int(max_nodes_per_seed),
            int(max_visit_counts), int(max_frequent_visited_nodes))
    return _split_traces(traces)

_init_api('dgl.randomwalk', __name__)
