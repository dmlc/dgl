
import numpy as np
from ... import utils
from ... import backend as F
from ..._ffi.function import _init_api
from ..._ffi.object import register_object, ObjectBase
from ... import ndarray
from ...base import dgl_warning

__all__ = ['random_walk',
           'random_walk_with_restart',
           'bipartite_single_sided_random_walk_with_restart',
           'metapath_random_walk',
           ]

@register_object('sampler.RandomWalkTraces')
class RandomWalkTraces(ObjectBase):
    pass

def random_walk(g, seeds, num_traces, num_hops):
    """**DEPRECATED**: please use :func:`dgl.sampling.random_walk` instead.

    Batch-generate random walk traces on given graph with the same length.

    Parameters
    ----------
    g : DGLGraphStale
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
    dgl_warning(
        "This function is deprecated; please use dgl.sampling.random_walk instead",
        DeprecationWarning)
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
    traces : RandomWalkTraces

    Returns
    -------
    traces : list[list[Tensor]]
        traces[i][j] is the j-th trace generated for i-th seed.
    """
    trace_counts = traces.trace_counts.asnumpy().tolist()
    trace_vertices = F.zerocopy_from_dgl_ndarray(traces.vertices)
    trace_vertices = F.split(
            trace_vertices, traces.trace_lengths.asnumpy().tolist(), 0)

    results = []
    s = 0
    for c in trace_counts:
        results.append(trace_vertices[s:s+c])
        s += c

    return results


def random_walk_with_restart(
        g, seeds, restart_prob, max_nodes_per_seed,
        max_visit_counts=0, max_frequent_visited_nodes=0):
    """**DEPRECATED**: please use :func:`dgl.sampling.random_walk` instead.

    Batch-generate random walk traces on given graph with restart probability.

    Parameters
    ----------
    g : DGLGraphStale
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
    dgl_warning(
        "This function is deprecated; please use dgl.sampling.random_walk instead",
        DeprecationWarning)
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
    """**DEPRECATED**: please use :func:`dgl.sampling.random_walk` instead.

    Batch-generate random walk traces on given graph with restart probability.

    The graph must be a bipartite graph.

    A single random walk step involves two normal steps, so that the "visited"
    nodes always stay on the same side. [1]

    Parameters
    ----------
    g : DGLGraphStale
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
    dgl_warning(
        "This function is deprecated; please use dgl.sampling.random_walk instead",
        DeprecationWarning)
    if len(seeds) == 0:
        return []
    seeds = utils.toindex(seeds).todgltensor()
    traces = _CAPI_DGLBipartiteSingleSidedRandomWalkWithRestart(
            g._graph, seeds, restart_prob, int(max_nodes_per_seed),
            int(max_visit_counts), int(max_frequent_visited_nodes))
    return _split_traces(traces)


def metapath_random_walk(hg, etypes, seeds, num_traces):
    """**DEPRECATED**: please use :func:`dgl.sampling.random_walk` instead.

    For a single seed node, ``num_traces`` traces would be generated.  A trace would

    1. Start from the given seed and set ``t`` to 0.
    2. Pick and traverse along edge type ``etypes[t % len(etypes)]`` from the current node.
    3. If no edge can be found, halt.  Otherwise, increment ``t`` and go to step 2.

    Parameters
    ----------
    hg : DGLHeteroGraph
        The heterogeneous graph.
    etypes : list[str or tuple of str]
        Metapath, specified as a list of edge types.
        The beginning and ending node type must be the same.
    seeds : Tensor
        The seed nodes.  Node type is the same as the beginning node type of metapath.
    num_traces : int
        The number of traces

    Returns
    -------
    traces : list[list[Tensor]]
        traces[i][j] is the j-th trace generated for i-th seed.
        traces[i][j][k] would have node type the same as the destination node type of edge
        type ``etypes[k % len(etypes)]``

    Notes
    -----
    The traces does **not** include the seed nodes themselves.
    """
    dgl_warning(
        "This function is deprecated; please use dgl.sampling.random_walk instead",
        DeprecationWarning)
    if len(etypes) == 0:
        raise ValueError('empty metapath')
    if hg.to_canonical_etype(etypes[0])[0] != hg.to_canonical_etype(etypes[-1])[2]:
        raise ValueError('beginning and ending node type mismatch')
    if len(seeds) == 0:
        return []
    etype_array = ndarray.array(np.asarray([hg.get_etype_id(et) for et in etypes], dtype="int64"))
    seed_array = utils.toindex(seeds, hg._idtype_str).todgltensor()
    traces = _CAPI_DGLMetapathRandomWalk(hg._graph, etype_array, seed_array, num_traces)
    return _split_traces(traces)

_init_api('dgl.sampler.randomwalk', __name__)
