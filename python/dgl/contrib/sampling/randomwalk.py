
from ... import utils
from ... import backend as F

__all__ = ['random_walk']


def random_walk(g, seeds, num_traces, num_hops):
    """Batch-generate random walk traces on given graph with the same length.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Must be readonly.
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
    return g._graph.random_walk(utils.toindex(seeds), num_traces, num_hops)
