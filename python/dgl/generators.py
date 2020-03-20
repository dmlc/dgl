"""Module for various graph generator functions."""

from . import backend as F
from . import convert
from . import random

__all__ = ['rand_graph']

def rand_graph(num_nodes, num_edges, restrict_format='any'):
    """Generate a random graph of the given number of edges.

    It uniformly chooses ``num_edges`` from all pairs and form a graph.

    TODO(minjie): support RNG as one of the arguments.

    Parameters
    ----------
    num_nodes : int
        The number of nodes
    num_edges : int
        The number of edges
    restrict_format : 'any', 'coo', 'csr', 'csc', optional
        Force the storage format. Default: 'any' (i.e. let DGL decide what to use).

    Returns
    -------
    DGLHeteroGraph
        Generated random graph.
    """
    eids = random.choice(num_nodes * num_nodes, num_edges, replace=False)
    rows = F.astype(eids / num_nodes, F.dtype(eids))
    cols = F.astype(eids % num_nodes, F.dtype(eids))
    g = convert.graph((rows, cols),
                      num_nodes=num_nodes, validate=False,
                      restrict_format=restrict_format)
    return g
