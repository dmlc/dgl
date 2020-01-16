"""Random walk routines
"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError
from .. import ndarray as nd

__all__ = [
    'random_walk']

def random_walk(g, nodes, *, metapath=None, length=None, p=None):
    """Generate random walk traces from an array of seed nodes (or starting nodes),
    based on the given metapath.
   
    For a single seed node, ``num_traces`` traces would be generated.  A trace would
   
    1. Start from the given seed and set ``t`` to 0.
    2. Pick and traverse along edge type ``metapath[t]`` from the current node.
    3. If no edge can be found, halt.  Otherwise, increment ``t`` and go to step 2.
   
    The returned traces all have length ``len(metapath) + 1``, where the first node
    is the seed node itself.
   
    If a random walk stops in advance, the trace is padded with -1 to have the same
    length.
   
    Parameters
    ----------
    g : DGLGraph
        The graph.
    nodes : Tensor
        Node ID tensor from which the random walk traces starts.
    metapath : list[str or tuple of str], optional
        Metapath, specified as a list of edge types.
        If omitted, we assume that ``g`` only has one node & edge type.  In this
        case, the argument ``length`` specifies the length of random walk traces.
    length : int, optional
        Length of random walks.
        Affects only when ``metapath`` is omitted.
    p : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.
        The feature tensor must be non-negative.
        If omitted, we assume the neighbors are picked uniformly.
 
    Returns
    -------
    traces : Tensor
        A 2-dimensional node ID tensor with shape (num_seeds, len(metapath) + 1).
    types : Tensor
        A 2-dimensional node type ID tensor with shape (num_seeds, len(metapath) + 1).
        The type IDs match the ones in the original graph ``g``.
    """
    n_etypes = len(g.canonical_etypes)
    n_ntypes = len(g.ntypes)

    if metapath is None:
        if n_etypes > 1 or n_ntypes > 1:
            raise DGLError("metapath not specified and the graph is not homogeneous.")
        if length is None:
            raise ValueError("Please specify either the metapath or the random walk length.")
        metapath = [0] * length
    else:
        metapath = np.array([g.get_etype_id(etype) for etype in metapath])

    gi = g._graph
    nodes = F.zerocopy_to_dgl_ndarray(nodes)
    metapath = nd.array(metapath, ctx=nodes.ctx)

    if p is None:
        p_nd = [nd.array([], ctx=nodes.ctx) for _ in g.canonical_etypes]
    else:
        p_nd = []
        for etype in canonical_etypes:
            if p in g.edges[etype].data:
                prob_nd = F.zerocopy_to_dgl_ndarray(g.edges[etype].data[p])
                if prob_nd.ctx != nodes.ctx:
                    raise ValueError(
                        'context of seed node array and edges[%s].data[%s] are different' %
                        (etype, p))
            else:
                prob_nd = nd.array([], ctx=nodes.ctx)
            p_nd.append(prob_nd)

    traces, types = _CAPI_DGLSamplingRandomWalk(gi, nodes, metapath, p_nd)
    traces = F.zerocopy_from_dgl_ndarray(traces)
    types = F.zerocopy_from_dgl_ndarray(types)
    return traces, types

_init_api('dgl.sampling.randomwalks', __name__)
