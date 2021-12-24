"""Negative sampling APIs"""

from .._ffi.function import _init_api
from .. import backend as F

__all__ = [
    'global_uniform_negative_sampling']

def global_uniform_negative_sampling(
        g, num_samples, exclude_self_loops=True, unique=True, etype=None, num_trials=3,
        redundancy=1.3):
    """Performs negative sampling, which generate source-destination pairs such that
    edges with the given type do not exist.

    Specifically, this function takes in an edge type and a number of samples.  It
    returns two tensors ``src`` and ``dst``, the former in the range of ``[0, num_src)``
    and the latter in the range of ``[0, num_dst)``, where ``num_src`` and ``num_dst``
    represents the number of nodes with the source and destination node type respectively.
    It guarantees that no edge will exist between the corresponding pairs of ``src``
    with the source node type and ``dst`` with the destination node type.

    .. note::

       This function uses rejection sampling, and may not always return the same number
       of negative examples as the given :attr:`num_samples`.  This is more likely to
       happen when a graph is so small or dense that not many unique negative examples exist.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    num_samples : int
        The number of negative examples to generate.
    exclude_self_loops : bool, optional
        Whether to exclude self-loops from the negative examples.  Only impacts the
        edge types whose source and destination node types are the same.

        Default: True.
    unique : bool, optional
        Whether to sample unique negative examples.  Setting it to False will make things
        faster.  (Default: True)
    etype : str or tuple of str, optional
        The edge type.  Can be omitted if the graph only has one edge type.
    num_trials : int, optional
        The number of rejection sampling trials.

        Increasing it will increase the likelihood of getting :attr:`num_samples` negative
        examples, but will also take more time.
    redundancy : float, optional
        Indicates how much more negative examples to actually generate during rejection sampling
        before finding the unique pairs.

        Increasing it will increase the likelihood of getting :attr:`num_samples` negative
        examples, but will also take more time and memory.

        (Default: 1.3)

    Returns
    -------
    tuple[Tensor, Tensor]
        The source and destination pairs.

    Examples
    --------
    """
    if len(g.etypes) > 1:
        utype, _, vtype = g.to_canonical_etype(etype)
        exclude_self_loops = exclude_self_loops and (utype == vtype)

    etype_id = g.get_etype_id(etype)
    src, dst = _CAPI_DGLGlobalUniformNegativeSampling(
        g._graph, etype_id, num_samples, num_trials, exclude_self_loops, unique, redundancy)
    return F.from_dgl_nd(src), F.from_dgl_nd(dst)

_init_api('dgl.sampling.negative', __name__)
