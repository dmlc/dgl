"""Negative sampling APIs"""

from numpy.polynomial import polynomial

from .. import backend as F, utils
from .._ffi.function import _init_api
from ..heterograph import DGLGraph

__all__ = ["global_uniform_negative_sampling"]


def _calc_redundancy(
    k_hat, num_edges, num_pairs, r=3
):  # pylint: disable=invalid-name
    # pylint: disable=invalid-name
    # Calculates the number of samples required based on a lower-bound
    # of the expected number of negative samples, based on N draws from
    # a binomial distribution.  Solves the following equation for N:
    #
    # k_hat = N*p_k - r * np.sqrt(N*p_k*(1-p_k))
    #
    # where p_k is the probability that a node pairing is a negative edge
    # and r is the number of standard deviations to construct the lower bound
    #
    # Credits to @zjost
    p_m = num_edges / num_pairs
    p_k = 1 - p_m

    a = p_k**2
    b = -p_k * (2 * k_hat + r**2 * p_m)
    c = k_hat**2

    poly = polynomial.Polynomial([c, b, a])
    N = poly.roots()[-1]
    redundancy = N / k_hat - 1.0
    return redundancy


def global_uniform_negative_sampling(
    g,
    num_samples,
    exclude_self_loops=True,
    replace=False,
    etype=None,
    redundancy=None,
):
    """Performs negative sampling, which generate source-destination pairs such that
    edges with the given type do not exist.

    Specifically, this function takes in an edge type and a number of samples.  It
    returns two tensors ``src`` and ``dst``, the former in the range of ``[0, num_src)``
    and the latter in the range of ``[0, num_dst)``, where ``num_src`` and ``num_dst``
    represents the number of nodes with the source and destination node type respectively.
    It guarantees that no edge will exist between the corresponding pairs of ``src``
    with the source node type and ``dst`` with the destination node type.

    .. note::

       This negative sampler will try to generate as many negative samples as possible, but
       it may rarely return less than :attr:`num_samples` negative samples.
       This is more likely to happen when a graph is so small or dense that not many
       unique negative samples exist.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    num_samples : int
        The number of desired negative samples to generate.
    exclude_self_loops : bool, optional
        Whether to exclude self-loops from the negative samples.  Only impacts the
        edge types whose source and destination node types are the same.

        Default: True.
    replace : bool, optional
        Whether to sample with replacement.  Setting it to True will make things
        faster.  (Default: False)
    etype : str or tuple of str, optional
        The edge type.  Can be omitted if the graph only has one edge type.
    redundancy : float, optional
        Indicates how much more negative samples to actually generate during rejection sampling
        before finding the unique pairs.

        Increasing it will increase the likelihood of getting :attr:`num_samples` negative
        samples, but will also take more time and memory.

        (Default: automatically determined by the density of graph)

    Returns
    -------
    tuple[Tensor, Tensor]
        The source and destination pairs.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> dgl.sampling.global_uniform_negative_sampling(g, 3)
    (tensor([0, 1, 3]), tensor([2, 0, 2]))
    """
    if etype is None:
        etype = g.etypes[0]
    utype, _, vtype = g.to_canonical_etype(etype)
    exclude_self_loops = exclude_self_loops and (utype == vtype)

    redundancy = _calc_redundancy(
        num_samples, g.num_edges(etype), g.num_nodes(utype) * g.num_nodes(vtype)
    )

    etype_id = g.get_etype_id(etype)
    src, dst = _CAPI_DGLGlobalUniformNegativeSampling(
        g._graph,
        etype_id,
        num_samples,
        3,
        exclude_self_loops,
        replace,
        redundancy,
    )
    return F.from_dgl_nd(src), F.from_dgl_nd(dst)


DGLGraph.global_uniform_negative_sampling = utils.alias_func(
    global_uniform_negative_sampling
)

_init_api("dgl.sampling.negative", __name__)
