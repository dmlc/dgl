"""Node2vec random walk"""

from .. import backend as F, ndarray as nd, utils
from .._ffi.function import _init_api

# pylint: disable=invalid-name

__all__ = ["node2vec_random_walk"]


def node2vec_random_walk(
    g, nodes, p, q, walk_length, prob=None, return_eids=False
):
    """
    Generate random walk traces from an array of starting nodes based on the node2vec model.
    Paper: `node2vec: Scalable Feature Learning for Networks
    <https://arxiv.org/abs/1607.00653>`__.

    The returned traces all have length ``walk_length + 1``, where the first node
    is the starting node itself.

    Note that if a random walk stops in advance, DGL pads the trace with -1 to have the same
    length.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Must be on CPU.

        Note that node2vec only support homogeneous graph.
    nodes : Tensor
        Node ID tensor from which the random walk traces starts.

        The tensor must be on CPU, and must have the same dtype as the ID type
        of the graph.
    p: float
        Likelihood of immediately revisiting a node in the walk.
    q: float
        Control parameter to interpolate between breadth-first strategy and depth-first strategy.
    walk_length: int
        Length of random walks.
    prob : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly.
    return_eids : bool, optional
        If True, additionally return the edge IDs traversed.

        Default: False.

    Returns
    -------
    traces : Tensor
        A 2-dimensional node ID tensor with shape ``(num_seeds, walk_length + 1)``.
    eids : Tensor, optional
        A 2-dimensional edge ID tensor with shape ``(num_seeds, length)``.
        Only returned if :attr:`return_eids` is True.

    Examples
    --------
    >>> g1 = dgl.graph(([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]))
    >>> dgl.sampling.node2vec_random_walk(g1, [0, 1, 2, 0], 1, 1, walk_length=4)
    tensor([[0, 1, 3, 0, 1],
            [1, 2, 0, 1, 3],
            [2, 0, 1, 3, 0],
            [0, 1, 2, 0, 1]])

    >>> dgl.sampling.node2vec_random_walk(g1, [0, 1, 2, 0], 1, 1, walk_length=4, return_eids=True)
    (tensor([[0, 1, 3, 0, 1],
             [1, 2, 0, 1, 2],
             [2, 0, 1, 2, 0],
             [0, 1, 2, 0, 1]]),
     tensor([[0, 2, 4, 0],
             [1, 3, 0, 1],
             [3, 0, 1, 3],
             [0, 1, 3, 0]]))
    """
    assert g.device == F.cpu(), "Graph must be on CPU."

    gidx = g._graph
    nodes = F.to_dgl_nd(utils.prepare_tensor(g, nodes, "nodes"))

    if prob is None:
        prob_nd = nd.array([], ctx=nodes.ctx)
    else:
        prob_nd = F.to_dgl_nd(g.edata[prob])

    traces, eids = _CAPI_DGLSamplingNode2vec(
        gidx, nodes, p, q, walk_length, prob_nd
    )

    traces = F.from_dgl_nd(traces)
    eids = F.from_dgl_nd(eids)

    return (traces, eids) if return_eids else traces


_init_api("dgl.sampling.randomwalks", __name__)
