"""Random walk routines
"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError
from .. import ndarray as nd
from .. import utils

__all__ = [
    'random_walk',
    'pack_traces']

def random_walk(g, nodes, *, metapath=None, length=None, prob=None, restart_prob=None):
    """Generate random walk traces from an array of starting nodes based on the given metapath.

    For a single starting node, ``num_traces`` traces would be generated.  A trace would

    1. Start from the given node and set ``t`` to 0.
    2. Pick and traverse along edge type ``metapath[t]`` from the current node.
    3. If no edge can be found, halt.  Otherwise, increment ``t`` and go to step 2.

    The returned traces all have length ``len(metapath) + 1``, where the first node
    is the starting node itself.

    If a random walk stops in advance, DGL pads the trace with -1 to have the same
    length.

    Parameters
    ----------
    g : DGLGraph
        The graph.  Must be on CPU.
    nodes : Tensor
        Node ID tensor from which the random walk traces starts.

        The tensor must be on CPU, and must have the same dtype as the ID type
        of the graph.
    metapath : list[str or tuple of str], optional
        Metapath, specified as a list of edge types.

        Mutually exclusive with :attr:`length`.

        If omitted, DGL assumes that ``g`` only has one node & edge type.  In this
        case, the argument ``length`` specifies the length of random walk traces.
    length : int, optional
        Length of random walks.

        Mutually exclusive with :attr:`metapath`.

        Only used when :attr:`metapath` is None.
    prob : str, optional
        The name of the edge feature tensor on the graph storing the (unnormalized)
        probabilities associated with each edge for choosing the next node.

        The feature tensor must be non-negative and the sum of the probabilities
        must be positive for the outbound edges of all nodes (although they don't have
        to sum up to one).  The result will be undefined otherwise.

        If omitted, DGL assumes that the neighbors are picked uniformly.
    restart_prob : float or Tensor, optional
        Probability to terminate the current trace before each transition.

        If a tensor is given, :attr:`restart_prob` should have the same length as
        :attr:`metapath` or :attr:`length`.

    Returns
    -------
    traces : Tensor
        A 2-dimensional node ID tensor with shape ``(num_seeds, len(metapath) + 1)`` or
        ``(num_seeds, length + 1)`` if :attr:`metapath` is None.
    types : Tensor
        A 1-dimensional node type ID tensor with shape ``(len(metapath) + 1)`` or
        ``(length + 1)``.
        The type IDs match the ones in the original graph ``g``.

    Notes
    -----
    The returned tensors are on CPU.

    Examples
    --------
    The following creates a homogeneous graph:
    >>> g1 = dgl.graph(([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]))

    Normal random walk:

    >>> dgl.sampling.random_walk(g1, [0, 1, 2, 0], length=4)
    (tensor([[0, 1, 2, 0, 1],
             [1, 3, 0, 1, 3],
             [2, 0, 1, 3, 0],
             [0, 1, 2, 0, 1]]), tensor([0, 0, 0, 0, 0]))

    The first tensor indicates the random walk path for each seed node.
    The j-th element in the second tensor indicates the node type ID of the j-th node
    in every path.  In this case, it is returning all 0.

    Random walk with restart:

    >>> dgl.sampling.random_walk_with_restart(g1, [0, 1, 2, 0], length=4, restart_prob=0.5)
    (tensor([[ 0, -1, -1, -1, -1],
             [ 1,  3,  0, -1, -1],
             [ 2, -1, -1, -1, -1],
             [ 0, -1, -1, -1, -1]]), tensor([0, 0, 0, 0, 0]))

    Non-uniform random walk:

    >>> g1.edata['p'] = torch.FloatTensor([1, 0, 1, 1, 1])     # disallow going from 1 to 2
    >>> dgl.sampling.random_walk(g1, [0, 1, 2, 0], length=4, prob='p')
    (tensor([[0, 1, 3, 0, 1],
             [1, 3, 0, 1, 3],
             [2, 0, 1, 3, 0],
             [0, 1, 3, 0, 1]]), tensor([0, 0, 0, 0, 0]))

    Metapath-based random walk:

    >>> g2 = dgl.heterograph({
    ...     ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ...     ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ...     ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])
    >>> dgl.sampling.random_walk(
    ...     g2, [0, 1, 2, 0], metapath=['follow', 'view', 'viewed-by'] * 2)
    (tensor([[0, 1, 1, 1, 2, 2, 3],
             [1, 3, 1, 1, 2, 2, 2],
             [2, 0, 1, 1, 3, 1, 1],
             [0, 1, 1, 0, 1, 1, 3]]), tensor([0, 0, 1, 0, 0, 1, 0]))

    Metapath-based random walk, with restarts only on items (i.e. after traversing a "view"
    relationship):

    >>> dgl.sampling.random_walk(
    ...     g2, [0, 1, 2, 0], metapath=['follow', 'view', 'viewed-by'] * 2,
    ...     restart_prob=torch.FloatTensor([0, 0.5, 0, 0, 0.5, 0]))
    (tensor([[ 0,  1, -1, -1, -1, -1, -1],
             [ 1,  3,  1,  0,  1,  1,  0],
             [ 2,  0,  1,  1,  3,  2,  2],
             [ 0,  1,  1,  3,  0,  0,  0]]), tensor([0, 0, 1, 0, 0, 1, 0]))
    """
    assert g.device == F.cpu(), "Graph must be on CPU."
    n_etypes = len(g.canonical_etypes)
    n_ntypes = len(g.ntypes)

    if metapath is None:
        if n_etypes > 1 or n_ntypes > 1:
            raise DGLError("metapath not specified and the graph is not homogeneous.")
        if length is None:
            raise ValueError("Please specify either the metapath or the random walk length.")
        metapath = [0] * length
    else:
        metapath = [g.get_etype_id(etype) for etype in metapath]

    gidx = g._graph
    nodes = F.to_dgl_nd(utils.prepare_tensor(g, nodes, 'nodes'))
    metapath = F.to_dgl_nd(utils.prepare_tensor(g, metapath, 'metapath'))

    # Load the probability tensor from the edge frames
    if prob is None:
        p_nd = [nd.array([], ctx=nodes.ctx) for _ in g.canonical_etypes]
    else:
        p_nd = []
        for etype in g.canonical_etypes:
            if prob in g.edges[etype].data:
                prob_nd = F.to_dgl_nd(g.edges[etype].data[prob])
                if prob_nd.ctx != nodes.ctx:
                    raise ValueError(
                        'context of seed node array and edges[%s].data[%s] are different' %
                        (etype, prob))
            else:
                prob_nd = nd.array([], ctx=nodes.ctx)
            p_nd.append(prob_nd)

    # Actual random walk
    if restart_prob is None:
        traces, types = _CAPI_DGLSamplingRandomWalk(gidx, nodes, metapath, p_nd)
    elif F.is_tensor(restart_prob):
        restart_prob = F.to_dgl_nd(restart_prob)
        traces, types = _CAPI_DGLSamplingRandomWalkWithStepwiseRestart(
            gidx, nodes, metapath, p_nd, restart_prob)
    else:
        traces, types = _CAPI_DGLSamplingRandomWalkWithRestart(
            gidx, nodes, metapath, p_nd, restart_prob)

    traces = F.from_dgl_nd(traces)
    types = F.from_dgl_nd(types)
    return traces, types

def pack_traces(traces, types):
    """Pack the padded traces returned by ``random_walk()`` into a concatenated array.
    The padding values (-1) are removed, and the length and offset of each trace is
    returned along with the concatenated node ID and node type arrays.

    Parameters
    ----------
    traces : Tensor
        A 2-dimensional node ID tensor.  Must be on CPU and either ``int32`` or ``int64``.
    types : Tensor
        A 1-dimensional node type ID tensor.  Must be on CPU and either ``int32`` or ``int64``.

    Returns
    -------
    concat_vids : Tensor
        An array of all node IDs concatenated and padding values removed.
    concat_types : Tensor
        An array of node types corresponding for each node in ``concat_vids``.
        Has the same length as ``concat_vids``.
    lengths : Tensor
        Length of each trace in the original traces tensor.
    offsets : Tensor
        Offset of each trace in the originial traces tensor in the new concatenated tensor.

    Notes
    -----
    The returned tensors are on CPU.

    Examples
    --------
    >>> g2 = dgl.heterograph({
    ...     ('user', 'follow', 'user'): ([0, 1, 1, 2, 3], [1, 2, 3, 0, 0]),
    ...     ('user', 'view', 'item'): ([0, 0, 1, 2, 3, 3], [0, 1, 1, 2, 2, 1]),
    ...     ('item', 'viewed-by', 'user'): ([0, 1, 1, 2, 2, 1], [0, 0, 1, 2, 3, 3])
    >>> traces, types = dgl.sampling.random_walk(
    ...     g2, [0, 0], metapath=['follow', 'view', 'viewed-by'] * 2,
    ...     restart_prob=torch.FloatTensor([0, 0.5, 0, 0, 0.5, 0]))
    >>> traces, types
    (tensor([[ 0,  1, -1, -1, -1, -1, -1],
             [ 0,  1,  1,  3,  0,  0,  0]]), tensor([0, 0, 1, 0, 0, 1, 0]))
    >>> concat_vids, concat_types, lengths, offsets = dgl.sampling.pack_traces(traces, types)
    >>> concat_vids
    tensor([0, 1, 0, 1, 1, 3, 0, 0, 0])
    >>> concat_types
    tensor([0, 0, 0, 0, 1, 0, 0, 1, 0])
    >>> lengths
    tensor([2, 7])
    >>> offsets
    tensor([0, 2]))

    The first tensor ``concat_vids`` is the concatenation of all paths, i.e. flattened array
    of ``traces``, excluding all padding values (-1).

    The second tensor ``concat_types`` stands for the node type IDs of all corresponding nodes
    in the first tensor.

    The third and fourth tensor indicates the length and the offset of each path.  With these
    tensors it is easy to obtain the i-th random walk path with:

    >>> vids = concat_vids.split(lengths.tolist())
    >>> vtypes = concat_vtypes.split(lengths.tolist())
    >>> vids[1], vtypes[1]
    (tensor([0, 1, 1, 3, 0, 0, 0]), tensor([0, 0, 1, 0, 0, 1, 0]))
    """
    assert F.is_tensor(traces) and F.context(traces) == F.cpu(), "traces must be a CPU tensor"
    assert F.is_tensor(types) and F.context(types) == F.cpu(), "types must be a CPU tensor"
    traces = F.to_dgl_nd(traces)
    types = F.to_dgl_nd(types)

    concat_vids, concat_types, lengths, offsets = _CAPI_DGLSamplingPackTraces(traces, types)

    concat_vids = F.from_dgl_nd(concat_vids)
    concat_types = F.from_dgl_nd(concat_types)
    lengths = F.from_dgl_nd(lengths)
    offsets = F.from_dgl_nd(offsets)

    return concat_vids, concat_types, lengths, offsets

_init_api('dgl.sampling.randomwalks', __name__)
