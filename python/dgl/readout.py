"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

from .base import DGLError
from . import backend as F
from .ops import segment

__all__ = ['readout_nodes', 'readout_edges',
           'sum_nodes', 'sum_edges', 'mean_nodes', 'mean_edges',
           'max_nodes', 'max_edges', 'softmax_nodes', 'softmax_edges',
           'broadcast_nodes', 'broadcast_edges', 'topk_nodes', 'topk_edges']

def readout_nodes(graph, feat, weight=None, *, op='sum', ntype=None):
    """Generate a graph-level representation by aggregating node features
    :attr:`feat`.

    The function is commonly used as a *readout* function on a batch of graphs
    to generate graph-level representation. Thus, the result tensor shape
    depends on the batch size of the input graph. Given a graph of batch size
    :math:`B`, and a feature size of :math:`D`, the result shape will be
    :math:`(B, D)`, with each row being the aggregated node features of each
    graph.

    Parameters
    ----------
    graph : DGLGraph.
        Input graph.
    feat : str
        Node feature name.
    weight : str, optional
        Node weight name. None means aggregating without weights.
        Otherwise, multiply each node feature by node feature :attr:`weight`
        before aggregation. The weight feature shape must be compatible with
        an element-wise multiplication with the feature tensor.
    op : str, optional
        Readout operator. Can be 'sum', 'max', 'min', 'mean'.
    ntype : str, optional
        Node type. Can be omitted if there is only one node type in the graph.

    Returns
    -------
    Tensor
        Result tensor.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.graph(([0, 1], [1, 0]))              # Graph 1
    >>> g1.ndata['h'] = th.tensor([1., 2.])
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> g2.ndata['h'] = th.tensor([1., 2., 3.])

    Sum over one graph:

    >>> dgl.readout_nodes(g1, 'h')
    tensor([3.])  # 1 + 2

    Sum over a batched graph:

    >>> bg = dgl.batch([g1, g2])
    >>> dgl.readout_nodes(bg, 'h')
    tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

    Weighted sum:

    >>> bg.ndata['w'] = th.tensor([.1, .2, .1, .5, .2])
    >>> dgl.readout_nodes(bg, 'h', 'w')
    tensor([.5, 1.7])

    Readout by max:

    >>> dgl.readout_nodes(bg, 'h', op='max')
    tensor([2., 3.])

    See Also
    --------
    readout_edges
    """
    x = graph.nodes[ntype].data[feat]
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    return segment.segment_reduce(graph.batch_num_nodes(ntype), x, reducer=op)

def readout_edges(graph, feat, weight=None, *, op='sum', etype=None):
    """Sum the edge feature :attr:`feat` in :attr:`graph`, optionally
    multiplies it by a edge :attr:`weight`.

    The function is commonly used as a *readout* function on a batch of graphs
    to generate graph-level representation. Thus, the result tensor shape
    depends on the batch size of the input graph. Given a graph of batch size
    :math:`B`, and a feature size of :math:`D`, the result shape will be
    :math:`(B, D)`, with each row being the aggregated edge features of each
    graph.

    Parameters
    ----------
    graph : DGLGraph.
        The input graph.
    feat : str
        The edge feature name.
    weight : str, optional
        The edge weight feature name. If None, no weighting will be performed,
        otherwise, weight each edge feature with field :attr:`feat`.
        for summation. The weight feature shape must be compatible with
        an element-wise multiplication with the feature tensor.
    op : str, optional
        Readout operator. Can be 'sum', 'max', 'min', 'mean'.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Returns
    -------
    Tensor
        Result tensor.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.graph(([0, 1], [1, 0]))              # Graph 1
    >>> g1.edata['h'] = th.tensor([1., 2.])
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> g2.edata['h'] = th.tensor([2., 3.])

    Sum over one graph:

    >>> dgl.readout_edges(g1, 'h')
    tensor([3.])  # 1 + 2

    Sum over a batched graph:

    >>> bg = dgl.batch([g1, g2])
    >>> dgl.readout_edges(bg, 'h')
    tensor([3., 5.])  # [1 + 2, 2 + 3]

    Weighted sum:

    >>> bg.edata['w'] = th.tensor([.1, .2, .1, .5])
    >>> dgl.readout_edges(bg, 'h', 'w')
    tensor([.5, 1.7])

    Readout by max:

    >>> dgl.readout_edges(bg, 'w', op='max')
    tensor([2., 3.])

    See Also
    --------
    readout_nodes
    """
    x = graph.edges[etype].data[feat]
    if weight is not None:
        x = x * graph.edges[etype].data[weight]
    return segment.segment_reduce(graph.batch_num_edges(etype), x, reducer=op)

def sum_nodes(graph, feat, weight=None, *, ntype=None):
    """Syntax sugar for ``dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='sum')``.

    See Also
    --------
    readout_nodes
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='sum')

def sum_edges(graph, feat, weight=None, *, etype=None):
    """Syntax sugar for ``dgl.readout_edges(graph, feat, weight, etype=etype, op='sum')``.

    See Also
    --------
    readout_edges
    """
    return readout_edges(graph, feat, weight, etype=etype, op='sum')

def mean_nodes(graph, feat, weight=None, *, ntype=None):
    """Syntax sugar for ``dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='mean')``.

    See Also
    --------
    readout_nodes
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='mean')

def mean_edges(graph, feat, weight=None, *, etype=None):
    """Syntax sugar for ``dgl.readout_edges(graph, feat, weight, etype=etype, op='mean')``.

    See Also
    --------
    readout_edges
    """
    return readout_edges(graph, feat, weight, etype=etype, op='mean')

def max_nodes(graph, feat, weight=None, *, ntype=None):
    """Syntax sugar for ``dgl.readout_nodes(graph, feat, weight, ntype=ntype, op='max')``.

    See Also
    --------
    readout_nodes
    """
    return readout_nodes(graph, feat, weight, ntype=ntype, op='max')

def max_edges(graph, feat, weight=None, *, etype=None):
    """Syntax sugar for ``dgl.readout_edges(graph, feat, weight, etype=etype, op='max')``.

    See Also
    --------
    readout_edges
    """
    return readout_edges(graph, feat, weight, etype=etype, op='max')

def softmax_nodes(graph, feat, *, ntype=None):
    r"""Perform graph-wise softmax on the node features.

    For each node :math:`v\in\mathcal{V}` and its feature :math:`x_v`,
    calculate its normalized feature as follows:

    .. math::
        z_v = \frac{\exp(x_v)}{\sum_{u\in\mathcal{V}}\exp(x_u)}

    If the graph is a batch of multiple graphs, each graph computes softmax
    independently. The result tensor has the same shape as the original node
    feature.

    Parameters
    ----------
    graph : DGLGraph.
        The input graph.
    feat : str
        The node feature name.
    ntype : str, optional
        The node type name. Can be omitted if there is only one node type in the graph.

    Returns
    -------
    Tensor
        Result tensor.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.graph(([0, 1], [1, 0]))              # Graph 1
    >>> g1.ndata['h'] = th.tensor([1., 1.])
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> g2.ndata['h'] = th.tensor([1., 1., 1.])

    Softmax over one graph:

    >>> dgl.softmax_nodes(g1, 'h')
    tensor([.5000, .5000])

    Softmax over a batched graph:

    >>> bg = dgl.batch([g1, g2])
    >>> dgl.softmax_nodes(bg, 'h')
    tensor([.5000, .5000, .3333, .3333, .3333])

    See Also
    --------
    softmax_edges
    """
    x = graph.nodes[ntype].data[feat]
    return segment.segment_softmax(graph.batch_num_nodes(ntype), x)

def softmax_edges(graph, feat, *, etype=None):
    r"""Perform graph-wise softmax on the edge features.

    For each edge :math:`e\in\mathcal{E}` and its feature :math:`x_e`,
    calculate its normalized feature as follows:

    .. math::
        z_e = \frac{\exp(x_e)}{\sum_{e'\in\mathcal{E}}\exp(x_{e'})}

    If the graph is a batch of multiple graphs, each graph computes softmax
    independently. The result tensor has the same shape as the original edge
    feature.

    Parameters
    ----------
    graph : DGLGraph.
        The input graph.
    feat : str
        The edge feature name.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Returns
    -------
    Tensor
        Result tensor.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.graph(([0, 1], [1, 0]))              # Graph 1
    >>> g1.edata['h'] = th.tensor([1., 1.])
    >>> g2 = dgl.graph(([0, 1, 0], [1, 2, 2]))        # Graph 2
    >>> g2.edata['h'] = th.tensor([1., 1., 1.])

    Softmax over one graph:

    >>> dgl.softmax_edges(g1, 'h')
    tensor([.5000, .5000])

    Softmax over a batched graph:

    >>> bg = dgl.batch([g1, g2])
    >>> dgl.softmax_edges(bg, 'h')
    tensor([.5000, .5000, .3333, .3333, .3333])

    See Also
    --------
    softmax_nodes
    """
    x = graph.edges[etype].data[feat]
    return segment.segment_softmax(graph.batch_num_edges(etype), x)

def broadcast_nodes(graph, graph_feat, *, ntype=None):
    """Generate a node feature equal to the graph-level feature :attr:`graph_feat`.

    The operation is similar to ``numpy.repeat`` (or ``torch.repeat_interleave``).
    It is commonly used to normalize node features by a global vector. For example,
    to normalize node features across graph to range :math:`[0~1)`:

    >>> g = dgl.batch([...])  # batch multiple graphs
    >>> g.ndata['h'] = ...  # some node features
    >>> h_sum = dgl.broadcast_nodes(g, dgl.sum_nodes(g, 'h'))
    >>> g.ndata['h'] /= h_sum  # normalize by summation

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    graph_feat : tensor
        The feature to broadcast. Tensor shape is :math:`(*)` for single graph, and
        :math:`(B, *)` for batched graph.
    ntype : str, optional
        Node type. Can be omitted if there is only one node type.

    Returns
    -------
    Tensor
        The node features tensor with shape :math:`(N, *)`, where :math:`N` is the
        number of nodes.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.graph(([0], [1]))                    # Graph 1
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> bg = dgl.batch([g1, g2])
    >>> feat = th.rand(2, 5)
    >>> feat
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all nodes in the batched graph, feat[i] is broadcast to nodes
    in the i-th example in the batch.

    >>> dgl.broadcast_nodes(bg, feat)
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all nodes in the single graph.

    >>> dgl.broadcast_nodes(g1, feat[0])
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368]])

    See Also
    --------
    broadcast_edges
    """
    return F.repeat(graph_feat, graph.batch_num_nodes(ntype), dim=0)

def broadcast_edges(graph, graph_feat, *, etype=None):
    """Generate an edge feature equal to the graph-level feature :attr:`graph_feat`.

    The operation is similar to ``numpy.repeat`` (or ``torch.repeat_interleave``).
    It is commonly used to normalize edge features by a global vector. For example,
    to normalize edge features across graph to range :math:`[0~1)`:

    >>> g = dgl.batch([...])  # batch multiple graphs
    >>> g.edata['h'] = ...  # some node features
    >>> h_sum = dgl.broadcast_edges(g, dgl.sum_edges(g, 'h'))
    >>> g.edata['h'] /= h_sum  # normalize by summation

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    graph_feat : tensor
        The feature to broadcast. Tensor shape is :math:`(*)` for single graph, and
        :math:`(B, *)` for batched graph.
    etype : str, typle of str, optional
        Edge type. Can be omitted if there is only one edge type in the graph.

    Returns
    -------
    Tensor
        The edge features tensor with shape :math:`(M, *)`, where :math:`M` is the
        number of edges.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.graph(([0], [1]))                    # Graph 1
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> bg = dgl.batch([g1, g2])
    >>> feat = th.rand(2, 5)
    >>> feat
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all edges in the batched graph, feat[i] is broadcast to edges
    in the i-th example in the batch.

    >>> dgl.broadcast_edges(bg, feat)
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all edges in the single graph.

    >>> dgl.broadcast_edges(g2, feat[1])
    tensor([[0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    See Also
    --------
    broadcast_nodes
    """
    return F.repeat(graph_feat, graph.batch_num_edges(etype), dim=0)

READOUT_ON_ATTRS = {
    'nodes': ('ndata', 'batch_num_nodes', 'number_of_nodes'),
    'edges': ('edata', 'batch_num_edges', 'number_of_edges'),
}

def _topk_on(graph, typestr, feat, k, descending, sortby, ntype_or_etype):
    """Internal function to take graph-wise top-k node/edge features of
    field :attr:`feat` in :attr:`graph` ranked by keys at given
    index :attr:`sortby`. If :attr:`descending` is set to False, return the
    k smallest elements instead.

    Parameters
    ---------
    graph : DGLGraph
        The graph
    typestr : str
        'nodes' or 'edges'
    feat : str
        The feature field name.
    k : int
        The :math:`k` in "top-:math`k`".
    descending : bool
        Controls whether to return the largest or smallest elements,
         defaults to True.
    sortby : int
        The key index we sort :attr:`feat` on, if set to None, we sort
        the whole :attr:`feat`.
    ntype_or_etype : str, tuple of str
        Node/edge type.

    Returns
    -------
    sorted_feat : Tensor
        A tensor with shape :math:`(B, K, D)`, where
        :math:`B` is the batch size of the input graph.
    sorted_idx : Tensor
        A tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if sortby
        is set to None), where
        :math:`B` is the batch size of the input graph, :math:`D`
        is the feature size.


    Notes
    -----
    If an example has :math:`n` nodes/edges and :math:`n<k`, in the first
    returned tensor the :math:`n+1` to :math:`k`th rows would be padded
    with all zero; in the second returned tensor, the behavior of :math:`n+1`
    to :math:`k`th elements is not defined.
    """
    _, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, typestr)[ntype_or_etype].data
    if F.ndim(data[feat]) > 2:
        raise DGLError('Only support {} feature `{}` with dimension less than or'
                       ' equal to 2'.format(typestr, feat))

    feat = data[feat]
    hidden_size = F.shape(feat)[-1]
    batch_num_objs = getattr(graph, batch_num_objs_attr)(ntype_or_etype)
    batch_size = len(batch_num_objs)

    length = max(max(F.asnumpy(batch_num_objs)), k)
    fill_val = -float('inf') if descending else float('inf')
    feat_ = F.pad_packed_tensor(feat, batch_num_objs, fill_val, l_min=k)

    if sortby is not None:
        keys = F.squeeze(F.slice_axis(feat_, -1, sortby, sortby+1), -1)
        order = F.argsort(keys, -1, descending=descending)
    else:
        order = F.argsort(feat_, 1, descending=descending)

    topk_indices = F.slice_axis(order, 1, 0, k)

    # zero padding
    feat_ = F.pad_packed_tensor(feat, batch_num_objs, 0, l_min=k)

    if sortby is not None:
        feat_ = F.reshape(feat_, (batch_size * length, -1))
        shift = F.repeat(F.arange(0, batch_size) * length, k, -1)
        shift = F.copy_to(shift, F.context(feat))
        topk_indices_ = F.reshape(topk_indices, (-1,)) + shift
    else:
        feat_ = F.reshape(feat_, (-1,))
        shift = F.repeat(F.arange(0, batch_size), k * hidden_size, -1) * length * hidden_size +\
                F.cat([F.arange(0, hidden_size)] * batch_size * k, -1)
        shift = F.copy_to(shift, F.context(feat))
        topk_indices_ = F.reshape(topk_indices, (-1,)) * hidden_size + shift

    return F.reshape(F.gather_row(feat_, topk_indices_), (batch_size, k, -1)),\
           topk_indices

def topk_nodes(graph, feat, k, *, descending=True, sortby=None, ntype=None):
    """Return a graph-level representation by a graph-wise top-k on
    node features :attr:`feat` in :attr:`graph` by feature at index :attr:`sortby`.

    If :attr:`descending` is set to False, return the k smallest elements instead.

    If :attr:`sortby` is set to None, the function would perform top-k on
    all dimensions independently, equivalent to calling
    :code:`torch.topk(graph.ndata[feat], dim=0)`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    k : int
        The k in "top-k"
    descending : bool
        Controls whether to return the largest or smallest elements.
    sortby : int, optional
        Sort according to which feature. If is None, all features are sorted independently.
    ntype : str, optional
        Node type. Can be omitted if there is only one node type in the graph.

    Returns
    -------
    sorted_feat : Tensor
        A tensor with shape :math:`(B, K, D)`, where
        :math:`B` is the batch size of the input graph.
    sorted_idx : Tensor
        A tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if sortby
        is set to None), where
        :math:`B` is the batch size of the input graph, :math:`D`
        is the feature size.

    Notes
    -----
    If an example has :math:`n` nodes and :math:`n<k`, the ``sorted_feat``
    tensor will pad the :math:`n+1` to :math:`k` th rows with zero;

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.graph(([0, 1], [2, 3]))              # Graph 1
    >>> g1.ndata['h'] = th.rand(4, 5)
    >>> g1.ndata['h']
    tensor([[0.0297, 0.8307, 0.9140, 0.6702, 0.3346],
            [0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
            [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
            [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]])

    >>> g2 = dgl.graph(([0, 1, 2], [2, 3, 4]))       # Graph 2
    >>> g2.ndata['h'] = th.rand(5, 5)
    >>> g2.ndata['h']
    tensor([[0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
            [0.1323, 0.2766, 0.4318, 0.6114, 0.1458],
            [0.1752, 0.9105, 0.5692, 0.8489, 0.0539],
            [0.1931, 0.4954, 0.3455, 0.3934, 0.0857],
            [0.5065, 0.5182, 0.5418, 0.1520, 0.3872]])

    Top-k over node attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], ndata=['h'])
    >>> dgl.topk_nodes(bg, 'h', 3)
    (tensor([[[0.5901, 0.8307, 0.9280, 0.8954, 0.7997],
              [0.5171, 0.6515, 0.9140, 0.7507, 0.5297],
              [0.0880, 0.6379, 0.4451, 0.6893, 0.5197]],
             [[0.5065, 0.9105, 0.5692, 0.8489, 0.3872],
              [0.3168, 0.5182, 0.5418, 0.6114, 0.3808],
              [0.1931, 0.4954, 0.5303, 0.3934, 0.1458]]]), tensor([[[1, 0, 1, 3, 1],
              [3, 2, 0, 2, 2],
              [2, 3, 2, 1, 3]],
             [[4, 2, 2, 2, 4],
              [0, 4, 4, 1, 0],
              [3, 3, 0, 3, 1]]]))

    Top-k over node attribute :attr:`h` along the last dimension in a batched graph.
    (used in SortPooling)

    >>> dgl.topk_nodes(bg, 'h', 3, sortby=-1)
    (tensor([[[0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
              [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
              [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]],
             [[0.5065, 0.5182, 0.5418, 0.1520, 0.3872],
              [0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
              [0.1323, 0.2766, 0.4318, 0.6114, 0.1458]]]), tensor([[1, 2, 3],
             [4, 0, 1]]))

    Top-k over node attribute :attr:`h` in a single graph.

    >>> dgl.topk_nodes(g1, 'h', 3)
    (tensor([[[0.5901, 0.8307, 0.9280, 0.8954, 0.7997],
              [0.5171, 0.6515, 0.9140, 0.7507, 0.5297],
              [0.0880, 0.6379, 0.4451, 0.6893, 0.5197]]]), tensor([[[1, 0, 1, 3, 1],
              [3, 2, 0, 2, 2],
              [2, 3, 2, 1, 3]]]))
    """
    return _topk_on(graph, 'nodes', feat, k,
                    descending=descending, sortby=sortby,
                    ntype_or_etype=ntype)

def topk_edges(graph, feat, k, *, descending=True, sortby=None, etype=None):
    """Return a graph-level representation by a graph-wise top-k
    on edge features :attr:`feat` in :attr:`graph` by feature at index :attr:`sortby`.

    If :attr:`descending` is set to False, return the k smallest elements instead.

    If :attr:`sortby` is set to None, the function would perform top-k on
    all dimensions independently, equivalent to calling
    :code:`torch.topk(graph.edata[feat], dim=0)`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    k : int
        The k in "top-k"
    descending : bool
        Controls whether to return the largest or smallest elements.
    sortby : int, optional
        Sort according to which feature. If is None, all features are sorted independently.
    etype : str, typle of str, optional
        Edge type. Can be omitted if there is only one edge type in the graph.

    Returns
    -------
    sorted_feat : Tensor
        A tensor with shape :math:`(B, K, D)`, where
        :math:`B` is the batch size of the input graph.
    sorted_idx : Tensor
        A tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if sortby
        is set to None), where
        :math:`B` is the batch size of the input graph, :math:`D`
        is the feature size.


    Notes
    -----
    If an example has :math:`n` nodes and :math:`n<k`, the ``sorted_feat``
    tensor will pad the :math:`n+1` to :math:`k` th rows with zero;
    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 0]))         # Graph 1
    >>> g1.edata['h'] = th.rand(4, 5)
    >>> g1.edata['h']
    tensor([[0.0297, 0.8307, 0.9140, 0.6702, 0.3346],
            [0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
            [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
            [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]])

    >>> g2 = dgl.graph(([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]))   # Graph 2
    >>> g2.edata['h'] = th.rand(5, 5)
    >>> g2.edata['h']
    tensor([[0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
            [0.1323, 0.2766, 0.4318, 0.6114, 0.1458],
            [0.1752, 0.9105, 0.5692, 0.8489, 0.0539],
            [0.1931, 0.4954, 0.3455, 0.3934, 0.0857],
            [0.5065, 0.5182, 0.5418, 0.1520, 0.3872]])

    Top-k over edge attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], edata=['h'])
    >>> dgl.topk_edges(bg, 'h', 3)
    (tensor([[[0.5901, 0.8307, 0.9280, 0.8954, 0.7997],
              [0.5171, 0.6515, 0.9140, 0.7507, 0.5297],
              [0.0880, 0.6379, 0.4451, 0.6893, 0.5197]],
             [[0.5065, 0.9105, 0.5692, 0.8489, 0.3872],
              [0.3168, 0.5182, 0.5418, 0.6114, 0.3808],
              [0.1931, 0.4954, 0.5303, 0.3934, 0.1458]]]), tensor([[[1, 0, 1, 3, 1],
              [3, 2, 0, 2, 2],
              [2, 3, 2, 1, 3]],
             [[4, 2, 2, 2, 4],
              [0, 4, 4, 1, 0],
              [3, 3, 0, 3, 1]]]))

    Top-k over edge attribute :attr:`h` along index -1 in a batched graph.
    (used in SortPooling)

    >>> dgl.topk_edges(bg, 'h', 3, sortby=-1)
    (tensor([[[0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
              [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
              [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]],
             [[0.5065, 0.5182, 0.5418, 0.1520, 0.3872],
              [0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
              [0.1323, 0.2766, 0.4318, 0.6114, 0.1458]]]), tensor([[1, 2, 3],
             [4, 0, 1]]))

    Top-k over edge attribute :attr:`h` in a single graph.

    >>> dgl.topk_edges(g1, 'h', 3)
    (tensor([[[0.5901, 0.8307, 0.9280, 0.8954, 0.7997],
              [0.5171, 0.6515, 0.9140, 0.7507, 0.5297],
              [0.0880, 0.6379, 0.4451, 0.6893, 0.5197]]]), tensor([[[1, 0, 1, 3, 1],
              [3, 2, 0, 2, 2],
              [2, 3, 2, 1, 3]]]))
    """
    return _topk_on(graph, 'edges', feat, k,
                    descending=descending, sortby=sortby,
                    ntype_or_etype=etype)
