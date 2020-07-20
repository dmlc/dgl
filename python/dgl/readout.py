"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

from .base import DGLError
from . import backend as F
from . import segment

__all__ = ['sum_nodes', 'sum_edges', 'mean_nodes', 'mean_edges',
           'max_nodes', 'max_edges', 'softmax_nodes', 'softmax_edges',
           'broadcast_nodes', 'broadcast_edges', 'topk_nodes', 'topk_edges']

READOUT_ON_ATTRS = {
    'nodes': ('ndata', 'batch_num_nodes', 'number_of_nodes'),
    'edges': ('edata', 'batch_num_edges', 'number_of_edges'),
}

def sum_nodes(graph, feat, weight=None, ntype=None):
    """Sum the node feature :attr:`feat` in :attr:`graph`, optionally
    multiplies it by a node :attr:`weight`.

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
        Node weight name. If None, no weighting will be performed,
        otherwise, weight each node feature with field :attr:`feat`.
        for summation. The weight feature shape must be compatible with
        an element-wise multiplication with the feature tensor.
    ntype : str, optional
        Node type. Can be omitted if there is only one node type in the graph.

    Returns
    -------
    tensor
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

    >>> dgl.sum_nodes(g1, 'h')
    tensor([3.])  # 1 + 2

    Sum over a batched graph:

    >>> bg = dgl.batch([g1, g2])
    >>> dgl.sum_nodes(bg, 'h')
    tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

    Weighted sum:

    >>> bg.ndata['w'] = th.tensor([.1, .2, .1, .5, .2])
    >>> dgl.sum_nodes(bg, 'h', 'w')
    tensor([.5, 1.7])

    See Also
    --------
    sum_edges
    """
    x = graph.nodes[ntype].data[feat]
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    if ntype is None:
        return segment.segment_reduce(graph.batch_num_nodes, x)
    else:
        return segment.segment_reduce(graph.batch_num_nodes[ntype], x)

def sum_edges(graph, feat, weight=None, etype=None):
    """
    TBD
    """
    x = graph.edges[etype].data[feat]
    if weight is not None:
        x = x * graph.edges[etype].data[weight]
    if etype is None:
        return segment.segment_reduce(graph.batch_num_edges, x)
    else:
        etype = graph.to_canonical_etype(etype)
        return segment.segment_reduce(graph.batch_num_edges[etype], x)

def mean_nodes(graph, feat, weight=None, ntype=None):
    """
    TBD
    """
    x = graph.nodes[ntype].data[feat]
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    if ntype is None:
        return segment.segment_reduce(graph.batch_num_nodes, x, reducer='mean')
    else:
        return segment.segment_reduce(graph.batch_num_nodes[ntype], x, reducer='mean')

def mean_edges(graph, feat, weight=None, etype=None):
    """
    TBD
    """
    x = graph.edges[etype].data[feat]
    if weight is not None:
        x = x * graph.edges[etype].data[weight]
    if etype is None:
        return segment.segment_reduce(graph.batch_num_edges, x, reducer='mean')
    else:
        etype = graph.to_canonical_etype(etype)
        return segment.segment_reduce(graph.batch_num_edges[etype], x, reducer='mean')

def max_nodes(graph, feat, weight=None, ntype=None):
    """
    TBD
    """
    x = graph.nodes[ntype].data[feat]
    if weight is not None:
        x = x * graph.nodes[ntype].data[weight]
    if ntype is None:
        return segment.segment_reduce(graph.batch_num_nodes, x, reducer='max')
    else:
        return segment.segment_reduce(graph.batch_num_nodes[ntype], x, reducer='max')

def max_edges(graph, feat, weight=None, etype=None):
    """
    TBD
    """
    x = graph.edges[etype].data[feat]
    if weight is not None:
        x = x * graph.edges[etype].data[weight]
    if etype is None:
        return segment.segment_reduce(graph.batch_num_edges, x, reducer='max')
    else:
        etype = graph.to_canonical_etype(etype)
        return segment.segment_reduce(graph.batch_num_edges[etype], x, reducer='max')

def softmax_nodes(graph, feat, ntype=None):
    """
    TBD
    """
    x = graph.nodes[ntype].data[feat]
    if ntype is None:
        return segment.segment_softmax(graph.batch_num_nodes, x)
    else:
        return segment.segment_softmax(graph.batch_num_nodes[ntype], x)

def softmax_edges(graph, feat, etype=None):
    """
    TBD
    """
    x = graph.edges[etype].data[feat]
    if etype is None:
        return segment.segment_softmax(graph.batch_num_edges, x)
    else:
        etype = graph.to_canonical_etype(etype)
        return segment.segment_softmax(graph.batch_num_edges[etype], x)

def broadcast_nodes(graph, feat, ntype=None):
    """
    TBD
    """
    return F.repeat(feat, graph.batch_num_nodes, dim=0)

def broadcast_edges(graph, feat, etype=None):
    """
    TBD
    """
    return F.repeat(feat, graph.batch_num_edges, dim=0)

def _topk_on(graph, typestr, feat, k, descending=True, idx=None):
    """Internal function to take graph-wise top-k node/edge features of
    field :attr:`feat` in :attr:`graph` ranked by keys at given
    index :attr:`idx`. If :attr:`descending` is set to False, return the
    k smallest elements instead.

    If idx is set to None, the function would return top-k value of all
    indices, which is equivalent to calling `th.topk(graph.ndata[feat], dim=0)`
    for each single graph of the input batched-graph.

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
    idx : int or None, defaults to None
        The key index we sort :attr:`feat` on, if set to None, we sort
        the whole :attr:`feat`.

    Returns
    -------
    tuple of tensors:
        The first tensor returns top-k features of each single graph of
        the input graph:
        a tensor with shape :math:`(B, K, D)` would be returned, where
        :math:`B` is the batch size of the input graph.
        The second tensor returns the top-k indices of each single graph
        of the input graph:
        a tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if` idx
        is set to None) would be returned, where
        :math:`B` is the batch size of the input graph.

    Notes
    -----
    If an example has :math:`n` nodes/edges and :math:`n<k`, in the first
    returned tensor the :math:`n+1` to :math:`k`th rows would be padded
    with all zero; in the second returned tensor, the behavior of :math:`n+1`
    to :math:`k`th elements is not defined.
    """
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    if F.ndim(data[feat]) > 2:
        raise DGLError('The {} feature `{}` should have dimension less than or'
                       ' equal to 2'.format(typestr, feat))

    feat = data[feat]
    hidden_size = F.shape(feat)[-1]
    batch_num_objs = getattr(graph, batch_num_objs_attr)
    batch_size = len(batch_num_objs)

    length = max(max(batch_num_objs), k)
    fill_val = -float('inf') if descending else float('inf')
    feat_ = F.pad_packed_tensor(feat, batch_num_objs, fill_val, l_min=k)

    if idx is not None:
        keys = F.squeeze(F.slice_axis(feat_, -1, idx, idx+1), -1)
        order = F.argsort(keys, -1, descending=descending)
    else:
        order = F.argsort(feat_, 1, descending=descending)

    topk_indices = F.slice_axis(order, 1, 0, k)

    # zero padding
    feat_ = F.pad_packed_tensor(feat, batch_num_objs, 0, l_min=k)

    if idx is not None:
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

def topk_nodes(graph, feat, k, descending=True, idx=None):
    """Return graph-wise top-k node features of field :attr:`feat` in
    :attr:`graph` ranked by keys at given index :attr:`idx`. If :attr:
    `descending` is set to False, return the k smallest elements instead.

    If idx is set to None, the function would return top-k value of all
    indices, which is equivalent to calling
    :code:`torch.topk(graph.ndata[feat], dim=0)`
    for each example of the input graph.

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
    idx : int or None, defaults to None
        The index of keys we rank :attr:`feat` on, if set to None, we sort
        the whole :attr:`feat`.

    Returns
    -------
    tuple of tensors
        The first tensor returns top-k node features of each single graph of
        the input graph:
        a tensor with shape :math:`(B, K, D)` would be returned, where
        :math:`B` is the batch size of the input graph.
        The second tensor returns the top-k node indices of each single graph
        of the input graph:
        a tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if` idx
        is set to None) would be returned, where
        :math:`B` is the batch size of the input graph.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(4)
    >>> g1.ndata['h'] = th.rand(4, 5)
    >>> g1.ndata['h']
    tensor([[0.0297, 0.8307, 0.9140, 0.6702, 0.3346],
            [0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
            [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
            [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(5)
    >>> g2.ndata['h'] = th.rand(5, 5)
    >>> g2.ndata['h']
    tensor([[0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
            [0.1323, 0.2766, 0.4318, 0.6114, 0.1458],
            [0.1752, 0.9105, 0.5692, 0.8489, 0.0539],
            [0.1931, 0.4954, 0.3455, 0.3934, 0.0857],
            [0.5065, 0.5182, 0.5418, 0.1520, 0.3872]])

    Top-k over node attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], node_attrs='h')
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

    Top-k over node attribute :attr:`h` along index -1 in a batched graph.
    (used in SortPooling)

    >>> dgl.topk_nodes(bg, 'h', 3, idx=-1)
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

    Notes
    -----
    If an example has :math:`n` nodes and :math:`n<k`, in the first
    returned tensor the :math:`n+1` to :math:`k`th rows would be padded
    with all zero; in the second returned tensor, the behavior of :math:`n+1`
    to :math:`k`th elements is not defined.
    """
    return _topk_on(graph, 'nodes', feat, k, descending=descending, idx=idx)

def topk_edges(graph, feat, k, descending=True, idx=None):
    """Return graph-wise top-k edge features of field :attr:`feat` in
    :attr:`graph` ranked by keys at given index :attr:`idx`. If
    :attr:`descending` is set to False, return the k smallest elements
    instead.

    If idx is set to None, the function would return top-k value of all
    indices, which is equivalent to calling
    :code:`torch.topk(graph.edata[feat], dim=0)`
    for each example of the input graph.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    k : int
        The k in "top-k".
    descending : bool
        Controls whether to return the largest or smallest elements.
    idx : int or None, defaults to None
        The key index we sort :attr:`feat` on, if set to None, we sort
        the whole :attr:`feat`.

    Returns
    -------
    tuple of tensors
        The first tensor returns top-k edge features of each single graph of
        the input graph:
        a tensor with shape :math:`(B, K, D)` would be returned, where
        :math:`B` is the batch size of the input graph.
        The second tensor returns the top-k edge indices of each single graph
        of the input graph:
        a tensor with shape :math:`(B, K)`(:math:`(B, K, D)` if` idx
        is set to None) would be returned, where
        :math:`B` is the batch size of the input graph.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(4)
    >>> g1.add_edges([0, 1, 2, 3], [1, 2, 3, 0])
    >>> g1.edata['h'] = th.rand(4, 5)
    >>> g1.edata['h']
    tensor([[0.0297, 0.8307, 0.9140, 0.6702, 0.3346],
            [0.5901, 0.3030, 0.9280, 0.6893, 0.7997],
            [0.0880, 0.6515, 0.4451, 0.7507, 0.5297],
            [0.5171, 0.6379, 0.2695, 0.8954, 0.5197]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(5)
    >>> g2.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0])
    >>> g2.edata['h'] = th.rand(5, 5)
    >>> g2.edata['h']
    tensor([[0.3168, 0.3174, 0.5303, 0.0804, 0.3808],
            [0.1323, 0.2766, 0.4318, 0.6114, 0.1458],
            [0.1752, 0.9105, 0.5692, 0.8489, 0.0539],
            [0.1931, 0.4954, 0.3455, 0.3934, 0.0857],
            [0.5065, 0.5182, 0.5418, 0.1520, 0.3872]])

    Top-k over edge attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], edge_attrs='h')
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

    >>> dgl.topk_edges(bg, 'h', 3, idx=-1)
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

    Notes
    -----
    If an example has :math:`n` edges and :math:`n<k`, in the first
    returned tensor the :math:`n+1` to :math:`k`th rows would be padded
    with all zero; in the second returned tensor, the behavior of :math:`n+1`
    to :math:`k`th elements is not defined.
    """
    return _topk_on(graph, 'edges', feat, k, descending=descending, idx=idx)
