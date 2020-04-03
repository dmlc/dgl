"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

from .base import DGLError
from . import backend as F

__all__ = ['sum_nodes', 'sum_edges', 'mean_nodes', 'mean_edges',
           'max_nodes', 'max_edges', 'softmax_nodes', 'softmax_edges',
           'broadcast_nodes', 'broadcast_edges', 'topk_nodes', 'topk_edges']

READOUT_ON_ATTRS = {
    'nodes': ('ndata', 'batch_num_nodes', 'number_of_nodes'),
    'edges': ('edata', 'batch_num_edges', 'number_of_edges'),
}

def _sum_on(graph, typestr, feat, weight):
    """Internal function to sum node or edge features.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    typestr : str
        'nodes' or 'edges'
    feat : str
        The feature field name.
    weight : str
        The weight field name.

    Returns
    -------
    tensor
        The (weighted) summed node or edge features.
    """
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    feat = data[feat]

    if weight is not None:
        weight = data[weight]
        weight = F.reshape(weight, (-1,) + (1,) * (F.ndim(feat) - 1))
        feat = weight * feat

    n_graphs = graph.batch_size
    batch_num_objs = getattr(graph, batch_num_objs_attr)
    seg_id = F.zerocopy_from_numpy(np.arange(n_graphs, dtype='int64').repeat(batch_num_objs))
    seg_id = F.copy_to(seg_id, F.context(feat))
    y = F.unsorted_1d_segment_sum(feat, seg_id, n_graphs, 0)
    return y

def sum_nodes(graph, feat, weight=None):
    """Sums all the values of node field :attr:`feat` in :attr:`graph`, optionally
    multiplies the field by a scalar node field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph.
        The graph.
    feat : str
        The feature field.
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each node feature with field :attr:`feat`.
        for summation. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_nodes(), 1]``.

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of the
    i-th graph in the batched graph. If a graph has no nodes,
    a zero tensor with the same shape is returned at the corresponding row.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.ndata['h'] = th.tensor([[1.], [2.]])
    >>> g1.ndata['w'] = th.tensor([[3.], [6.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.ndata['h'] = th.tensor([[1.], [2.], [3.]])

    Sum over node attribute :attr:`h` without weighting for each graph in a
    batched graph.

    >>> bg = dgl.batch([g1, g2], node_attrs='h')
    >>> dgl.sum_nodes(bg, 'h')
    tensor([[3.],   # 1 + 2
            [6.]])  # 1 + 2 + 3

    Sum node attribute :attr:`h` with weight from node attribute :attr:`w`
    for a single graph.

    >>> dgl.sum_nodes(g1, 'h', 'w')
    tensor([[15.]]) # 1 * 3 + 2 * 6

    See Also
    --------
    mean_nodes
    sum_edges
    mean_edges
    """
    return _sum_on(graph, 'nodes', feat, weight)

def sum_edges(graph, feat, weight=None):
    """Sums all the values of edge field :attr:`feat` in :attr:`graph`,
    optionally multiplies the field by a scalar edge field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each edge feature with field :attr:`feat`.
        for summation. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_edges(), 1]``.

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of the
    i-th graph in the batched graph. If a graph has no edges,
    a zero tensor with the same shape is returned at the corresponding row.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.add_edges([0, 1], [1, 0])
    >>> g1.edata['h'] = th.tensor([[1.], [2.]])
    >>> g1.edata['w'] = th.tensor([[3.], [6.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 0])
    >>> g2.edata['h'] = th.tensor([[1.], [2.], [3.]])

    Sum over edge attribute :attr:`h` without weighting for each graph in a
    batched graph.

    >>> bg = dgl.batch([g1, g2], edge_attrs='h')
    >>> dgl.sum_edges(bg, 'h')
    tensor([[3.],   # 1 + 2
            [6.]])  # 1 + 2 + 3

    Sum edge attribute :attr:`h` with weight from edge attribute :attr:`w`
    for a single graph.

    >>> dgl.sum_edges(g1, 'h', 'w')
    tensor([[15.]]) # 1 * 3 + 2 * 6

    See Also
    --------
    sum_nodes
    mean_nodes
    mean_edges
    """
    return _sum_on(graph, 'edges', feat, weight)


def _mean_on(graph, typestr, feat, weight):
    """Internal function to sum node or edge features.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    typestr : str
        'nodes' or 'edges'
    feat : str
        The feature field name.
    weight : str
        The weight field name.

    Returns
    -------
    tensor
        The (weighted) summed node or edge features.
    """
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    feat = data[feat]

    if weight is not None:
        weight = data[weight]
        weight = F.reshape(weight, (-1,) + (1,) * (F.ndim(feat) - 1))
        feat = weight * feat

    n_graphs = graph.batch_size
    batch_num_objs = getattr(graph, batch_num_objs_attr)
    seg_id = F.zerocopy_from_numpy(np.arange(n_graphs, dtype='int64').repeat(batch_num_objs))
    seg_id = F.copy_to(seg_id, F.context(feat))
    if weight is not None:
        w = F.unsorted_1d_segment_sum(weight, seg_id, n_graphs, 0)
        y = F.unsorted_1d_segment_sum(feat, seg_id, n_graphs, 0)
        y = y / w
    else:
        y = F.unsorted_1d_segment_mean(feat, seg_id, n_graphs, 0)
    return y

def mean_nodes(graph, feat, weight=None):
    """Averages all the values of node field :attr:`feat` in :attr:`graph`,
    optionally multiplies the field by a scalar node field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each node feature with field :attr:`feat`.
        for calculating mean. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_nodes(), 1]``.

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of
    the i-th graph in the batch. If a graph has no nodes,
    a zero tensor with the same shape is returned at the corresponding row.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.ndata['h'] = th.tensor([[1.], [2.]])
    >>> g1.ndata['w'] = th.tensor([[3.], [6.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.ndata['h'] = th.tensor([[1.], [2.], [3.]])

    Average over node attribute :attr:`h` without weighting for each graph in a
    batched graph.

    >>> bg = dgl.batch([g1, g2], node_attrs='h')
    >>> dgl.mean_nodes(bg, 'h')
    tensor([[1.5000],    # (1 + 2) / 2
            [2.0000]])   # (1 + 2 + 3) / 3

    Sum node attribute :attr:`h` with normalized weight from node attribute :attr:`w`
    for a single graph.

    >>> dgl.mean_nodes(g1, 'h', 'w') # h1 * (w1 / (w1 + w2)) + h2 * (w2 / (w1 + w2))
    tensor([[1.6667]])               # 1 * (3 / (3 + 6)) + 2 * (6 / (3 + 6))

    See Also
    --------
    sum_nodes
    sum_edges
    mean_edges
    """
    return _mean_on(graph, 'nodes', feat, weight)

def mean_edges(graph, feat, weight=None):
    """Averages all the values of edge field :attr:`feat` in :attr:`graph`,
    optionally multiplies the field by a scalar edge field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.
    weight : optional, str
        The weight field. If None, no weighting will be performed,
        otherwise, weight each edge feature with field :attr:`feat`.
        for calculating mean. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_edges(), 1]``.

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of
    the i-th graph in the batched graph. If a graph has no edges,
    a zero tensor with the same shape is returned at the corresponding row.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.add_edges([0, 1], [1, 0])
    >>> g1.edata['h'] = th.tensor([[1.], [2.]])
    >>> g1.edata['w'] = th.tensor([[3.], [6.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 0])
    >>> g2.edata['h'] = th.tensor([[1.], [2.], [3.]])

    Average over edge attribute :attr:`h` without weighting for each graph in a
    batched graph.

    >>> bg = dgl.batch([g1, g2], edge_attrs='h')
    >>> dgl.mean_edges(bg, 'h')
    tensor([[1.5000],    # (1 + 2) / 2
            [2.0000]])   # (1 + 2 + 3) / 3

    Sum edge attribute :attr:`h` with normalized weight from edge attribute :attr:`w`
    for a single graph.

    >>> dgl.mean_edges(g1, 'h', 'w') # h1 * (w1 / (w1 + w2)) + h2 * (w2 / (w1 + w2))
    tensor([[1.6667]])               # 1 * (3 / (3 + 6)) + 2 * (6 / (3 + 6))

    See Also
    --------
    sum_nodes
    mean_nodes
    sum_edges
    """
    return _mean_on(graph, 'edges', feat, weight)

def _max_on(graph, typestr, feat):
    """Internal function to take elementwise maximum
     over node or edge features.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    typestr : str
        'nodes' or 'edges'
    feat : str
        The feature field name.

    Returns
    -------
    tensor
        The (weighted) summed node or edge features.
    """
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    feat = data[feat]

    # TODO: the current solution pads the different graph sizes to the same,
    #  a more efficient way is to use segment max, we need to implement it in
    #  the future.
    batch_num_objs = getattr(graph, batch_num_objs_attr)
    feat = F.pad_packed_tensor(feat, batch_num_objs, -float('inf'))
    return F.max(feat, 1)


def _softmax_on(graph, typestr, feat):
    """Internal function of applying batch-wise graph-level softmax
    over node or edge features of a given field.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    typestr : str
        'nodes' or 'edges'
    feat : str
        The feature field name.

    Returns
    -------
    tensor
        The obtained tensor.
    """
    data_attr, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]
    data = getattr(graph, data_attr)
    feat = data[feat]

    # TODO: the current solution pads the different graph sizes to the same,
    #  a more efficient way is to use segment sum/max, we need to implement
    #  it in the future.
    batch_num_objs = getattr(graph, batch_num_objs_attr)
    feat = F.pad_packed_tensor(feat, batch_num_objs, -float('inf'))
    feat = F.softmax(feat, 1)
    return F.pack_padded_tensor(feat, batch_num_objs)

def _broadcast_on(graph, typestr, feat_data):
    """Internal function of broadcasting features to all nodes/edges.

    Parameters
    ----------
    graph : DGLGraph
        The graph
    typestr : str
        'nodes' or 'edges'
    feat_data : tensor
        The feature to broadcast. Tensor shape is :math:`(*)` for single graph,
        and :math:`(B, *)` for batched graph.

    Returns
    -------
    tensor
        The node/edge features tensor with shape :math:`(N, *)`.
    """
    _, batch_num_objs_attr, _ = READOUT_ON_ATTRS[typestr]

    batch_num_objs = getattr(graph, batch_num_objs_attr)
    index = []
    for i, num_obj in enumerate(batch_num_objs):
        index.extend([i] * num_obj)
    ctx = F.context(feat_data)
    index = F.copy_to(F.tensor(index), ctx)
    return F.gather_row(feat_data, index)

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


def max_nodes(graph, feat):
    """Take elementwise maximum over all the values of node field
    :attr:`feat` in :attr:`graph`

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.

    Returns
    -------
    tensor
        The tensor obtained.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.ndata['h'] = th.tensor([[1.], [2.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.ndata['h'] = th.tensor([[1.], [2.], [3.]])

    Max over node attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], node_attrs='h')
    >>> dgl.max_nodes(bg, 'h')
    tensor([[2.],    # max(1, 2)
            [3.]])   # max(1, 2, 3)

    Max over node attribute :attr:`h` in a single graph.

    >>> dgl.max_nodes(g1, 'h')
    tensor([[2.]])

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of
    the i-th graph in the batched graph. If a graph has no nodes,
    a tensor filed with -inf of the same shape is returned at the
    corresponding row.
    """
    return _max_on(graph, 'nodes', feat)

def max_edges(graph, feat):
    """Take elementwise maximum over all the values of edge field
    :attr:`feat` in :attr:`graph`

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.

    Returns
    -------
    tensor
        The tensor obtained.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.add_edges([0, 1], [1, 0])
    >>> g1.edata['h'] = th.tensor([[1.], [2.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 0])
    >>> g2.edata['h'] = th.tensor([[1.], [2.], [3.]])

    Max over edge attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], edge_attrs='h')
    >>> dgl.max_edges(bg, 'h')
    tensor([[2.],    # max(1, 2)
            [3.]])   # max(1, 2, 3)

    Max over edge attribute :attr:`h` in a single graph.

    >>> dgl.max_edges(g1, 'h')
    tensor([[2.]])

    Notes
    -----
    Return a stacked tensor with an extra first dimension whose size equals
    batch size of the input graph.
    The i-th row of the stacked tensor contains the readout result of
    the i-th graph in the batched graph. If a graph has no edges,
    a tensor filled with -inf of the same shape is returned at the
    corresponding row.
    """
    return _max_on(graph, 'edges', feat)

def softmax_nodes(graph, feat):
    """Apply batch-wise graph-level softmax over all the values of node field
    :attr:`feat` in :attr:`graph`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.

    Returns
    -------
    tensor
        The tensor obtained.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.ndata['h'] = th.tensor([[1., 0.], [2., 0.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.ndata['h'] = th.tensor([[1., 0.], [2., 0.], [3., 0.]])

    Softmax over node attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], node_attrs='h')
    >>> dgl.softmax_nodes(bg, 'h')
    tensor([[0.2689, 0.5000], # [0.2689, 0.7311] = softmax([1., 2.])
            [0.7311, 0.5000], # [0.5000, 0.5000] = softmax([0., 0.])
            [0.0900, 0.3333], # [0.0900, 0.2447, 0.6652] = softmax([1., 2., 3.])
            [0.2447, 0.3333], # [0.3333, 0.3333, 0.3333] = softmax([0., 0., 0.])
            [0.6652, 0.3333]])

    Softmax over node attribute :attr:`h` in a single graph.

    >>> dgl.softmax_nodes(g1, 'h')
    tensor([[0.2689, 0.5000],   # [0.2689, 0.7311] = softmax([1., 2.])
            [0.7311, 0.5000]]), # [0.5000, 0.5000] = softmax([0., 0.])

    Notes
    -----
    If the input graph has batch size greater then one, the softmax is applied at
    each single graph in the batched graph.
    """
    return _softmax_on(graph, 'nodes', feat)


def softmax_edges(graph, feat):
    """Apply batch-wise graph-level softmax over all the values of edge field
    :attr:`feat` in :attr:`graph`.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat : str
        The feature field.

    Returns
    -------
    tensor
        The tensor obtained.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.add_edges([0, 1], [1, 0])
    >>> g1.edata['h'] = th.tensor([[1., 0.], [2., 0.]])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 0])
    >>> g2.edata['h'] = th.tensor([[1., 0.], [2., 0.], [3., 0.]])

    Softmax over edge attribute :attr:`h` in a batched graph.

    >>> bg = dgl.batch([g1, g2], edge_attrs='h')
    >>> dgl.softmax_edges(bg, 'h')
    tensor([[0.2689, 0.5000], # [0.2689, 0.7311] = softmax([1., 2.])
            [0.7311, 0.5000], # [0.5000, 0.5000] = softmax([0., 0.])
            [0.0900, 0.3333], # [0.0900, 0.2447, 0.6652] = softmax([1., 2., 3.])
            [0.2447, 0.3333], # [0.3333, 0.3333, 0.3333] = softmax([0., 0., 0.])
            [0.6652, 0.3333]])

    Softmax over edge attribute :attr:`h` in a single graph.

    >>> dgl.softmax_edges(g1, 'h')
    tensor([[0.2689, 0.5000],   # [0.2689, 0.7311] = softmax([1., 2.])
            [0.7311, 0.5000]]), # [0.5000, 0.5000] = softmax([0., 0.])

    Notes
    -----
    If the input graph has batch size greater then one, the softmax is applied at each
    example in the batch.
    """
    return _softmax_on(graph, 'edges', feat)

def broadcast_nodes(graph, feat_data):
    """Broadcast :attr:`feat_data` to all nodes in :attr:`graph`, and return a
    tensor of node features.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat_data : tensor
        The feature to broadcast. Tensor shape is :math:`(*)` for single graph, and
        :math:`(B, *)` for batched graph.

    Returns
    -------
    tensor
        The node features tensor with shape :math:`(N, *)`.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)

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

    Broadcast feature to all nodes in the batched graph.

    >>> dgl.broadcast_nodes(g1, feat[0])
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368]])

    Notes
    -----
    feat[i] is broadcast to the nodes in i-th graph in the batched graph.
    """
    return _broadcast_on(graph, 'nodes', feat_data)

def broadcast_edges(graph, feat_data):
    """Broadcast :attr:`feat_data` to all edges in :attr:`graph`, and return a
    tensor of edge features.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    feat_data : tensor
        The feature to broadcast. Tensor shape is :math:`(*)` for single
        graph, and :math:`(B, *)` for batched graph.

    Returns
    -------
    tensor
        The edge features tensor with shape :math:`(E, *)`

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    edge features.

    >>> g1 = dgl.DGLGraph()                           # Graph 1
    >>> g1.add_nodes(2)
    >>> g1.add_edges([0, 1], [1, 0])

    >>> g2 = dgl.DGLGraph()                           # Graph 2
    >>> g2.add_nodes(3)
    >>> g2.add_edges([0, 1, 2], [1, 2, 0])

    >>> bg = dgl.batch([g1, g2])
    >>> feat = th.rand(2, 5)
    >>> feat
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all edges in the batched graph, feat[i] is broadcast to edges
    in the i-th example in the batch.

    >>> dgl.broadcast_edges(bg, feat)
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all edges in the batched graph.

    >>> dgl.broadcast_edges(g1, feat[0])
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368]])

    Notes
    -----
    feat[i] is broadcast to the edges in i-th graph in the batched graph.
    """
    return _broadcast_on(graph, 'edges', feat_data)

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
