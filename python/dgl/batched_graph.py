"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np
from collections import Iterable

from .base import ALL, is_all
from .frame import FrameRef, Frame
from .graph import DGLGraph
from . import graph_index as gi
from . import backend as F
from . import utils

__all__ = ['BatchedDGLGraph', 'batch', 'unbatch', 'split',
           'sum_nodes', 'sum_edges', 'mean_nodes', 'mean_edges']

class BatchedDGLGraph(DGLGraph):
    """Class for batched DGL graphs.

    A :class:`BatchedDGLGraph` basically merges a list of small graphs into a giant
    graph so that one can perform message passing and readout over a batch of graphs
    simultaneously.

    The nodes and edges are re-indexed with a new id in the batched graph with the
    rule below:

    ======  ==========  ========================  ===  ==========================
    item    Graph 1     Graph 2                   ...  Graph k
    ======  ==========  ========================  ===  ==========================
    raw id  0, ..., N1       0, ..., N2           ...  ..., Nk
    new id  0, ..., N1  N1 + 1, ..., N1 + N2 + 1  ...  ..., N1 + ... + Nk + k - 1
    ======  ==========  ========================  ===  ==========================

    The batched graph is read-only, i.e. one cannot further add nodes and edges.
    A ``RuntimeError`` will be raised if one attempts.

    To modify the features in :class:`BatchedDGLGraph` has no effect on the original
    graphs. See the examples below about how to work around.

    Parameters
    ----------
    graph_list : iterable
        A collection of :class:`~dgl.DGLGraph` objects to be batched.
    node_attrs : None, str or iterable, optional
        The node attributes to be batched. If ``None``, the :class:`BatchedDGLGraph` object
        will not have any node attributes. By default, all node attributes will be batched.
        An error will be raised if graphs having nodes have different attributes. If ``str``
        or ``iterable``, this should specify exactly what node attributes to be batched.
    edge_attrs : None, str or iterable, optional
        Same as for the case of :attr:`node_attrs`

    Examples
    --------
    Create two :class:`~dgl.DGLGraph` objects.

    **Instantiation:**

    >>> import dgl
    >>> import torch as th
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)                                # Add 2 nodes
    >>> g1.add_edge(0, 1)                              # Add edge 0 -> 1
    >>> g1.ndata['hv'] = th.tensor([[0.], [1.]])       # Initialize node features
    >>> g1.edata['he'] = th.tensor([[0.]])             # Initialize edge features

    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)                                # Add 3 nodes
    >>> g2.add_edges([0, 2], [1, 1])                   # Add edges 0 -> 1, 2 -> 1
    >>> g2.ndata['hv'] = th.tensor([[2.], [3.], [4.]]) # Initialize node features
    >>> g2.edata['he'] = th.tensor([[1.], [2.]])       # Initialize edge features

    Merge two :class:`~dgl.DGLGraph` objects into one :class:`BatchedDGLGraph` object.
    When merging a list of graphs, we can choose to include only a subset of the attributes.

    >>> bg = dgl.batch([g1, g2], edge_attrs=None)
    >>> bg.edata
    {}

    Below one can see that the nodes are re-indexed. The edges are re-indexed in
    the same way.

    >>> bg.nodes()
    tensor([0, 1, 2, 3, 4])
    >>> bg.ndata['hv']
    tensor([[0.],
            [1.],
            [2.],
            [3.],
            [4.]])

    **Property:**

    We can still get a brief summary of the graphs that constitute the batched graph.

    >>> bg.batch_size
    2
    >>> bg.batch_num_nodes
    [2, 3]
    >>> bg.batch_num_edges
    [1, 2]

    **Readout:**

    Another common demand for graph neural networks is graph readout, which is a
    function that takes in the node attributes and/or edge attributes for a graph
    and outputs a vector summarizing the information in the graph.
    :class:`BatchedDGLGraph` also supports performing readout for a batch of graphs at once.

    Below we take the built-in readout function :func:`sum_nodes` as an example, which
    sums over a particular kind of node attribute for each graph.

    >>> dgl.sum_nodes(bg, 'hv') # Sum the node attribute 'hv' for each graph.
    tensor([[1.],               # 0 + 1
            [9.]])              # 2 + 3 + 4

    **Message passing:**

    For message passing and related operations, :class:`BatchedDGLGraph` acts exactly
    the same as :class:`~dgl.DGLGraph`.

    **Update Attributes:**

    Updating the attributes of the batched graph has no effect on the original graphs.

    >>> bg.edata['he'] = th.zeros(3, 2)
    >>> g2.edata['he']
    tensor([[1.],
            [2.]])}

    Instead, we can decompose the batched graph back into a list of graphs and use them
    to replace the original graphs.

    >>> g1, g2 = dgl.unbatch(bg)    # returns a list of DGLGraph objects
    >>> g2.edata['he']
    tensor([[0., 0.],
            [0., 0.]])}
    """
    def __init__(self, graph_list, node_attrs, edge_attrs):

        def _get_num_item_and_attr_types(g, mode):
            if mode == 'node':
                num_items = g.number_of_nodes()
                attr_types = set(g.node_attr_schemes().keys())
            elif mode == 'edge':
                num_items = g.number_of_edges()
                attr_types = set(g.edge_attr_schemes().keys())
            return num_items, attr_types

        def _init_attrs(attrs, mode):
            if attrs is None:
                return []
            elif is_all(attrs):
                attrs = set()
                # Check if at least a graph has mode items and associated features.
                for i in range(len(graph_list)):
                    g = graph_list[i]
                    g_num_items, g_attrs = _get_num_item_and_attr_types(g, mode)
                    if g_num_items > 0 and len(g_attrs) > 0:
                        attrs = g_attrs
                        ref_g_index = i
                        break
                # Check if all the graphs with mode items have the same associated features.
                if len(attrs) > 0:
                    for i in range(len(graph_list)):
                        g = graph_list[i]
                        g_num_items, g_attrs = _get_num_item_and_attr_types(g, mode)
                        if g_attrs != attrs and g_num_items > 0:
                            raise ValueError('Expect graph {} and {} to have the same {} '
                                             'attributes when {}_attrs=ALL, got {} and '
                                             '{}'.format(ref_g_index, i, mode, mode, attrs, g_attrs))
                return attrs
            elif isinstance(attrs, str):
                return [attrs]
            elif isinstance(attrs, Iterable):
                return attrs
            else:
                raise ValueError('Expected {} attrs to be of type None str or Iterable, '
                                 'got type {}'.format(mode, type(attrs)))

        node_attrs = _init_attrs(node_attrs, 'node')
        edge_attrs = _init_attrs(edge_attrs, 'edge')

        # create batched graph index
        batched_index = gi.disjoint_union([g._graph for g in graph_list])
        # create batched node and edge frames
        if len(node_attrs) == 0:
            batched_node_frame = FrameRef(Frame(num_rows=batched_index.number_of_nodes()))
        else:
            # NOTE: following code will materialize the columns of the input graphs.
            cols = {key: F.cat([gr._node_frame[key] for gr in graph_list
                                if gr.number_of_nodes() > 0], dim=0)
                    for key in node_attrs}
            batched_node_frame = FrameRef(Frame(cols))

        if len(edge_attrs) == 0:
            batched_edge_frame = FrameRef(Frame(num_rows=batched_index.number_of_edges()))
        else:
            cols = {key: F.cat([gr._edge_frame[key] for gr in graph_list
                                if gr.number_of_edges() > 0], dim=0)
                    for key in edge_attrs}
            batched_edge_frame = FrameRef(Frame(cols))

        super(BatchedDGLGraph, self).__init__(
                graph_data=batched_index,
                node_frame=batched_node_frame,
                edge_frame=batched_edge_frame)

        # extra members
        self._batch_size = 0
        self._batch_num_nodes = []
        self._batch_num_edges = []
        for gr in graph_list:
            if isinstance(gr, BatchedDGLGraph):
                # handle the input is again a batched graph.
                self._batch_size += gr._batch_size
                self._batch_num_nodes += gr._batch_num_nodes
                self._batch_num_edges += gr._batch_num_edges
            else:
                self._batch_size += 1
                self._batch_num_nodes.append(gr.number_of_nodes())
                self._batch_num_edges.append(gr.number_of_edges())

    @property
    def batch_size(self):
        """Number of graphs in this batch.

        Returns
        -------
        int
            Number of graphs in this batch."""
        return self._batch_size

    @property
    def batch_num_nodes(self):
        """Number of nodes of each graph in this batch.

        Returns
        -------
        list
            Number of nodes of each graph in this batch."""
        return self._batch_num_nodes

    @property
    def batch_num_edges(self):
        """Number of edges of each graph in this batch.

        Returns
        -------
        list
            Number of edges of each graph in this batch."""
        return self._batch_num_edges

    # override APIs
    def add_nodes(self, num, reprs=None):
        """Add nodes. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edge(self, u, v, reprs=None):
        """Add one edge. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edges(self, u, v, reprs=None):
        """Add many edges. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    # new APIs
    def __getitem__(self, idx):
        """Slice the batch and return the batch of graphs specified by the idx."""
        # TODO
        pass

    def __setitem__(self, idx, val):
        """Set the value of the slice. The graph size cannot be changed."""
        # TODO
        pass

def split(graph_batch, num_or_size_splits):
    """Split the batch."""
    # TODO(minjie): could follow torch.split syntax
    pass

def unbatch(graph):
    """Return the list of graphs in this batch.

    Parameters
    ----------
    graph : BatchedDGLGraph
        The batched graph.

    Returns
    -------
    list
        A list of :class:`~dgl.DGLGraph` objects whose attributes are obtained
        by partitioning the attributes of the :attr:`graph`. The length of the
        list is the same as the batch size of :attr:`graph`.

    Notes
    -----
    Unbatching will break each field tensor of the batched graph into smaller
    partitions.

    For simpler tasks such as node/edge state aggregation, try to use
    readout functions.

    See Also
    --------
    batch
    """
    assert isinstance(graph, BatchedDGLGraph)
    bsize = graph.batch_size
    bn = graph.batch_num_nodes
    be = graph.batch_num_edges
    pttns = gi.disjoint_partition(graph._graph, utils.toindex(bn))
    # split the frames
    node_frames = [FrameRef(Frame(num_rows=n)) for n in bn]
    edge_frames = [FrameRef(Frame(num_rows=n)) for n in be]
    for attr, col in graph._node_frame.items():
        col_splits = F.split(col, bn, dim=0)
        for i in range(bsize):
            node_frames[i][attr] = col_splits[i]
    for attr, col in graph._edge_frame.items():
        col_splits = F.split(col, be, dim=0)
        for i in range(bsize):
            edge_frames[i][attr] = col_splits[i]
    return [DGLGraph(graph_data=pttns[i],
                     node_frame=node_frames[i],
                     edge_frame=edge_frames[i]) for i in range(bsize)]

def batch(graph_list, node_attrs=ALL, edge_attrs=ALL):
    """Batch a collection of :class:`~dgl.DGLGraph` and return a
    :class:`BatchedDGLGraph` object that is independent of the :attr:`graph_list`.

    Parameters
    ----------
    graph_list : iterable
        A collection of :class:`~dgl.DGLGraph` to be batched.
    node_attrs : None, str or iterable
        The node attributes to be batched. If ``None``, the :class:`BatchedDGLGraph`
        object will not have any node attributes. By default, all node attributes will
        be batched. If ``str`` or iterable, this should specify exactly what node
        attributes to be batched.
    edge_attrs : None, str or iterable, optional
        Same as for the case of :attr:`node_attrs`

    Returns
    -------
    BatchedDGLGraph
        one single batched graph

    See Also
    --------
    BatchedDGLGraph
    unbatch
    """
    return BatchedDGLGraph(graph_list, node_attrs, edge_attrs)


_readout_on_attrs = {
        'nodes': ('ndata', 'batch_num_nodes', 'number_of_nodes'),
        'edges': ('edata', 'batch_num_edges', 'number_of_edges'),
        }

def _sum_on(graph, on, input, weight):
    data_attr, batch_num_objs_attr, num_objs_attr = _readout_on_attrs[on]
    data = getattr(graph, data_attr)
    input = data[input]

    if weight is not None:
        weight = data[weight]
        weight = F.reshape(weight, (-1,) + (1,) * (F.ndim(input) - 1))
        input = weight * input

    if isinstance(graph, BatchedDGLGraph):
        n_graphs = graph.batch_size
        batch_num_objs = getattr(graph, batch_num_objs_attr)
        n_objs = getattr(graph, num_objs_attr)()

        seg_id = F.zerocopy_from_numpy(
                np.arange(n_graphs, dtype='int64').repeat(batch_num_objs))
        seg_id = F.copy_to(seg_id, F.context(input))
        y = F.unsorted_1d_segment_sum(input, seg_id, n_graphs, 0)
        return y
    else:
        return F.sum(input, 0)

def sum_nodes(graph, input, weight=None):
    """Sums all the values of node field :attr:`input` in :attr:`graph`, optionally
    multiplies the field by a scalar node field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each node feature with field :attr:`input`.
        for summation. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_nodes(), 1]``.

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    If graph is a :class:`BatchedDGLGraph` object, a stacked tensor is
    returned instead, i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of the
    corresponding example in the batch. If an example has no nodes,
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
    tensor([15.])   # 1 * 3 + 2 * 6

    See Also
    --------
    mean_nodes
    sum_edges
    mean_edges
    """
    return _sum_on(graph, 'nodes', input, weight)

def sum_edges(graph, input, weight=None):
    """Sums all the values of edge field :attr:`input` in :attr:`graph`,
    optionally multiplies the field by a scalar edge field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each edge feature with field :attr:`input`.
        for summation. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_edges(), 1]``.

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    If graph is a :class:`BatchedDGLGraph` object, a stacked tensor is
    returned instead, i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of the
    corresponding example in the batch.  If an example has no edges,
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
    tensor([15.])   # 1 * 3 + 2 * 6

    See Also
    --------
    sum_nodes
    mean_nodes
    mean_edges
    """
    return _sum_on(graph, 'edges', input, weight)


def _mean_on(graph, on, input, weight):
    data_attr, batch_num_objs_attr, num_objs_attr = _readout_on_attrs[on]
    data = getattr(graph, data_attr)
    input = data[input]

    if weight is not None:
        weight = data[weight]
        weight = F.reshape(weight, (-1,) + (1,) * (F.ndim(input) - 1))
        input = weight * input

    if isinstance(graph, BatchedDGLGraph):
        n_graphs = graph.batch_size
        batch_num_objs = getattr(graph, batch_num_objs_attr)
        n_objs = getattr(graph, num_objs_attr)()

        seg_id = F.zerocopy_from_numpy(
                np.arange(n_graphs, dtype='int64').repeat(batch_num_objs))
        seg_id = F.copy_to(seg_id, F.context(input))
        if weight is not None:
            w = F.unsorted_1d_segment_sum(weight, seg_id, n_graphs, 0)
            y = F.unsorted_1d_segment_sum(input, seg_id, n_graphs, 0)
            y = y / w
        else:
            y = F.unsorted_1d_segment_mean(input, seg_id, n_graphs, 0)
        return y
    else:
        if weight is None:
            return F.mean(input, 0)
        else:
            y = F.sum(input, 0) / F.sum(weight, 0)
            return y

def mean_nodes(graph, input, weight=None):
    """Averages all the values of node field :attr:`input` in :attr:`graph`,
    optionally multiplies the field by a scalar node field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : str, optional
        The weight field. If None, no weighting will be performed,
        otherwise, weight each node feature with field :attr:`input`.
        for calculating mean. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_nodes(), 1]``.

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    If graph is a :class:`BatchedDGLGraph` object, a stacked tensor is
    returned instead, i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch. If an example has no nodes,
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
    tensor([1.6667])                 # 1 * (3 / (3 + 6)) + 2 * (6 / (3 + 6))

    See Also
    --------
    sum_nodes
    sum_edges
    mean_edges
    """
    return _mean_on(graph, 'nodes', input, weight)

def mean_edges(graph, input, weight=None):
    """Averages all the values of edge field :attr:`input` in :attr:`graph`,
    optionally multiplies the field by a scalar edge field :attr:`weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : optional, str
        The weight field. If None, no weighting will be performed,
        otherwise, weight each edge feature with field :attr:`input`.
        for calculating mean. The weight feature associated in the :attr:`graph`
        should be a tensor of shape ``[graph.number_of_edges(), 1]``.

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    If graph is a :class:`BatchedDGLGraph` object, a stacked tensor is
    returned instead, i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch.  If an example has no edges,
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
    tensor([1.6667])                 # 1 * (3 / (3 + 6)) + 2 * (6 / (3 + 6))

    See Also
    --------
    sum_nodes
    mean_nodes
    sum_edges
    """
    return _mean_on(graph, 'edges', input, weight)
