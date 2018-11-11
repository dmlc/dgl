"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

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

    The batched graph is read-only.

    Parameters
    ----------
    graph_list : iterable
        A list of DGLGraphs to be batched.
    node_attrs : str or iterable
        The node attributes to also be batched.
    edge_attrs : str or iterable, optional
        The edge attributes to also be batched.
    """
    def __init__(self, graph_list, node_attrs, edge_attrs):
        # create batched graph index
        batched_index = gi.disjoint_union([g._graph for g in graph_list])
        # create batched node and edge frames
        # NOTE: following code will materialize the columns of the input graphs.
        cols = {key: F.cat([gr._node_frame[key] for gr in graph_list
                            if gr.number_of_nodes() > 0], dim=0)
                for key in node_attrs}
        batched_node_frame = FrameRef(Frame(cols))

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
        """Number of graphs in this batch."""
        return self._batch_size

    @property
    def batch_num_nodes(self):
        """Number of nodes of each graph in this batch."""
        return self._batch_num_nodes

    @property
    def batch_num_edges(self):
        """Number of edges of each graph in this batch."""
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
    """Unbatch and return the list of graphs in this batch.

    Parameters
    ----------
    graph : BatchedDGLGraph
        The batched graph.

    Notes
    -----
    Unbatching will partition each field tensor of the batched graph into
    smaller partitions.  This is usually wasteful.

    For simpler tasks such as node/edge state aggregation by example,
    try to use readout functions.
    """
    assert isinstance(graph, BatchedDGLGraph)
    bsize = graph.batch_size
    bn = graph.batch_num_nodes
    be = graph.batch_num_edges
    pttns = gi.disjoint_partition(graph._graph, utils.toindex(bn))
    # split the frames
    node_frames = [FrameRef() for i in range(bsize)]
    edge_frames = [FrameRef() for i in range(bsize)]
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
    """Batch a list of DGLGraphs into one single graph.

    Once batch is called, the structure of both merged graph and graphs in graph_list
    must not be mutated, or unbatch's behavior will be undefined.

    Parameters
    ----------
    graph_list : iterable
        A list of DGLGraphs to be batched.
    node_attrs : str or iterable, optional
        The node attributes to also be batched. Specify None to not batch any attributes.
    edge_attrs : str or iterable, optional
        The edge attributes to also be batched. Specify None to not batch any attributes.

    Returns
    -------
    newgrh: BatchedDGLGraph
        one single batched graph
    """
    if node_attrs is None:
        node_attrs = []
    elif is_all(node_attrs):
        node_attrs = graph_list[0].node_attr_schemes()
    elif isinstance(node_attrs, str):
        node_attrs = [node_attrs]
    if edge_attrs is None:
        edge_attrs = []
    elif is_all(edge_attrs):
        edge_attrs = graph_list[0].edge_attr_schemes()
    elif isinstance(edge_attrs, str):
        edge_attrs = [edge_attrs]
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
    """Sums all the values of node field `input` in `graph`, optionally
    multiplies the field by a scalar node field `weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : optional, str
        The weight field.  Default is all 1 (i.e. not weighting)

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    If graph is a BatchedDGLGraph, a stacked tensor is returned instead,
    i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch.  If an example has no nodes,
    a zero tensor with the same shape is returned at the corresponding row.
    """
    return _sum_on(graph, 'nodes', input, weight)

def sum_edges(graph, input, weight=None):
    """Sums all the values of edge field `input` in `graph`, optionally
    multiplies the field by a scalar edge field `weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : optional, str
        The weight field.  Default is all 1 (i.e. not weighting)

    Returns
    -------
    tensor
        The summed tensor.

    Notes
    -----
    If graph is a BatchedDGLGraph, a stacked tensor is returned instead,
    i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch.  If an example has no edges,
    a zero tensor with the same shape is returned at the corresponding row.
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
    """Averages all the values of node field `input` in `graph`, optionally
    multiplies the field by a scalar node field `weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : optional, str
        The weight field.  Default is all 1 (i.e. not weighting)

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    If graph is a BatchedDGLGraph, a stacked tensor is returned instead,
    i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch.  If an example has no nodes,
    a zero tensor with the same shape is returned at the corresponding row.
    """
    return _mean_on(graph, 'nodes', input, weight)

def mean_edges(graph, input, weight=None):
    """Averages all the values of edge field `input` in `graph`, optionally
    multiplies the field by a scalar edge field `weight`.

    Parameters
    ----------
    graph : DGLGraph or BatchedDGLGraph
        The graph
    input : str
        The input field
    weight : optional, str
        The weight field.  Default is all 1 (i.e. not weighting)

    Returns
    -------
    tensor
        The averaged tensor.

    Notes
    -----
    If graph is a BatchedDGLGraph, a stacked tensor is returned instead,
    i.e. having an extra first dimension.
    Each row of the stacked tensor contains the readout result of
    corresponding example in the batch.  If an example has no edges,
    a zero tensor with the same shape is returned at the corresponding row.
    """
    return _mean_on(graph, 'edges', input, weight)
