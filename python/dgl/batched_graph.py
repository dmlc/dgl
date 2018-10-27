"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

from .base import ALL, is_all
from .frame import FrameRef, Frame
from .graph import DGLGraph
from . import graph_index as gi
from . import backend as F
from . import utils

__all__ = ['BatchedDGLGraph', 'batch', 'unbatch', 'split']

class BatchedDGLGraph(DGLGraph):
    """The batched DGL graph.

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
        batched_node_frame = FrameRef()
        for gr in graph_list:
            cols = {key : gr._node_frame[key] for key in node_attrs}
            batched_node_frame.append(cols)
        batched_edge_frame = FrameRef()
        for gr in graph_list:
            cols = {key : gr._edge_frame[key] for key in edge_attrs}
            batched_edge_frame.append(cols)

        cols = {key: F.pack([gr._node_frame[key] for gr in graph_list])
                for key in node_attrs}
        batched_node_frame = FrameRef(Frame(cols))

        cols = {key: F.pack([gr._edge_frame[key] for gr in graph_list])
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

    def readout(self, reduce_func):
        """Perform readout for each graph in the batch.

        The readout value is a tensor of shape (B, D1, D2, ...) where B is the
        batch size.

        Parameters
        ----------
        reduce_func : callable
            The reduce function for readout.

        Returns
        -------
        dict of tensors
            The readout values.
        """
        # TODO
        pass

    '''
    def query_new_node(self, g, u):
        idx = self.graph_idx[g]
        offset = self.node_offset[idx]
        if isinstance(u, (int, np.array, F.Tensor)):
            return u + offset
        else:
            return np.array(u) + offset

    def query_new_edge(self, g, src, dst):
        idx = self.graph_idx[g]
        offset = self.node_offset[idx]
        if isinstance(src, (int, np.ndarray, F.Tensor)) and \
                isinstance(dst, (int, np.ndarray, F.Tensor)):
            return src + offset, dst + offset
        else:
            return np.array(src) + offset, np.array(dst) + offset

    def query_node_start_offset(self):
        return self.node_offset[:-1].copy()

    def query_edge_start_offset(self):
        return self.edge_offset[:-1].copy()
    '''

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
    try to use BatchedDGLGraph.readout().
    """
    assert isinstance(graph, BatchedDGLGraph)
    bsize = graph.batch_size
    bn = graph.batch_num_nodes
    be = graph.batch_num_edges
    bn_offset = [0] + np.cumsum(bn).tolist()
    be_offset = [0] + np.cumsum(be).tolist()
    pttns = gi.disjoint_partition(graph._graph, utils.toindex(bn))
    # split the frames
    node_frames = [FrameRef(graph._node_frame, slice(bn_offset[i], bn_offset[i+1]))
                   for i in range(bsize)]
    edge_frames = [FrameRef(graph._edge_frame, slice(be_offset[i], be_offset[i+1]))
                   for i in range(bsize)]
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
