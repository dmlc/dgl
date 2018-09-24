"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

from .base import ALL, is_all
from .frame import FrameRef
from .graph import DGLGraph
from . import graph_index as gi
from . import backend as F

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
        # TODO(minjie): handle the input is again a batched graph.
        # create batched graph index
        batched_index = gi.disjoint_union([g._graph for g in graph_list])
        # create batched node and edge frames
        # NOTE: following code will materialize the columns of the input graphs.
        batched_node_frame = FrameRef()
        for gr in graph_list:
            cols = {gr._node_frame[key] for key in node_attrs}
            batched_node_frame.append(cols)
        batched_edge_frame = FrameRef()
        for gr in graph_list:
            cols = {gr._edge_frame[key] for key in edge_attrs}
            batched_edge_frame.append(cols)
        super(BatchedDGLGraph, self).__init__(
                graph_data=batched_index,
                node_frame=batched_node_frame,
                edge_frame=batched_edge_frame)

        # extra members
        self._batch_size = len(graph_list)
        self._batch_num_nodes = [gr.number_of_nodes() for gr in graph_list]
        self._batch_num_edges = [gr.number_of_edges() for gr in graph_list]

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
        """Add nodes."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edge(self, u, v, reprs=None):
        """Add one edge."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edges(self, u, v, reprs=None):
        """Add many edges."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    # new APIs
    def __getitem__(self, idx):
        """Slice the batch and return the batch of graphs specified by the idx."""
        pass

    def __setitem__(self, idx, val):
        """Set the value of the slice. The graph size cannot be changed."""
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

def unbatch(graph_batch):
    """Unbatch the graph and return a list of subgraphs.

    Parameters
    ----------
    graph_batch : DGLGraph
        The batched graph.
    """
    assert False, "disabled for now"
    graph_list = graph_batch.graph_list
    num_graphs = len(graph_list)
    # split and set node attrs
    attrs = [{} for _ in range(num_graphs)] # node attr dict for each graph
    for key in graph_batch.node_attr_schemes():
        vals = F.unpack(graph_batch.pop_n_repr(key), graph_batch.num_nodes)
        for attr, val in zip(attrs, vals):
            attr[key] = val
    for attr, g in zip(attrs, graph_list):
        g.set_n_repr(attr)

    # split and set edge attrs
    attrs = [{} for _ in range(num_graphs)] # edge attr dict for each graph
    for key in graph_batch.edge_attr_schemes():
        vals = F.unpack(graph_batch.pop_e_repr(key), graph_batch.num_edges)
        for attr, val in zip(attrs, vals):
            attr[key] = val
    for attr, g in zip(attrs, graph_list):
        g.set_e_repr(attr)

    return graph_list

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
    elif if isinstance(node_attrs, str):
        node_attrs = [node_attrs]
    if edge_attrs is None:
        edge_attrs = []
    elif is_all(edge_attrs):
        edge_attrs = graph_list[0].edge_attr_schemes()
    elif if isinstance(edge_attrs, str):
        edge_attrs = [edge_attrs]
    return BatchedDGLGraph(graph_list, node_attrs, edge_attrs)
