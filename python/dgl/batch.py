"""Classes and functions for batching multiple graphs together."""
from __future__ import absolute_import

import numpy as np

from dgl.graph import DGLGraph
import dgl.backend as F
import dgl

class BatchedDGLGraph(DGLGraph):
    def __init__(self, graph_list, node_attrs=None, edge_attrs=None, **attr):
        super(BatchedDGLGraph, self).__init__(**attr)
        self.graph_list = graph_list
        self.graph_idx = {}
        for idx, g in enumerate(self.graph_list):
            self.graph_idx[g] = idx

        self.num_nodes = [len(g) for g in self.graph_list]
        self.num_edges = [g.size() for g in self.graph_list]

        # calc index offset
        self.node_offset = np.cumsum([0] + self.num_nodes)
        self.edge_offset = np.cumsum([0] + self.num_edges)

        # in-order add relabeled nodes
        self.add_nodes_from(range(self.node_offset[-1]))

        # in-order add relabeled edges
        self.new_edge_list = [np.array(g.edge_list) + offset
                        for g, offset in zip(self.graph_list, self.node_offset[:-1])
                        if len(g.edge_list) > 0]
        self.new_edges = np.concatenate(self.new_edge_list)
        self.add_edges_from(self.new_edges)

        assert self.size() == self.edge_offset[-1]

        # set new node attr
        if node_attrs:
            attrs = {}
            for key in node_attrs:
                vals = [g.pop_n_repr(key) for g in self.graph_list]
                attrs[key] = F.pack(vals)
            self.set_n_repr(attrs)
        else:
            for g in self.graph_list:
                self._node_frame.append(g._node_frame)

        # set new edge attr
        if edge_attrs:
            attrs = {}
            for key in edge_attrs:
                vals = [g.pop_e_repr(key) for g in self.graph_list]
                attrs[key] = F.pack(vals)
            self.set_e_repr(attrs)
        else:
            for g in self.graph_list:
                self._edge_frame.append(g._edge_frame)

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


def unbatch(graph_batch):
    """Unbatch the graph and return a list of subgraphs.

    Parameters
    ----------
    graph_batch : DGLGraph
        The batched graph.
    """
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


# FIXME (lingfan): Do we really need the batch API?
#                  Can't we let user call BatchedDGLGraph(graph_list) directly
#                  and make unbatch a member function of BatchedDGLGraph
def batch(graph_list, node_attrs=None, edge_attrs=None):
    """Batch a list of DGLGraphs into one single graph.
    Once batch is called, the structure of both merged graph and graphs in graph_list
    must not bbe mutated, or unbatch's behavior will be undefined.

    Parameters
    ----------
    graph_list : iterable
        A list of DGLGraphs to be batched.
    node_attrs : str or iterable
        A list of node attributes needed for merged graph
        It's user's resposiblity to make sure node_attrs exists
    edge_attrs : str or iterable
        A list of edge attributes needed for merged graph
        It's user's resposiblity to make sure edge_attrs exists

    Return
    ------
    newgrh: DGLGraph
        one single merged graph
    """
    return BatchedDGLGraph(graph_list, node_attrs, edge_attrs)
