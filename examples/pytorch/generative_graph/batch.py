# (lingfan) copy paste Minjie's experimental batched graph

from dgl.graph import DGLGraph
import dgl.backend as F
import dgl
import numpy as np

def batch(graph_list, newgrh=None,  node_attrs=None, edge_attrs=None):
    """Batch a list of DGLGraphs into one single graph.
    The structure of merged graph is assumed to be immutable, or unbatch may fail
    The graph_list will be reused when merged graph is split, so they should not be mutated.

    Parameters
    ----------
    graph_list : iterable
        A list of DGLGraphs to be batched.
    newgrh : DGLGraph
        If newgrh is None, then a new graph will be created
    node_attrs : str or iterable
        A list of node attributes needed for merged graph
        It's user's resposiblity to make sure node_attrs exists
    edge_attrs : str or iterable
        A list of edge attributes needed for merged graph
        It's user's resposiblity to make sure edge_attrs exists

    Return
    ------
    newgrh: DGLGraph
        a single merged graph
    unbatch: callable
        callable to split the merged graph
    """
    # FIXME(lingfan): make merged graph read-only
    if newgrh is None:
        newgrh = DGLGraph()
    else:
        newgrh.clear()

    # calc index offset
    num_nodes = [len(g) for g in graph_list]
    node_offset = np.cumsum([0] + num_nodes)
    num_edges = [g.size() for g in graph_list]
    edge_offset = np.cumsum([0] + num_edges)

    # in-order add relabeled nodes
    new_node_list = [np.array(g.nodes) + offset
                        for g, offset in zip(graph_list, node_offset[:-1])]
    newgrh.add_nodes_from(range(node_offset[-1]))

    # in-order add relabeled edges
    new_edge_list = [np.array(g.edges) + offset
                        for g, offset in zip(graph_list, node_offset[:-1])]
    newgrh.add_edges_from(np.concatenate(new_edge_list))
    assert(newgrh.size() == edge_offset[-1])

    # set new graph attr
    if node_attrs is None:
        node_attrs = [dgl.__REPR__]
    hu = {}
    for key in node_attrs:
        vals = [g.pop_n_repr(key) for g in graph_list]
        if all(v is not None for v in vals):
            hu[key] = F.pack(vals)
        else:
            pass # ignore attributes with missing values
    if len(hu) > 0:
        newgrh.set_n_repr(hu)

    if edge_attrs is None:
        edge_attrs = [dgl.__REPR__]
    he = {}
    for key in edge_attrs:
        vals = [g.pop_e_repr(key) for g in graph_list]
        if all(v is not None for v in vals):
            he[key] = F.pack(vals)
        else:
            pass # ignore attributes with missing values
    if len(he) > 0:
        newgrh.set_e_repr(he)

    # record valid node/edge attr name
    node_attrs = hu.keys()
    edge_attrs = he.keys()

    # closure for unbatch
    def unbatch(graph_batch):
        """Unbatch the graph and return a list of subgraphs.

        Parameters
        ----------
        graph_batch : DGLGraph
            The batched graph.
        """
        num_graphs = len(graph_list)
        # split and set node attrs
        if len(node_attrs) > 0:
            hu = [{}] * num_graphs # node attr dict for each graph
            for key in node_attrs:
                vals = F.unpack(graph_batch.pop_n_repr(key), num_nodes)
                for h, val in zip(hu, vals):
                    h[key] = val
            for h, g in zip(hu, graph_list):
                g.set_n_repr(h)

        # split and set edge attrs
        if len(edge_attrs) > 0:
            he = [{}] * num_graphs # edge attr dict for each graph
            for key in edge_attrs:
                vals = F.unpack(graph_batch.pop_e_repr(key), num_edges)
                for h, val in zip(hu, vals):
                    h[key] = val
            for h, g in zip(hu, graph_list):
                g.set_e_repr(h)

        return graph_list

    return newgrh, unbatch, new_node_list, new_edge_list
