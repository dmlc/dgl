"""Utilities for merging graphs."""

import dgl
from . import backend as F
from .base import DGLError

__all__ = ['merge']

def merge(graphs):
    r"""Merge a sequence of graphs together into a single graph.

    Nodes and edges that exist in ``graphs[i+1]`` but not in ``dgl.merge(graphs[0:i+1])``
    will be added to ``dgl.merge(graphs[0:i+1])`` along with their data.
    Nodes that exist in both ``dgl.merge(graphs[0:i+1])`` and ``graphs[i+1]``
    will be updated with ``graphs[i+1]``'s data if they do not match.

    Parameters
    ----------
    graphs : list[DGLGraph]
        Input graphs.

    Notes
    ----------
    * Inplace updates are applied to a new, empty graph.
    * Features that exist in ``dgl.graphs[i+1]`` will be created in
      ``dgl.merge(dgl.graphs[i+1])`` if they do not already exist.

    Examples
    ----------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch
    >>> g = dgl.graph((torch.tensor([0,1]), torch.tensor([2,3])))
    >>> g.ndata["x"] = torch.zeros(4)
    >>> h = dgl.graph((torch.tensor([1,2]), torch.tensor([0,4])))
    >>> h.ndata["x"] = torch.ones(5)
    >>> m = dgl.merge([g,h])

    ``m`` now contains edges and nodes from ``h`` and ``g``.
    >>> m.edges()
    (tensor([0, 1, 1, 2]), tensor([2, 3, 0, 4]))
    >>> m.nodes()
    tensor([0, 1, 2, 3, 4])

    ``g``'s data has updated with ``h``'s in ``m``.
    >>> m.ndata["x"]
    tensor([1., 1., 1., 1., 1.])

    See Also
    ----------
    add_nodes
    add_edges
    """

    if len(graphs) == 0:
        raise DGLError('The input list of graphs cannot be empty.')

    ref = graphs[0]
    ntypes = ref.ntypes
    etypes = ref.canonical_etypes
    data_dict = {etype: ([],[]) for etype in etypes}
    num_nodes_dict = {ntype: 0 for ntype in ntypes}
    merged = dgl.heterograph(data_dict, num_nodes_dict, ref.idtype, ref.device)

    for next_graph in graphs:
        # Add edges, edge data, and non-isolated nodes from next_graph to merged.
        for etype in etypes:
            next_etype_id = next_graph.get_etype_id(etype)
            next_edges = next_graph.edges(etype=etype)
            next_edge_data = next_graph._edge_frames[next_etype_id]
            merged.add_edges(*next_edges, next_edge_data, etype)

        # Add node data and isolated nodes from next_graph to merged.
        for ntype in ntypes:
            merged_ntype_id = merged.get_ntype_id(ntype)
            next_ntype_id = next_graph.get_ntype_id(ntype)
            next_ndata = next_graph._node_frames[next_ntype_id]
            node_diff = (next_graph.num_nodes(ntype=ntype) -
                         merged.num_nodes(ntype=ntype))
            n_extra_nodes = max(0, node_diff)
            merged.add_nodes(n_extra_nodes, ntype=ntype)
            next_nodes = F.arange(
                0, next_graph.num_nodes(ntype=ntype), merged.idtype, merged.device
            )
            merged._node_frames[merged_ntype_id].update_row(
                next_nodes, next_ndata
            )

    return merged
