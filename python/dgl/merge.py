"""Utilities for merging graphs."""

import dgl

from . import backend as F
from .base import DGLError

__all__ = ["merge"]


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

    Returns
    -------
    DGLGraph
        The merged graph.

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
    >>> m = dgl.merge([g, h])

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
        raise DGLError("The input list of graphs cannot be empty.")

    ref = graphs[0]
    ntypes = ref.ntypes
    etypes = ref.canonical_etypes
    data_dict = {etype: ([], []) for etype in etypes}
    num_nodes_dict = {ntype: 0 for ntype in ntypes}
    merged = dgl.heterograph(data_dict, num_nodes_dict, ref.idtype, ref.device)

    # Merge edges and edge data.
    for etype in etypes:
        unmerged_us = []
        unmerged_vs = []
        edata_frames = []
        for graph in graphs:
            etype_id = graph.get_etype_id(etype)
            us, vs = graph.edges(etype=etype)
            unmerged_us.append(us)
            unmerged_vs.append(vs)
            edge_data = graph._edge_frames[etype_id]
            edata_frames.append(edge_data)
        keys = ref.edges[etype].data.keys()
        if len(keys) == 0:
            edges_data = None
        else:
            edges_data = {
                k: F.cat([f[k] for f in edata_frames], dim=0) for k in keys
            }
        merged_us = F.copy_to(
            F.astype(F.cat(unmerged_us, dim=0), ref.idtype), ref.device
        )
        merged_vs = F.copy_to(
            F.astype(F.cat(unmerged_vs, dim=0), ref.idtype), ref.device
        )
        merged.add_edges(merged_us, merged_vs, edges_data, etype)

    # Add node data and isolated nodes from next_graph to merged.
    for next_graph in graphs:
        for ntype in ntypes:
            merged_ntype_id = merged.get_ntype_id(ntype)
            next_ntype_id = next_graph.get_ntype_id(ntype)
            next_ndata = next_graph._node_frames[next_ntype_id]
            node_diff = next_graph.num_nodes(ntype=ntype) - merged.num_nodes(
                ntype=ntype
            )
            n_extra_nodes = max(0, node_diff)
            merged.add_nodes(n_extra_nodes, ntype=ntype)
            next_nodes = F.arange(
                0,
                next_graph.num_nodes(ntype=ntype),
                merged.idtype,
                merged.device,
            )
            merged._node_frames[merged_ntype_id].update_row(
                next_nodes, next_ndata
            )

    return merged
