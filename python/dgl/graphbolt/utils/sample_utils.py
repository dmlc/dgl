"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, Tuple, Union

import torch


def unique_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    ]
):
    """
    Compact node pairs and return unique nodes (per type).

    Parameters
    ----------
    node_pairs : Tuple[torch.Tensor, torch.Tensor] or \
        Dict(Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor])
        Node pairs representing source-destination edges.
        - If `node_pairs` is a tuple: It means the graph is homogeneous.
        Also, it should be in the format ('u', 'v') representing source
        and destination pairs. And IDs inside are homogeneous ids.
        - If `node_pairs` is a dictionary: The keys should be edge type and
        the values should be corresponding node pairs. And IDs inside are
        heterogeneous ids.

    Returns
    -------
    Tuple[node_pairs, unique_nodes]
        The compacted node pairs, where node IDs are replaced with mapped node
        IDs, and the unique nodes (per type).
        "Compacted node pairs" indicates that the node IDs in the input node
        pairs are replaced with mapped node IDs, where each type of node is
        mapped to a contiguous space of IDs ranging from 0 to N.

    Examples
    --------
    >>> import dgl.graphbolt as gb
    >>> N1 = torch.LongTensor([1, 2, 2])
    >>> N2 = torch.LongTensor([5, 6, 5])
    >>> node_pairs = {("n1", "e1", "n2"): (N1, N2),
    ...     ("n2", "e2", "n1"): (N2, N1)}
    >>> unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
    ...     node_pairs
    ... )
    >>> print(unique_nodes)
    {'n1': tensor([1, 2]), 'n2': tensor([5, 6])}
    >>> print(compacted_node_pairs)
    {('n1', 'e1', 'n2'): (tensor([0, 1, 1]), tensor([0, 1, 0])),
    ('n2', 'e2', 'n1'): (tensor([0, 1, 0]), tensor([0, 1, 1]))}
    """
    is_homogeneous = not isinstance(node_pairs, Dict)
    if is_homogeneous:
        node_pairs = {("_N", "_E", "_N"): node_pairs}
    nodes_dict = defaultdict(list)
    # Collect nodes for each node type.
    for etype, node_pair in node_pairs.items():
        u_type, _, v_type = etype
        u, v = node_pair
        nodes_dict[u_type].append(u)
        nodes_dict[v_type].append(v)

    unique_nodes_dict = {}
    inverse_indices_dict = {}
    for ntype, nodes in nodes_dict.items():
        collected_nodes = torch.cat(nodes)
        # Compact and find unique nodes.
        unique_nodes, inverse_indices = torch.unique(
            collected_nodes,
            return_inverse=True,
        )
        unique_nodes_dict[ntype] = unique_nodes
        inverse_indices_dict[ntype] = inverse_indices

    # Map back in same order as collect.
    compacted_node_pairs = {}
    unique_nodes = unique_nodes_dict
    for etype, node_pair in node_pairs.items():
        u_type, _, v_type = etype
        u, v = node_pair
        u_size, v_size = u.numel(), v.numel()
        u = inverse_indices_dict[u_type][:u_size]
        inverse_indices_dict[u_type] = inverse_indices_dict[u_type][u_size:]
        v = inverse_indices_dict[v_type][:v_size]
        inverse_indices_dict[v_type] = inverse_indices_dict[v_type][v_size:]
        compacted_node_pairs[etype] = (u, v)

    # Return singleton for homogeneous graph.
    if is_homogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        unique_nodes = list(unique_nodes_dict.values())[0]
    return unique_nodes, compacted_node_pairs
