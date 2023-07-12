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
            The compacted node pairs and the unique nodes (per type).
    """
    is_heterogeneous = isinstance(node_pairs, Dict)
    if not is_heterogeneous:
        node_pairs = {("_N", "_E", "_N"): node_pairs}
    nodes_dict = defaultdict(list)
    # Collect nodes for each node type.
    for etype, node_pair in node_pairs.items():
        u_type, _, v_type = etype
        u, v = node_pair
        nodes_dict[u_type].append(u)
        nodes_dict[v_type].append(v)

    unique_nodes_dict = {}
    compacted_nodes_dict = {}
    for ntype, nodes in nodes_dict.items():
        collected_nodes = torch.cat(nodes)
        # Compact and find unique nodes.
        unique_nodes, collected_nodes = torch.unique(
            collected_nodes,
            return_inverse=True,
        )
        unique_nodes_dict[ntype] = unique_nodes
        compacted_nodes_dict[ntype] = collected_nodes

    # Map back in same order as collect.
    compacted_node_pairs = {}
    unique_nodes = unique_nodes_dict
    for etype, node_pair in node_pairs.items():
        u_type, _, v_type = etype
        u, v = node_pair
        u_size, v_size = u.numel(), v.numel()
        u = compacted_nodes_dict[u_type][:u_size]
        compacted_nodes_dict[u_type] = compacted_nodes_dict[u_type][u_size:]
        v = compacted_nodes_dict[v_type][:v_size]
        compacted_nodes_dict[v_type] = compacted_nodes_dict[v_type][v_size:]
        compacted_node_pairs[etype] = (u, v)
    if not is_heterogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        unique_nodes = list(unique_nodes_dict.values())[0]
    return unique_nodes, compacted_node_pairs
