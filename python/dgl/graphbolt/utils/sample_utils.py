"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, Tuple, Union

import torch


def unique_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    ],
    dst_nodes=None,
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
    dst_nodes = defaultdict(list)
    num_seeds = defaultdict(int)
    if dst_nodes is None:
        # Find all nodes that appeared as destinations.
        for etype, node_pair in node_pairs.items():
            dst_nodes[etype[2]].append(node_pair[1])
        dst_nodes = {
            ntype: torch.unique(torch.cat(values, 0))
            for ntype, values in dst_nodes.items()
        }
        num_seeds = {
            ntype: nodes.size(0)
            for ntype, nodes in dst_nodes.items()
        }
    all_nodes = defaultdict(list)
    all_nodes = {
        ntype: [nodes]
        for ntype, nodes in dst_nodes.items()
    }
    # Colllect all nodes that appeared as sources.
    for etype, node_pair in node_pairs.items():
        all_nodes[etype[0]].append(node_pair[0])
    all_nodes = {
        ntype: torch.cat(values, 0)
        for ntype, values in all_nodes.items()
    }
    
    unique_nodes = {}
    compacted_node_pairs = {}
    for ntype, nodes in all_nodes.items():
        unique_nodes, compacted_node_pairs =torch.ops.graphbolt.unique_and_compact(node_pairs, seed_nodes)

    # Return singleton for homogeneous graph.
    if is_homogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        unique_nodes = list(unique_nodes.values())[0]
    return unique_nodes, compacted_node_pairs
