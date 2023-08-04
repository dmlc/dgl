"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, Tuple, Union

import torch


def unique_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    ],
    unique_dst_nodes=None,
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
    is_homogeneous = not isinstance(node_pairs, dict)

    if is_homogeneous:
        node_pairs = {("_N", "_E", "_N"): node_pairs}

    # Find all destination nodes.
    dst_nodes = defaultdict(list)
    for etype, (src_node, dst_node) in node_pairs.items():
        dst_nodes[etype[2]].append(dst_node)

    # Compute unique destination nodes if not provided.
    if unique_dst_nodes is None:
        unique_dst_nodes = {
            ntype: torch.unique(torch.cat(nodes))
            for ntype, nodes in dst_nodes.items()
        }

    # Collect all source nodes.
    src_nodes = defaultdict(list)
    for etype, (src_node, dst_node) in node_pairs.items():
        src_nodes[etype[0]].append(src_node)

    ntypes = set(dst_nodes.keys()) | set(src_nodes.keys())

    unique_nodes = {}
    compacted_src = defaultdict(list)
    compacted_dst = defaultdict(list)
    for ntype in ntypes:
        src, dst = src_nodes[ntype], dst_nodes[ntype]
        # When 'unique_dst_nodes' is empty, 'dst' must also be empty and 'src' must have value.
        dtype = src[0].dtype if src else dst[0].dtype
        unique_dst = unique_dst_nodes.get(ntype, torch.tensor([], dtype=dtype))
        (
            unique_nodes[ntype],
            compacted_src[ntype],
            compacted_dst[ntype],
        ) = torch.ops.graphbolt.unique_and_compact(src, dst, unique_dst)

    compacted_node_pairs = {}
    # Map back with the same order.
    for etype, _ in node_pairs.items():
        u_type, _, v_type = etype
        u, v = compacted_src[u_type].pop(0), compacted_dst[v_type].pop(0)
        compacted_node_pairs[etype] = (u, v)

    # Return singleton for a homogeneous graph.
    if is_homogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        unique_nodes = list(unique_nodes.values())[0]

    return unique_nodes, compacted_node_pairs
