"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, Tuple, Union

import torch


def unique_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    ],
    unique_dst_nodes: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ] = None,
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
    unique_dst_nodes: torch.Tensor or Dict[str, torch.Tensor]
        Unique nodes of all destination nodes in the node pairs.
        - If `unique_dst_nodes` is a tensor: It means the graph is homogeneous.
        - If `node_pairs` is a dictionary: The keys are node type and the
        values are corresponding nodes. And IDs inside are heterogeneous ids.

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
        if unique_dst_nodes is not None:
            assert isinstance(
                unique_dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            unique_dst_nodes = {"_N": unique_dst_nodes}

    # Collect all source and destination nodes for each node type.
    src_nodes = defaultdict(list)
    dst_nodes = defaultdict(list)
    for etype, (src_node, dst_node) in node_pairs.items():
        src_nodes[etype[0]].append(src_node)
        dst_nodes[etype[2]].append(dst_node)
    src_nodes = {ntype: torch.cat(nodes) for ntype, nodes in src_nodes.items()}
    dst_nodes = {ntype: torch.cat(nodes) for ntype, nodes in dst_nodes.items()}
    # Compute unique destination nodes if not provided.
    if unique_dst_nodes is None:
        unique_dst_nodes = {
            ntype: torch.unique(nodes) for ntype, nodes in dst_nodes.items()
        }

    ntypes = set(dst_nodes.keys()) | set(src_nodes.keys())
    unique_nodes = {}
    compacted_src = {}
    compacted_dst = {}
    dtype = list(src_nodes.values())[0].dtype
    default_tensor = torch.tensor([], dtype=dtype)
    for ntype in ntypes:
        src = src_nodes.get(ntype, default_tensor)
        unique_dst = unique_dst_nodes.get(ntype, default_tensor)
        dst = dst_nodes.get(ntype, default_tensor)
        (
            unique_nodes[ntype],
            compacted_src[ntype],
            compacted_dst[ntype],
        ) = torch.ops.graphbolt.unique_and_compact(src, dst, unique_dst)

    compacted_node_pairs = {}
    # Map back with the same order.
    for etype, pair in node_pairs.items():
        num_elem = pair[0].size(0)
        src_type, _, dst_type = etype
        src = compacted_src[src_type][:num_elem]
        dst = compacted_dst[dst_type][:num_elem]
        compacted_node_pairs[etype] = (src, dst)
        compacted_src[src_type] = compacted_src[src_type][num_elem:]
        compacted_dst[dst_type] = compacted_dst[dst_type][num_elem:]

    # Return singleton for a homogeneous graph.
    if is_homogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        unique_nodes = list(unique_nodes.values())[0]

    return unique_nodes, compacted_node_pairs
