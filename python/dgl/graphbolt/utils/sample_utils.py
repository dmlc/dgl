"""Utility functions for sampling."""

from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch

from ..base import etype_str_to_tuple
from ..minibatch import MiniBatch


def add_reverse_edges(
    edges: Union[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
    ],
    reverse_etypes_mapping: Dict[str, str] = None,
):
    r"""
    This function finds the reverse edges of the given `edges` and returns the
    composition of them. In a homogeneous graph, reverse edges have inverted
    source and destination node IDs. While in a heterogeneous graph, reversing
    also involves swapping node IDs and their types. This function could be
    used before `exclude_edges` function to help find targeting edges.
    Note: The found reverse edges may not really exists in the original graph.
    And repeat edges could be added becasue reverse edges may already exists in
    the `edges`.

    Parameters
    ----------
    edges : Union[Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                Tuple[torch.Tensor, torch.Tensor]]
        - If sampled subgraph is homogeneous, then `edges` should be a pair of
        of tensors.
        - If sampled subgraph is heterogeneous, then `edges` should be a
        dictionary of edge types and the corresponding edges to exclude.
    reverse_etypes_mapping : Dict[str, str], optional
        The mapping from the original edge types to their reverse edge types.

    Returns
    -------
    Union[Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor]]
        The node pairs contain both the original edges and their reverse
        counterparts.

    Examples
    --------
    >>> edges = {"A:r:B": (torch.tensor([0, 1]), torch.tensor([1, 2]))}
    >>> print(gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"}))
    {'A:r:B': (tensor([0, 1]), tensor([1, 2])),
    'B:rr:A': (tensor([1, 2]), tensor([0, 1]))}

    >>> edges = (torch.tensor([0, 1]), torch.tensor([2, 1]))
    >>> print(gb.add_reverse_edges(edges))
    (tensor([0, 1, 2, 1]), tensor([2, 1, 0, 1]))
    """
    if isinstance(edges, tuple):
        u, v = edges
        return (torch.cat([u, v]), torch.cat([v, u]))
    else:
        combined_edges = edges.copy()
        for etype, reverse_etype in reverse_etypes_mapping.items():
            if etype in edges:
                if reverse_etype in combined_edges:
                    u, v = combined_edges[reverse_etype]
                    u = torch.cat([u, edges[etype][1]])
                    v = torch.cat([v, edges[etype][0]])
                    combined_edges[reverse_etype] = (u, v)
                else:
                    combined_edges[reverse_etype] = (
                        edges[etype][1],
                        edges[etype][0],
                    )
        return combined_edges


def exclude_seed_edges(
    minibatch: MiniBatch,
    include_reverse_edges: bool = False,
    reverse_etypes_mapping: Dict[str, str] = None,
):
    """
    Exclude seed edges with or without their reverse edges from the sampled
    subgraphs in the minibatch.

    Parameters
    ----------
    minibatch : MiniBatch
        The minibatch.
    reverse_etypes_mapping : Dict[str, str] = None
        The mapping from the original edge types to their reverse edge types.
    """
    edges_to_exclude = minibatch.node_pairs
    if include_reverse_edges:
        edges_to_exclude = add_reverse_edges(
            minibatch.node_pairs, reverse_etypes_mapping
        )
    minibatch.sampled_subgraphs = [
        subgraph.exclude_edges(edges_to_exclude)
        for subgraph in minibatch.sampled_subgraphs
    ]
    return minibatch


def unique_and_compact(
    nodes: Union[
        List[torch.Tensor],
        Dict[str, List[torch.Tensor]],
    ],
):
    """
    Compact a list of nodes tensor.

    Parameters
    ----------
    nodes : List[torch.Tensor] or Dict[str, List[torch.Tensor]]
        List of nodes for compacting.
        the unique_and_compact will be done per type
        - If `nodes` is a list of tensor: All the tensors will do unique and
        compact together, usually it is used for homogeneous graph.
        - If `nodes` is a list of dictionary: The keys should be node type and
        the values should be corresponding nodes, the unique and compact will
        be done per type, usually it is used for heterogeneous graph.

    Returns
    -------
    Tuple[unique_nodes, compacted_node_list]
    The Unique nodes (per type) of all nodes in the input. And the compacted
    nodes list, where IDs inside are replaced with compacted node IDs.
    "Compacted node list" indicates that the node IDs in the input node
    list are replaced with mapped node IDs, where each type of node is
    mapped to a contiguous space of IDs ranging from 0 to N.
    """
    is_heterogeneous = isinstance(nodes, dict)

    def unique_and_compact_per_type(nodes):
        nums = [node.size(0) for node in nodes]
        nodes = torch.cat(nodes)
        empty_tensor = nodes.new_empty(0)
        unique, compacted, _ = torch.ops.graphbolt.unique_and_compact(
            nodes, empty_tensor, empty_tensor
        )
        compacted = compacted.split(nums)
        return unique, list(compacted)

    if is_heterogeneous:
        unique, compacted = {}, {}
        for ntype, nodes_of_type in nodes.items():
            unique[ntype], compacted[ntype] = unique_and_compact_per_type(
                nodes_of_type
            )
        return unique, compacted
    else:
        return unique_and_compact_per_type(nodes)


def unique_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
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
    node_pairs : Union[Tuple[torch.Tensor, torch.Tensor],
                    Dict(str, Tuple[torch.Tensor, torch.Tensor])]
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
    >>> node_pairs = {"n1:e1:n2": (N1, N2),
    ...     "n2:e2:n1": (N2, N1)}
    >>> unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
    ...     node_pairs
    ... )
    >>> print(unique_nodes)
    {'n1': tensor([1, 2]), 'n2': tensor([5, 6])}
    >>> print(compacted_node_pairs)
    {"n1:e1:n2": (tensor([0, 1, 1]), tensor([0, 1, 0])),
    "n2:e2:n1": (tensor([0, 1, 0]), tensor([0, 1, 1]))}
    """
    is_homogeneous = not isinstance(node_pairs, dict)
    if is_homogeneous:
        node_pairs = {"_N:_E:_N": node_pairs}
        if unique_dst_nodes is not None:
            assert isinstance(
                unique_dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            unique_dst_nodes = {"_N": unique_dst_nodes}

    # Collect all source and destination nodes for each node type.
    src_nodes = defaultdict(list)
    dst_nodes = defaultdict(list)
    for etype, (src_node, dst_node) in node_pairs.items():
        src_type, _, dst_type = etype_str_to_tuple(etype)
        src_nodes[src_type].append(src_node)
        dst_nodes[dst_type].append(dst_node)
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
        src_type, _, dst_type = etype_str_to_tuple(etype)
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


def duplicated_and_compact_node_pairs(
    node_pairs: Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ],
    duplicated_dst_nodes: Union[
        torch.Tensor,
        Dict[str, torch.Tensor],
    ] = None,
):
    """
    Compact node pairs and return duplicated nodes (per type).

    Parameters
    ----------
    node_pairs : Union[Tuple[torch.Tensor, torch.Tensor],
                    Dict(str, Tuple[torch.Tensor, torch.Tensor])]
        Node pairs representing source-destination edges.
        - If `node_pairs` is a tuple: It means the graph is homogeneous.
        Also, it should be in the format ('u', 'v') representing source
        and destination pairs. And IDs inside are homogeneous ids.
        - If `node_pairs` is a dictionary: The keys should be edge type and
        the values should be corresponding node pairs. And IDs inside are
        heterogeneous ids.
    duplicated_dst_nodes: torch.Tensor or Dict[str, torch.Tensor]
        All destination nodes in the node pairs.
        - If `duplicated_dst_nodes` is a tensor: It means the graph is homogeneous.
        - If `node_pairs` is a dictionary: The keys are node type and the
        values are corresponding nodes. And IDs inside are heterogeneous ids.

    Returns
    -------
    Tuple[duplicated_nodes, node_pairs]
        The compacted node pairs, where node IDs are replaced with mapped node
        IDs, and the duplicated nodes (per type).
        "Compacted node pairs" indicates that the node IDs in the input node
        pairs are replaced with mapped node IDs, where each type of node is
        mapped to a contiguous space of IDs ranging from 0 to N.

    Examples
    --------
    >>> import dgl.graphbolt as gb
    >>> N1 = torch.LongTensor([1, 2, 2])
    >>> N2 = torch.LongTensor([5, 6, 5])
    >>> node_pairs = {"n1:e1:n2": (N1, N2),
    ...     "n2:e2:n1": (N2, N1)}
    >>> unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
    ...     node_pairs
    ... )
    >>> print(unique_nodes)
    {'n1': tensor([1, 2]), 'n2': tensor([5, 6])}
    >>> print(compacted_node_pairs)
    {"n1:e1:n2": (tensor([0, 1, 1]), tensor([0, 1, 0])),
    "n2:e2:n1": (tensor([0, 1, 0]), tensor([0, 1, 1]))}
    """
    is_homogeneous = not isinstance(node_pairs, dict)
    if is_homogeneous:
        node_pairs = {"_N:_E:_N": node_pairs}
        if duplicated_dst_nodes is not None:
            assert isinstance(
                duplicated_dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            duplicated_dst_nodes = {"_N": duplicated_dst_nodes}

    # Collect all source and destination nodes for each node type.
    src_nodes = defaultdict(list)
    dst_nodes = defaultdict(list)
    for etype, (src_node, dst_node) in node_pairs.items():
        src_type, _, dst_type = etype_str_to_tuple(etype)
        src_nodes[src_type].append(src_node)
        dst_nodes[dst_type].append(dst_node)
    src_nodes = {ntype: torch.cat(nodes) for ntype, nodes in src_nodes.items()}
    dst_nodes = {ntype: torch.cat(nodes) for ntype, nodes in dst_nodes.items()}
    # Compute unique destination nodes if not provided.
    if duplicated_dst_nodes is None:
        raise ValueError("Seeds should not be None.")

    ntypes = set(dst_nodes.keys()) | set(src_nodes.keys())
    duplicated_nodes = {}
    compacted_src = {}
    compacted_dst = {}
    dtype = list(src_nodes.values())[0].dtype
    default_tensor = torch.tensor([], dtype=dtype)
    for ntype in ntypes:
        src = src_nodes.get(ntype, default_tensor)
        duplicated_dst = duplicated_dst_nodes.get(ntype, default_tensor)
        dst = dst_nodes.get(ntype, default_tensor)
        duplicated_nodes[ntype] = torch.cat([duplicated_dst, src])
        compacted_src[ntype] = torch.arange(
            len(duplicated_dst), len(duplicated_dst) + len(src), dtype=int
        )
        compacted_dst[ntype] = not_deduplication_compact_dst(
            dst, duplicated_dst
        )

    compacted_node_pairs = {}
    # Map back with the same order.
    for etype, pair in node_pairs.items():
        num_elem = pair[0].size(0)
        src_type, _, dst_type = etype_str_to_tuple(etype)
        src = compacted_src[src_type][:num_elem]
        dst = compacted_dst[dst_type][:num_elem]
        compacted_node_pairs[etype] = (src, dst)
        compacted_src[src_type] = compacted_src[src_type][num_elem:]
        compacted_dst[dst_type] = compacted_dst[dst_type][num_elem:]

    # Return singleton for a homogeneous graph.
    if is_homogeneous:
        compacted_node_pairs = list(compacted_node_pairs.values())[0]
        duplicated_nodes = list(duplicated_nodes.values())[0]

    return duplicated_nodes, compacted_node_pairs


def not_deduplication_compact_dst(dst: torch.Tensor, duplicated_dst: torch.Tensor):
    """
    Compact dst without deduplication.

    Parameters
    ----------
    dst: torch.Tensor
        The original dst that need to be compacted.
    duplicated_dst: torch.Tensor
        Duplicated_dst record the nodes that generate dst in order. Its index
        will be used to help generate the compacted dst.

    Returns
    -------
    torch.Tenor
        Compacted dst without deduplication.
    """
    # Record the number of consecutive identical values.
    dst_constinuous = [1 for _ in duplicated_dst]
    for i in range(1, len(duplicated_dst)):
        dst_constinuous[i] = (
            dst_constinuous[i - 1] + 1
            if (duplicated_dst[i] == duplicated_dst[i - 1])
            else 1
        )
    # Initially, compatc_dst is processed as the corresponding
    # subscript in dst. In this case, consecutively identical values
    # are grouped into the same subscript.
    dst_compact = []
    dst_index = len(dst) - 1
    duplicated_dst_index = len(duplicated_dst) - 1
    compacted_dst_num = {}
    while dst_index >= 0:
        if dst[dst_index] != duplicated_dst[duplicated_dst_index]:
            duplicated_dst_index -= 1
        else:
            dst_compact.append(duplicated_dst_index)
            compacted_dst_num[duplicated_dst_index] = (
                compacted_dst_num.get(duplicated_dst_index, 0) + 1
            )
            dst_index -= 1
    dst_compact.reverse()
    # Renumber the same consecutive subscripts so that they correspond
    # to the correct dst.
    dst_index = len(dst) - 2
    matchtime = 1
    while dst_index >= 0:
        index_nw = dst_compact[dst_index]
        if dst[dst_index] == dst[dst_index + 1]:
            matchtime += 1
            max_matchtime = int(
                compacted_dst_num[index_nw] / dst_constinuous[index_nw]
            )
            if matchtime > max_matchtime:
                dst_compact[dst_index] = (
                    dst_compact[dst_index + max_matchtime] - 1
                )
        else:
            matchtime = 1
        dst_index -= 1
    return torch.tensor(dst_compact, dtype=int)
