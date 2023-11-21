"""Utility functions for sampling."""

import copy
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import torch

from ..base import CSCFormatBase, etype_str_to_tuple
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


def compact_csc_format(
    csc_formats: Union[CSCFormatBase, Dict[str, CSCFormatBase]],
    dst_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
):
    """
    Compact csc formats and return original_row_ids (per type).

    Parameters
    ----------
    csc_formats: Union[CSCFormatBase, Dict[str, CSCFormatBase]]
        CSC formats representing source-destination edges.
        - If `csc_formats` is a CSCFormatBase: It means the graph is
        homogeneous. Also, indptr and indice in it should be torch.tensor
        representing source and destination pairs in csc format. And IDs inside
        are homogeneous ids.
        - If `csc_formats` is a Dict[str, CSCFormatBase]: The keys
        should be edge type and the values should be csc format node pairs.
        And IDs inside are heterogeneous ids.
    dst_nodes: Union[torch.Tensor, Dict[str, torch.Tensor]]
        Nodes of all destination nodes in the node pairs.
        - If `dst_nodes` is a tensor: It means the graph is homogeneous.
        - If `dst_nodes` is a dictionary: The keys are node type and the
        values are corresponding nodes. And IDs inside are heterogeneous ids.

    Returns
    -------
    Tuple[original_row_node_ids, compacted_csc_formats]
        The compacted CSC formats, where node IDs are replaced with mapped node
        IDs, and all nodes (per type).
        "Compacted CSC formats" indicates that the node IDs in the input node
        pairs are replaced with mapped node IDs, where each type of node is
        mapped to a contiguous space of IDs ranging from 0 to N.

    Examples
    --------
    >>> import dgl.graphbolt as gb
    >>> N1 = torch.LongTecnsor([1, 2, 2])
    >>> N2 = torch.LongTensor([5, 6, 5])
    >>> csc_formats = {"n2:e2:n1": CSCFormatBase(indptr=torch.tensor([0, 1]),
    ... indices=torch.tensor([5]))}
    >>> dst_nodes = {"n1": N1[:1]}
    >>> original_row_node_ids, compacted_csc_formats = gb.compact_csc_format(
    ...     csc_formats, dst_nodes
    ... )
    >>> print(original_row_node_ids)
    {'n1': tensor([1]), 'n2': tensor([5])}
    >>> print(compacted_csc_formats)
    {"n2:e2:n1": CSCFormatBase(indptr=tensor([0, 1]),
    ... indices=tensor([0]))}
    """
    is_homogeneous = not isinstance(csc_formats, dict)
    if is_homogeneous:
        if dst_nodes is not None:
            assert isinstance(
                dst_nodes, torch.Tensor
            ), "Edge type not supported in homogeneous graph."
            assert csc_formats.indptr[-1] == len(
                csc_formats.indices
            ), "The last element of indptr should be the same as the length of indices."
            assert len(dst_nodes) + 1 == len(
                csc_formats.indptr
            ), "The seed nodes should correspond to indptr."
        offset = dst_nodes.size(0)
        original_row_ids = torch.cat((dst_nodes, csc_formats.indices))
        compacted_csc_formats = CSCFormatBase(
            indptr=csc_formats.indptr,
            indices=(torch.arange(0, csc_formats.indices.size(0)) + offset),
        )
    else:
        compacted_csc_formats = {}
        original_row_ids = copy.deepcopy(dst_nodes)
        for etype, csc_format in csc_formats.items():
            assert csc_format.indptr[-1] == len(
                csc_format.indices
            ), "The last element of indptr should be the same as the length of indices."
            src_type, _, dst_type = etype_str_to_tuple(etype)
            assert len(dst_nodes[dst_type]) + 1 == len(
                csc_format.indptr
            ), "The seed nodes should correspond to indptr."
            offset = original_row_ids.get(src_type, torch.tensor([])).size(0)
            original_row_ids[src_type] = torch.cat(
                (
                    original_row_ids.get(
                        src_type,
                        torch.tensor([], dtype=csc_format.indices.dtype),
                    ),
                    csc_format.indices,
                )
            )
            compacted_csc_formats[etype] = CSCFormatBase(
                indptr=csc_format.indptr,
                indices=(
                    torch.arange(
                        0,
                        csc_format.indices.size(0),
                        dtype=csc_format.indices.dtype,
                    )
                    + offset
                ),
            )
    return original_row_ids, compacted_csc_formats
