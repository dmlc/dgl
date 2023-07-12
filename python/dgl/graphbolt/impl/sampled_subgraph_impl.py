"""Sampled subgraph for CSCSamplingGraph."""
# pylint: disable= invalid-name
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from ..sampled_subgraph import SampledSubgraph


@dataclass
class SampledSubgraphImpl(SampledSubgraph):
    r"""Class for sampled subgraph specific for CSCSamplingGraph.

    Examples
    --------
    >>> node_pairs = {('A', 'relation', 'B'): (torch.tensor([0, 1, 2]),
    ... torch.tensor([0, 1, 2]))}
    >>> reverse_column_node_ids = {'B': torch.tensor([10, 11, 12])}
    >>> reverse_row_node_ids = {'A': torch.tensor([13, 14, 15])}
    >>> reverse_edge_ids = {('A', 'relation', 'B'): torch.tensor([19, 20, 21])}
    >>> subgraph = gb.SampledSubgraphImpl(
    ... node_pairs=node_pairs,
    ... reverse_column_node_ids=reverse_column_node_ids,
    ... reverse_row_node_ids=reverse_row_node_ids,
    ... reverse_edge_ids=reverse_edge_ids
    ... )
    >>> print(subgraph.node_pairs)
    {('A', 'relation', 'B'): (tensor([0, 1, 2]), tensor([0, 1, 2]))}
    >>> print(subgraph.reverse_column_node_ids)
    {'B': tensor([10, 11, 12])}
    >>> print(subgraph.reverse_row_node_ids)
    {'A': tensor([13, 14, 15])}
    >>> print(subgraph.reverse_edge_ids)
    {('A', 'relation', 'B'): tensor([19, 20, 21])}
    """
    node_pairs: Union[
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
    ] = None
    reverse_column_node_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None
    reverse_row_node_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None
    reverse_edge_ids: Union[
        Dict[Tuple[str, str, str], torch.Tensor], torch.Tensor
    ] = None

    def __post_init__(self):
        if isinstance(self.node_pairs, dict):
            for etype, pair in self.node_pairs.items():
                assert (
                    isinstance(etype, tuple) and len(etype) == 3
                ), "Edge type should be a triplet of strings (str, str, str)."
                assert all(
                    isinstance(item, str) for item in etype
                ), "Edge type should be a triplet of strings (str, str, str)."
                assert (
                    isinstance(pair, tuple) and len(pair) == 2
                ), "Node pair should be a source-destination tuple (u, v)."
                assert all(
                    isinstance(item, torch.Tensor) for item in pair
                ), "Nodes in pairs should be of type torch.Tensor."
        else:
            assert (
                isinstance(self.node_pairs, tuple) and len(self.node_pairs) == 2
            ), "Node pair should be a source-destination tuple (u, v)."
            assert all(
                isinstance(item, torch.Tensor) for item in self.node_pairs
            ), "Nodes in pairs should be of type torch.Tensor."


def _to_reverse_ids(node_pair, reverse_row_node_ids, reverse_column_node_ids):
    u, v = node_pair
    if reverse_row_node_ids is not None:
        u = reverse_row_node_ids[u]
    if reverse_column_node_ids is not None:
        v = reverse_column_node_ids[v]
    return (u, v)


def _relabel_homo_edges(lhs_edges, rhs_edges):
    """Relabel the source nodes and destination nodes independently for two set
    of edges."""
    src = torch.cat([lhs_edges[0], rhs_edges[0]])
    dst = torch.cat([lhs_edges[1], rhs_edges[1]])
    _, src = torch.unique(src, return_inverse=True)
    _, dst = torch.unique(dst, return_inverse=True)
    lhs_edges = (src[: lhs_edges[0].numel()], dst[: lhs_edges[0].numel()])
    rhs_edges = (src[lhs_edges[0].numel() :], dst[lhs_edges[0].numel() :])
    return lhs_edges, rhs_edges


def _exclude_homo_edges(lhs_edges, rhs_edges):
    lhs_edges, rhs_edges = _relabel_homo_edges(lhs_edges, rhs_edges)
    # Compact edges to integers to use torch.isin to exclude edges
    dst_range = lhs_edges[0].numel() + rhs_edges[0].numel()
    lhs_val = lhs_edges[0] * dst_range + lhs_edges[1]
    rhs_val = rhs_edges[0] * dst_range + rhs_edges[1]
    mask = ~torch.isin(lhs_val, rhs_val)
    return torch.nonzero(mask, as_tuple=True)[0]


def _slice_subgraph(subgraph: SampledSubgraphImpl, index: torch.Tensor):
    """Slice the subgraph according to the index."""

    def _index_select(obj, index):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj[index]
        if isinstance(obj, tuple):
            return tuple(_index_select(v, index) for v in obj)
        # Handle the case when obj is a dictionary
        assert isinstance(obj, dict)
        assert isinstance(index, dict)
        ret = {}
        for k, v in obj.items():
            ret[k] = _index_select(v, index[k])
        return ret

    return SampledSubgraphImpl(
        node_pairs=_index_select(subgraph.node_pairs, index),
        reverse_column_node_ids=subgraph.reverse_column_node_ids,
        reverse_row_node_ids=subgraph.reverse_row_node_ids,
        reverse_edge_ids=_index_select(subgraph.reverse_edge_ids, index),
    )


def exclude_edges(
    subgraph: SampledSubgraphImpl,
    edges: Union[
        Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
    ],
) -> SampledSubgraphImpl:
    r"""Exclude edges from the sampled subgraph.


    Parameters
    ----------
    subgraph : SampledSubgraphImpl
        The sampled subgraph.
    edges : Union[Dict[Tuple[str, str, str], Tuple[torch.Tensor, torch.Tensor]],
    Tuple[torch.Tensor, torch.Tensor]]
        Edges to exclude. If sampled subgraph is homogeneous, then `edges`
        should be a pair of tensors representing the edges to exclude. If
        sampled subgraph is heterogeneous, then `edges` should be a dictionary
        of edge types and the corresponding edges to exclude.

    Returns
    -------
    SampledSubgraphImpl
        The sampled subgraph with the excluded edges.
    """
    assert isinstance(subgraph.node_pairs, tuple) == isinstance(edges, tuple), (
        "The sampled subgraph and the edges to exclude should be both "
        "homogeneous or both heterogeneous."
    )
    # Three steps to exclude edges:
    # 1. Convert the node pairs to the reverse ids.
    # 2. Exclude the edges and get the index of the edges to keep.
    # 3. Slice the subgraph according to the index.
    if isinstance(subgraph.node_pairs, tuple):
        reverse_edges = _to_reverse_ids(
            subgraph.node_pairs,
            subgraph.reverse_row_node_ids,
            subgraph.reverse_column_node_ids,
        )
        index = _exclude_homo_edges(reverse_edges, edges)
        return _slice_subgraph(subgraph, index)
    else:
        index = {}
        for etype, pair in subgraph.node_pairs.items():
            reverse_row_node_ids = (
                None
                if subgraph.reverse_row_node_ids is None
                else subgraph.reverse_row_node_ids.get(etype[0])
            )
            reverse_column_node_ids = (
                None
                if subgraph.reverse_column_node_ids is None
                else subgraph.reverse_column_node_ids.get(etype[2])
            )
            reverse_edges = _to_reverse_ids(
                pair,
                reverse_row_node_ids,
                reverse_column_node_ids,
            )
            index[etype] = _exclude_homo_edges(reverse_edges, edges.get(etype))
        return _slice_subgraph(subgraph, index)
