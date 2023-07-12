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
    >>> node_pairs = {('A', 'B', 'relation'): (torch.tensor([1, 2, 3]),
    ... torch.tensor([4, 5, 6]))}
    >>> reverse_column_node_ids = {'A': torch.tensor([7, 8, 9]),
    ... 'B': torch.tensor([10, 11, 12])}
    >>> reverse_row_node_ids = {'A': torch.tensor([13, 14, 15]),
    ... 'B': torch.tensor([16, 17, 18])}
    >>> reverse_edge_ids = {('A', 'B', 'relation'): torch.tensor([19, 20, 21])}
    >>> subgraph = gb.SampledSubgraphImpl(
    ... node_pairs=node_pairs,
    ... reverse_column_node_ids=reverse_column_node_ids,
    ... reverse_row_node_ids=reverse_row_node_ids,
    ... reverse_edge_ids=reverse_edge_ids
    ... )
    >>> print(subgraph.node_pairs)
    {('A', 'B', 'relation'): (tensor([1, 2, 3]), tensor([4, 5, 6]))}
    >>> print(subgraph.reverse_column_node_ids)
    {'A': tensor([7, 8, 9]), 'B': tensor([10, 11, 12])}
    >>> print(subgraph.reverse_row_node_ids)
    {'A': tensor([13, 14, 15]), 'B': tensor([16, 17, 18])}
    >>> print(subgraph.reverse_edge_ids)
    {('A', 'B', 'relation'): tensor([19, 20, 21])}
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
    # TODO(zhenkun): Implement this.
    raise NotImplementedError
