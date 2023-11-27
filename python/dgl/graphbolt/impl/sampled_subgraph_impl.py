"""Sampled subgraph for FusedCSCSamplingGraph."""
# pylint: disable= invalid-name
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch

from ..base import CSCFormatBase, etype_str_to_tuple
from ..sampled_subgraph import SampledSubgraph

__all__ = ["FusedSampledSubgraphImpl", "SampledSubgraphImpl"]


@dataclass
class FusedSampledSubgraphImpl(SampledSubgraph):
    r"""Sampled subgraph of FusedCSCSamplingGraph.

    Examples
    --------
    >>> node_pairs = {"A:relation:B"): (torch.tensor([0, 1, 2]),
    ... torch.tensor([0, 1, 2]))}
    >>> original_column_node_ids = {'B': torch.tensor([10, 11, 12])}
    >>> original_row_node_ids = {'A': torch.tensor([13, 14, 15])}
    >>> original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
    >>> subgraph = gb.FusedSampledSubgraphImpl(
    ... node_pairs=node_pairs,
    ... original_column_node_ids=original_column_node_ids,
    ... original_row_node_ids=original_row_node_ids,
    ... original_edge_ids=original_edge_ids
    ... )
    >>> print(subgraph.node_pairs)
    {"A:relation:B": (tensor([0, 1, 2]), tensor([0, 1, 2]))}
    >>> print(subgraph.original_column_node_ids)
    {'B': tensor([10, 11, 12])}
    >>> print(subgraph.original_row_node_ids)
    {'A': tensor([13, 14, 15])}
    >>> print(subgraph.original_edge_ids)
    {"A:relation:B": tensor([19, 20, 21])}
    """
    node_pairs: Union[
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor],
    ] = None
    original_column_node_ids: Union[
        Dict[str, torch.Tensor], torch.Tensor
    ] = None
    original_row_node_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None
    original_edge_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None

    def __post_init__(self):
        if isinstance(self.node_pairs, dict):
            for etype, pair in self.node_pairs.items():
                assert (
                    isinstance(etype, str)
                    and len(etype_str_to_tuple(etype)) == 3
                ), "Edge type should be a string in format of str:str:str."
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

    def __repr__(self) -> str:
        return _sampled_subgraph_str(self, "FusedSampledSubgraphImpl")


@dataclass
class SampledSubgraphImpl(SampledSubgraph):
    r"""Sampled subgraph of CSCSamplingGraph.

    Examples
    --------
    >>> node_pairs = {"A:relation:B": CSCFormatBase(indptr=torch.tensor([0, 1, 2, 3]),
    ... indices=torch.tensor([0, 1, 2]))}
    >>> original_column_node_ids = {'B': torch.tensor([10, 11, 12])}
    >>> original_row_node_ids = {'A': torch.tensor([13, 14, 15])}
    >>> original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
    >>> subgraph = gb.SampledSubgraphImpl(
    ... node_pairs=node_pairs,
    ... original_column_node_ids=original_column_node_ids,
    ... original_row_node_ids=original_row_node_ids,
    ... original_edge_ids=original_edge_ids
    ... )
    >>> print(subgraph.node_pairs)
    {"A:relation:B": CSCForamtBase(indptr=torch.tensor([0, 1, 2, 3]),
    ... indices=torch.tensor([0, 1, 2]))}
    >>> print(subgraph.original_column_node_ids)
    {'B': tensor([10, 11, 12])}
    >>> print(subgraph.original_row_node_ids)
    {'A': tensor([13, 14, 15])}
    >>> print(subgraph.original_edge_ids)
    {"A:relation:B": tensor([19, 20, 21])}
    """
    node_pairs: Union[
        CSCFormatBase,
        Dict[str, CSCFormatBase],
    ] = None
    original_column_node_ids: Union[
        Dict[str, torch.Tensor], torch.Tensor
    ] = None
    original_row_node_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None
    original_edge_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None

    def __post_init__(self):
        if isinstance(self.node_pairs, dict):
            for etype, pair in self.node_pairs.items():
                assert (
                    isinstance(etype, str)
                    and len(etype_str_to_tuple(etype)) == 3
                ), "Edge type should be a string in format of str:str:str."
                assert (
                    pair.indptr is not None and pair.indices is not None
                ), "Node pair should be have indptr and indice."
                assert isinstance(pair.indptr, torch.Tensor) and isinstance(
                    pair.indices, torch.Tensor
                ), "Nodes in pairs should be of type torch.Tensor."
        else:
            assert (
                self.node_pairs.indptr is not None
                and self.node_pairs.indices is not None
            ), "Node pair should be have indptr and indice."
            assert isinstance(
                self.node_pairs.indptr, torch.Tensor
            ) and isinstance(
                self.node_pairs.indices, torch.Tensor
            ), "Nodes in pairs should be of type torch.Tensor."

    def __repr__(self) -> str:
        return _sampled_subgraph_str(self, "SampledSubgraphImpl")


def _sampled_subgraph_str(sampled_subgraph: SampledSubgraph, classname) -> str:
    final_str = classname + "("

    def _get_attributes(_obj) -> list:
        attributes = [
            attribute
            for attribute in dir(_obj)
            if not attribute.startswith("__")
            and not callable(getattr(_obj, attribute))
        ]
        return attributes

    attributes = _get_attributes(sampled_subgraph)
    attributes.reverse()

    for name in attributes:
        val = getattr(sampled_subgraph, name)

        def _add_indent(_str, indent):
            lines = _str.split("\n")
            lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
            return "\n".join(lines)

        val = str(val)
        final_str = (
            final_str
            + f"{name}={_add_indent(val, len(name) + len(classname) + 1)},\n"
            + " " * len(classname)
        )
    return final_str[: -len(classname)] + ")"
