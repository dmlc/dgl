"""Sampled subgraph for FusedCSCSamplingGraph."""
# pylint: disable= invalid-name
from dataclasses import dataclass
from typing import Dict, Union

import torch

from ..base import CSCFormatBase, etype_str_to_tuple
from ..internal_utils import get_attributes
from ..sampled_subgraph import SampledSubgraph

__all__ = ["SampledSubgraphImpl"]


@dataclass
class SampledSubgraphImpl(SampledSubgraph):
    r"""Sampled subgraph of CSCSamplingGraph.

    Examples
    --------
    >>> sampled_csc = {"A:relation:B": CSCFormatBase(indptr=torch.tensor([0, 1, 2, 3]),
    ... indices=torch.tensor([0, 1, 2]))}
    >>> original_column_node_ids = {'B': torch.tensor([10, 11, 12])}
    >>> original_row_node_ids = {'A': torch.tensor([13, 14, 15])}
    >>> original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
    >>> subgraph = gb.SampledSubgraphImpl(
    ... sampled_csc=sampled_csc,
    ... original_column_node_ids=original_column_node_ids,
    ... original_row_node_ids=original_row_node_ids,
    ... original_edge_ids=original_edge_ids
    ... )
    >>> print(subgraph.sampled_csc)
    {"A:relation:B": CSCForamtBase(indptr=torch.tensor([0, 1, 2, 3]),
    ... indices=torch.tensor([0, 1, 2]))}
    >>> print(subgraph.original_column_node_ids)
    {'B': tensor([10, 11, 12])}
    >>> print(subgraph.original_row_node_ids)
    {'A': tensor([13, 14, 15])}
    >>> print(subgraph.original_edge_ids)
    {"A:relation:B": tensor([19, 20, 21])}
    """
    sampled_csc: Union[CSCFormatBase, Dict[str, CSCFormatBase]] = None
    original_column_node_ids: Union[
        Dict[str, torch.Tensor], torch.Tensor
    ] = None
    original_row_node_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None
    original_edge_ids: Union[Dict[str, torch.Tensor], torch.Tensor] = None

    def __post_init__(self):
        if isinstance(self.sampled_csc, dict):
            for etype, pair in self.sampled_csc.items():
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
                self.sampled_csc.indptr is not None
                and self.sampled_csc.indices is not None
            ), "Node pair should be have indptr and indice."
            assert isinstance(
                self.sampled_csc.indptr, torch.Tensor
            ) and isinstance(
                self.sampled_csc.indices, torch.Tensor
            ), "Nodes in pairs should be of type torch.Tensor."

    def __repr__(self) -> str:
        return _sampled_subgraph_str(self, "SampledSubgraphImpl")


def _sampled_subgraph_str(sampled_subgraph: SampledSubgraph, classname) -> str:
    final_str = classname + "("

    attributes = get_attributes(sampled_subgraph)
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
