from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch


@dataclass
class CSCSamplingGraphSampledSubGraph:
    r"""Class for sampled subgraph."""

    node_pairs: Union[
        Dict[Tuple[str, str, str], Tuple[torch.tensor, torch.tensor]],
        Tuple[torch.tensor, torch.tensor],
    ]
    reverse_column_node_ids: Union[Dict[str, torch.tensor], torch.tensor] = None
    reverse_row_node_ids: Union[Dict[str, torch.tensor], torch.tensor] = None
    reverse_edge_ids: Union[
        Dict[Tuple[str, str, str], torch.tensor], torch.tensor
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
                    isinstance(item, str) for item in pair
                ), "Nodes in pairs should be of type string."
        else:
            assert (
                isinstance(self.node_pairs, tuple) and len(self.node_pairs) == 2
            ), "Node pair should be a source-destination tuple (u, v)."
            assert all(
                isinstance(item, str) for item in self.node_pairs
            ), "Nodes in pairs should be of type string."
