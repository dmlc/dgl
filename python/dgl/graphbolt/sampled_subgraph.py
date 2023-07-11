"""Graphbolt sampled subgraph."""
# pylint: disable= invalid-name
from typing import Dict, Tuple

import torch


class SampledSubgraph:
    r"""An abstract class for sampled subgraph. In the context of a
    heterogeneous graph, each field should be of `Dict` type. Otherwise,
    for homogeneous graphs, each field should correspond to its respective
    value type."""

    @property
    def node_pairs(
        self,
    ) -> Tuple[torch.Tensor] or Dict[(str, str, str), Tuple[torch.Tensor]]:
        """Returns the node pairs representing source-destination edges.
        - If `node_pairs` is a tuple: It should be in the format ('u', 'v')
            representing source and destination pairs.
        - If `node_pairs` is a dictionary: The keys should be edge type and
            the values should be corresponding node pairs. The ids inside
            is heterogeneous ids."""
        raise NotImplementedError

    @property
    def reverse_column_node_ids(
        self,
    ) -> torch.Tensor or Dict[str, torch.Tensor]:
        """Returns corresponding reverse column node ids the original graph.
        Column's reverse node ids in the original graph. A graph structure
        can be treated as a coordinated row and column pair, and this is
        the mapped ids of the column.
        - If `reverse_column_node_ids` is a tensor: It represents the
            original node ids.
        - If `reverse_column_node_ids` is a dictionary: The keys should be
            node type and the values should be corresponding original
            heterogeneous node ids.
        If present, it means column IDs are compacted, and `node_pairs`
        column IDs match these compacted ones.
        """
        return None

    @property
    def reverse_row_node_ids(self) -> torch.Tensor or Dict[str, torch.Tensor]:
        """Returns corresponding reverse row node ids the original graph.
        Row's reverse node ids in the original graph. A graph structure
        can be treated as a coordinated row and column pair, and this is
        the mapped ids of the row.
        - If `reverse_row_node_ids` is a tensor: It represents the
            original node ids.
        - If `reverse_row_node_ids` is a dictionary: The keys should be
            node type and the values should be corresponding original
            heterogeneous node ids.
        If present, it means row IDs are compacted, and `node_pairs`
        row IDs match these compacted ones."""
        return None

    @property
    def reverse_edge_ids(self) -> torch.Tensor or Dict[str, torch.Tensor]:
        """Returns corresponding reverse edge ids the original graph.
        Reverse edge ids in the original graph. This is useful when edge
            features are needed.
            - If `reverse_edge_ids` is a tensor: It represents the
                original edge ids.
            - If `reverse_edge_ids` is a dictionary: The keys should be
                edge type and the values should be corresponding original
                heterogeneous edge ids.
        """
        return None
