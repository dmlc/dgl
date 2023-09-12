"""Graphbolt sampled subgraph."""
# pylint: disable= invalid-name
from typing import Dict, Tuple, Union

import torch

from .base import etype_str_to_tuple


class SampledSubgraph:
    r"""An abstract class for sampled subgraph. In the context of a
    heterogeneous graph, each field should be of `Dict` type. Otherwise,
    for homogeneous graphs, each field should correspond to its respective
    value type."""

    @property
    def node_pairs(
        self,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    ]:
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
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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
    def reverse_row_node_ids(
        self,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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
    def reverse_edge_ids(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
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

    def exclude_edges(
        self,
        edges: Union[
            Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ):
        r"""Exclude edges from the sampled subgraph.

        This function can be used with sampled subgraphs, regardless of whether they
        have compacted row/column nodes or not. If the original subgraph has
        compacted row or column nodes, the corresponding row or column nodes in the
        returned subgraph will also be compacted.

        Parameters
        ----------
        self : SampledSubgraph
            The sampled subgraph.
        edges : Union[Dict[str, Tuple[torch.Tensor, torch.Tensor]],
                    Tuple[torch.Tensor, torch.Tensor]]
            Edges to exclude. If sampled subgraph is homogeneous, then `edges`
            should be a pair of tensors representing the edges to exclude. If
            sampled subgraph is heterogeneous, then `edges` should be a dictionary
            of edge types and the corresponding edges to exclude.

        Returns
        -------
        Tuple[node_pairs, reverse_column_node_ids
                reverse_row_node_ids, reverse_edge_ids]
            The data tuple is employed for constructing an implemented sampled
            subgraph.
        """
        assert isinstance(self.node_pairs, tuple) == isinstance(edges, tuple), (
            "The sampled subgraph and the edges to exclude should be both "
            "homogeneous or both heterogeneous."
        )
        # Three steps to exclude edges:
        # 1. Convert the node pairs to the original ids if they are compacted.
        # 2. Exclude the edges and get the index of the edges to keep.
        # 3. Slice the subgraph according to the index.
        if isinstance(self.node_pairs, tuple):
            reverse_edges = _to_reverse_ids(
                self.node_pairs,
                self.reverse_row_node_ids,
                self.reverse_column_node_ids,
            )
            index = _exclude_homo_edges(reverse_edges, edges)
            return _slice_subgraph(self, index)
        else:
            index = {}
            for etype, pair in self.node_pairs.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                reverse_row_node_ids = (
                    None
                    if self.reverse_row_node_ids is None
                    else self.reverse_row_node_ids.get(src_type)
                )
                reverse_column_node_ids = (
                    None
                    if self.reverse_column_node_ids is None
                    else self.reverse_column_node_ids.get(dst_type)
                )
                reverse_edges = _to_reverse_ids(
                    pair,
                    reverse_row_node_ids,
                    reverse_column_node_ids,
                )
                index[etype] = _exclude_homo_edges(
                    reverse_edges, edges.get(etype)
                )
            return _slice_subgraph(self, index)


def _to_reverse_ids(node_pair, reverse_row_node_ids, reverse_column_node_ids):
    u, v = node_pair
    if reverse_row_node_ids is not None:
        u = reverse_row_node_ids[u]
    if reverse_column_node_ids is not None:
        v = reverse_column_node_ids[v]
    return (u, v)


def _relabel_two_arrays(lhs_array, rhs_array):
    """Relabel two arrays into a consecutive range starting from 0."""
    concated = torch.cat([lhs_array, rhs_array])
    _, mapping = torch.unique(concated, return_inverse=True)
    return mapping[: lhs_array.numel()], mapping[lhs_array.numel() :]


def _exclude_homo_edges(edges, edges_to_exclude):
    """Return the indices of edges that are not in edges_to_exclude."""
    # 1. Relabel edges.
    src, src_to_exclude = _relabel_two_arrays(edges[0], edges_to_exclude[0])
    dst, dst_to_exclude = _relabel_two_arrays(edges[1], edges_to_exclude[1])
    # 2. Compact the edges to integers.
    dst_max_range = dst.numel() + dst_to_exclude.numel()
    val = src * dst_max_range + dst
    val_to_exclude = src_to_exclude * dst_max_range + dst_to_exclude
    # 3. Use torch.isin to get the indices of edges to keep.
    mask = ~torch.isin(val, val_to_exclude)
    return torch.nonzero(mask, as_tuple=True)[0]


def _slice_subgraph(subgraph: SampledSubgraph, index: torch.Tensor):
    """Slice the subgraph according to the index."""

    def _index_select(obj, index):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj[index]
        if isinstance(obj, tuple):
            return tuple(_index_select(v, index) for v in obj)
        # Handle the case when obj is a dictionary.
        assert isinstance(obj, dict)
        assert isinstance(index, dict)
        ret = {}
        for k, v in obj.items():
            ret[k] = _index_select(v, index[k])
        return ret

    return (
        _index_select(subgraph.node_pairs, index),
        subgraph.reverse_column_node_ids,
        subgraph.reverse_row_node_ids,
        _index_select(subgraph.reverse_edge_ids, index),
    )
