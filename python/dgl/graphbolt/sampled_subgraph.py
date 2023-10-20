"""Graphbolt sampled subgraph."""
# pylint: disable= invalid-name
from typing import Dict, Tuple, Union

import torch

from .base import etype_str_to_tuple, isin


__all__ = ["SampledSubgraph"]


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
    def original_column_node_ids(
        self,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns corresponding reverse column node ids the original graph.
        Column's reverse node ids in the original graph. A graph structure
        can be treated as a coordinated row and column pair, and this is
        the mapped ids of the column.
        - If `original_column_node_ids` is a tensor: It represents the
            original node ids.
        - If `original_column_node_ids` is a dictionary: The keys should be
            node type and the values should be corresponding original
            heterogeneous node ids.
        If present, it means column IDs are compacted, and `node_pairs`
        column IDs match these compacted ones.
        """
        return None

    @property
    def original_row_node_ids(
        self,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns corresponding reverse row node ids the original graph.
        Row's reverse node ids in the original graph. A graph structure
        can be treated as a coordinated row and column pair, and this is
        the mapped ids of the row.
        - If `original_row_node_ids` is a tensor: It represents the
            original node ids.
        - If `original_row_node_ids` is a dictionary: The keys should be
            node type and the values should be corresponding original
            heterogeneous node ids.
        If present, it means row IDs are compacted, and `node_pairs`
        row IDs match these compacted ones."""
        return None

    @property
    def original_edge_ids(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns corresponding reverse edge ids the original graph.
        Reverse edge ids in the original graph. This is useful when edge
            features are needed.
            - If `original_edge_ids` is a tensor: It represents the
                original edge ids.
            - If `original_edge_ids` is a dictionary: The keys should be
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
        assume_num_node_within_int32: bool = True,
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
        assume_num_node_within_int32: bool
            If True, assumes the value of node IDs in the provided `edges` fall
            within the int32 range, which can significantly enhance computation
            speed. Default: True

        Returns
        -------
        SampledSubgraph
           An instance of a class that inherits from `SampledSubgraph`.

        Examples
        --------
        >>> node_pairs = {"A:relation:B": (torch.tensor([0, 1, 2]),
        ...     torch.tensor([0, 1, 2]))}
        >>> original_column_node_ids = {'B': torch.tensor([10, 11, 12])}
        >>> original_row_node_ids = {'A': torch.tensor([13, 14, 15])}
        >>> original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
        >>> subgraph = gb.SampledSubgraphImpl(
        ...     node_pairs=node_pairs,
        ...     original_column_node_ids=original_column_node_ids,
        ...     original_row_node_ids=original_row_node_ids,
        ...     original_edge_ids=original_edge_ids
        ... )
        >>> edges_to_exclude = (torch.tensor([14, 15]), torch.tensor([11, 12]))
        >>> result = subgraph.exclude_edges(edges_to_exclude)
        >>> print(result.node_pairs)
        {"A:relation:B": (tensor([0]), tensor([0]))}
        >>> print(result.original_column_node_ids)
        {'B': tensor([10, 11, 12])}
        >>> print(result.original_row_node_ids)
        {'A': tensor([13, 14, 15])}
        >>> print(result.original_edge_ids)
        {"A:relation:B": tensor([19])}
        """
        # TODO: Add support for value > in32, then remove this line.
        assert (
            assume_num_node_within_int32
        ), "Values > int32 are not supported yet."
        assert isinstance(self.node_pairs, tuple) == isinstance(edges, tuple), (
            "The sampled subgraph and the edges to exclude should be both "
            "homogeneous or both heterogeneous."
        )
        # Get type of calling class.
        calling_class = type(self)

        # Three steps to exclude edges:
        # 1. Convert the node pairs to the original ids if they are compacted.
        # 2. Exclude the edges and get the index of the edges to keep.
        # 3. Slice the subgraph according to the index.
        if isinstance(self.node_pairs, tuple):
            reverse_edges = _to_reverse_ids(
                self.node_pairs,
                self.original_row_node_ids,
                self.original_column_node_ids,
            )
            index = _exclude_homo_edges(
                reverse_edges, edges, assume_num_node_within_int32
            )
            return calling_class(*_slice_subgraph(self, index))
        else:
            index = {}
            for etype, pair in self.node_pairs.items():
                src_type, _, dst_type = etype_str_to_tuple(etype)
                original_row_node_ids = (
                    None
                    if self.original_row_node_ids is None
                    else self.original_row_node_ids.get(src_type)
                )
                original_column_node_ids = (
                    None
                    if self.original_column_node_ids is None
                    else self.original_column_node_ids.get(dst_type)
                )
                reverse_edges = _to_reverse_ids(
                    pair,
                    original_row_node_ids,
                    original_column_node_ids,
                )
                index[etype] = _exclude_homo_edges(
                    reverse_edges,
                    edges.get(etype),
                    assume_num_node_within_int32,
                )
            return calling_class(*_slice_subgraph(self, index))


def _to_reverse_ids(node_pair, original_row_node_ids, original_column_node_ids):
    u, v = node_pair
    if original_row_node_ids is not None:
        u = original_row_node_ids[u]
    if original_column_node_ids is not None:
        v = original_column_node_ids[v]
    return (u, v)


def _relabel_two_arrays(lhs_array, rhs_array):
    """Relabel two arrays into a consecutive range starting from 0."""
    concated = torch.cat([lhs_array, rhs_array])
    _, mapping = torch.unique(concated, return_inverse=True)
    return mapping[: lhs_array.numel()], mapping[lhs_array.numel() :]


def _exclude_homo_edges(edges, edges_to_exclude, assume_num_node_within_int32):
    """Return the indices of edges that are not in edges_to_exclude."""
    if assume_num_node_within_int32:
        val = edges[0] << 32 | edges[1]
        val_to_exclude = edges_to_exclude[0] << 32 | edges_to_exclude[1]
    else:
        # TODO: Add support for value > int32.
        raise NotImplementedError(
            "Values out of range int32 are not supported yet"
        )
    mask = ~isin(val, val_to_exclude)
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
        subgraph.original_column_node_ids,
        subgraph.original_row_node_ids,
        _index_select(subgraph.original_edge_ids, index),
    )
