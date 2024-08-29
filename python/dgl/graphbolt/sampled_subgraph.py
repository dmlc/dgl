"""Graphbolt sampled subgraph."""

# pylint: disable= invalid-name
from typing import Dict, NamedTuple, Tuple, Union

import torch

from .base import (
    apply_to,
    CSCFormatBase,
    etype_str_to_tuple,
    expand_indptr,
    is_object_pinned,
    isin,
)

from .internal_utils import recursive_apply


__all__ = ["SampledSubgraph"]


class _ExcludeEdgesWaiter:
    def __init__(self, sampled_subgraph, index):
        self.sampled_subgraph = sampled_subgraph
        self.index = index

    def wait(self):
        """Returns the stored value when invoked."""
        sampled_subgraph = self.sampled_subgraph
        index = self.index
        # Ensure there is no memory leak.
        self.sampled_subgraph = self.index = None

        if isinstance(index, dict):
            for k in list(index.keys()):
                index[k] = index[k].wait()
        else:
            index = index.wait()

        return type(sampled_subgraph)(*_slice_subgraph(sampled_subgraph, index))


class PyGLayerData(NamedTuple):
    """A named tuple class to represent homogenous inputs to a PyG model layer.
    The fields are x (input features), edge_index and size
    (source and destination sizes).
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    size: Tuple[int, int]


class PyGLayerHeteroData(NamedTuple):
    """A named tuple class to represent heterogenous inputs to a PyG model
    layer. The fields are x (input features), edge_index and size
    (source and destination sizes), and all fields are dictionaries.
    """

    x: Dict[str, torch.Tensor]
    edge_index: Dict[str, torch.Tensor]
    size: Dict[str, Tuple[int, int]]


class SampledSubgraph:
    r"""An abstract class for sampled subgraph. In the context of a
    heterogeneous graph, each field should be of `Dict` type. Otherwise,
    for homogeneous graphs, each field should correspond to its respective
    value type."""

    @property
    def sampled_csc(
        self,
    ) -> Union[CSCFormatBase, Dict[str, CSCFormatBase],]:
        """Returns the node pairs representing edges in csc format.
          - If `sampled_csc` is a CSCFormatBase: It should be in the csc
            format. `indptr` stores the index in the data array where each
            column starts. `indices` stores the row indices of the non-zero
            elements.
          - If `sampled_csc` is a dictionary: The keys should be edge type and
            the values should be corresponding node pairs. The ids inside is
            heterogeneous ids.

        Examples
        --------
        1. Homogeneous graph.

        >>> import dgl.graphbolt as gb
        >>> import torch
        >>> sampled_csc = gb.CSCFormatBase(
        ...     indptr=torch.tensor([0, 1, 2, 3]),
        ...     indices=torch.tensor([0, 1, 2]))
        >>> print(sampled_csc)
        CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                    indices=tensor([0, 1, 2]),
        )

        2. Heterogeneous graph.

        >>> sampled_csc = {"A:relation:B": gb.CSCFormatBase(
        ...     indptr=torch.tensor([0, 1, 2, 3]),
        ...     indices=torch.tensor([0, 1, 2]))}
        >>> print(sampled_csc)
        {'A:relation:B': CSCFormatBase(indptr=tensor([0, 1, 2, 3]),
                    indices=tensor([0, 1, 2]),
        )}
        """
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
        If present, it means column IDs are compacted, and `sampled_csc`
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
          - If `original_row_node_ids` is a tensor: It represents the original
            node ids.
          - If `original_row_node_ids` is a dictionary: The keys should be node
            type and the values should be corresponding original heterogeneous
            node ids.
        If present, it means row IDs are compacted, and `sampled_csc`
        row IDs match these compacted ones."""
        return None

    @property
    def original_edge_ids(self) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns corresponding reverse edge ids the original graph.
        Reverse edge ids in the original graph. This is useful when edge
        features are needed.
          - If `original_edge_ids` is a tensor: It represents the original edge
            ids.
          - If `original_edge_ids` is a dictionary: The keys should be edge
            type and the values should be corresponding original heterogeneous
            edge ids.
        """
        return None

    def exclude_edges(
        self,
        edges: Union[
            Dict[str, torch.Tensor],
            torch.Tensor,
        ],
        assume_num_node_within_int32: bool = True,
        async_op: bool = False,
    ):
        r"""Exclude edges from the sampled subgraph.

        This function can be used with sampled subgraphs, regardless of
        whether they have compacted row/column nodes or not. If the original
        subgraph has compacted row or column nodes, the corresponding row or
        column nodes in the returned subgraph will also be compacted.

        Parameters
        ----------
        self : SampledSubgraph
            The sampled subgraph.
        edges : Union[torch.Tensor, Dict[str, torch.Tensor]]
            Edges to exclude. If sampled subgraph is homogeneous, then `edges`
            should be a N*2 tensors representing the edges to exclude. If
            sampled subgraph is heterogeneous, then `edges` should be a
            dictionary of edge types and the corresponding edges to exclude.
        assume_num_node_within_int32: bool
            If True, assumes the value of node IDs in the provided `edges` fall
            within the int32 range, which can significantly enhance computation
            speed. Default: True
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

        Returns
        -------
        SampledSubgraph
            An instance of a class that inherits from `SampledSubgraph`.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> import torch
        >>> sampled_csc = {"A:relation:B": gb.CSCFormatBase(
        ...     indptr=torch.tensor([0, 1, 2, 3]),
        ...     indices=torch.tensor([0, 1, 2]))}
        >>> original_column_node_ids = {"B": torch.tensor([10, 11, 12])}
        >>> original_row_node_ids = {"A": torch.tensor([13, 14, 15])}
        >>> original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
        >>> subgraph = gb.SampledSubgraphImpl(
        ...     sampled_csc=sampled_csc,
        ...     original_column_node_ids=original_column_node_ids,
        ...     original_row_node_ids=original_row_node_ids,
        ...     original_edge_ids=original_edge_ids
        ... )
        >>> edges_to_exclude = {"A:relation:B": torch.tensor([[14, 11], [15, 12]])}
        >>> result = subgraph.exclude_edges(edges_to_exclude)
        >>> print(result.sampled_csc)
        {'A:relation:B': CSCFormatBase(indptr=tensor([0, 1, 1, 1]),
                    indices=tensor([0]),
        )}
        >>> print(result.original_column_node_ids)
        {'B': tensor([10, 11, 12])}
        >>> print(result.original_row_node_ids)
        {'A': tensor([13, 14, 15])}
        >>> print(result.original_edge_ids)
        {'A:relation:B': tensor([19])}
        """
        # TODO: Add support for value > in32, then remove this line.
        assert (
            assume_num_node_within_int32
        ), "Values > int32 are not supported yet."
        assert (isinstance(self.sampled_csc, CSCFormatBase)) == isinstance(
            edges, torch.Tensor
        ), (
            "The sampled subgraph and the edges to exclude should be both "
            "homogeneous or both heterogeneous."
        )
        # Get type of calling class.
        calling_class = type(self)

        # Three steps to exclude edges:
        # 1. Convert the node pairs to the original ids if they are compacted.
        # 2. Exclude the edges and get the index of the edges to keep.
        # 3. Slice the subgraph according to the index.
        if isinstance(self.sampled_csc, CSCFormatBase):
            reverse_edges = _to_reverse_ids(
                self.sampled_csc,
                self.original_row_node_ids,
                self.original_column_node_ids,
            )
            index = _exclude_homo_edges(
                reverse_edges, edges, assume_num_node_within_int32, async_op
            )
        else:
            index = {}
            for etype, pair in self.sampled_csc.items():
                if etype not in edges:
                    # No edges need to be excluded.
                    index[etype] = None
                    continue
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
                    edges[etype],
                    assume_num_node_within_int32,
                    async_op,
                )
        if async_op:
            return _ExcludeEdgesWaiter(self, index)
        else:
            return calling_class(*_slice_subgraph(self, index))

    def to_pyg(
        self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> Union[PyGLayerData, PyGLayerHeteroData]:
        """
        Process layer inputs so that they can be consumed by a PyG model layer.

        Parameters
        ----------
        x : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The input node features to the GNN layer.

        Returns
        -------
        Union[PyGLayerData, PyGLayerHeteroData]
            A named tuple class with `x`, `edge_index` and `size` fields.
            Typically, a PyG GNN layer's forward method will accept these as
            arguments.
        """
        if isinstance(x, torch.Tensor):
            # Homogenous
            src = self.sampled_csc.indices
            dst = expand_indptr(
                self.sampled_csc.indptr,
                dtype=src.dtype,
                output_size=src.size(0),
            )
            edge_index = torch.stack([src, dst], dim=0).long()
            dst_size = self.sampled_csc.indptr.size(0) - 1
            # h and h[:dst_size] correspond to source and destination features resp.
            return PyGLayerData(
                (x, x[:dst_size]), edge_index, (x.size(0), dst_size)
            )
        else:
            # Heterogenous
            x_dst_dict = {}
            edge_index_dict = {}
            sizes_dict = {}
            for etype, sampled_csc in self.sampled_csc.items():
                src = sampled_csc.indices
                dst = expand_indptr(
                    sampled_csc.indptr,
                    dtype=src.dtype,
                    output_size=src.size(0),
                )
                edge_index = torch.stack([src, dst], dim=0).long()
                dst_size = sampled_csc.indptr.size(0) - 1
                # h and h[:dst_size] correspond to source and destination features resp.
                src_ntype, _, dst_ntype = etype_str_to_tuple(etype)
                x_dst_dict[dst_ntype] = x[dst_ntype][:dst_size]
                edge_index_dict[etype] = edge_index
                sizes_dict[etype] = (x[src_ntype].size(0), dst_size)

            return PyGLayerHeteroData(
                (x, x_dst_dict), edge_index_dict, sizes_dict
            )

    def to(
        self, device: torch.device, non_blocking=False
    ) -> None:  # pylint: disable=invalid-name
        """Copy `SampledSubgraph` to the specified device using reflection."""

        for attr in dir(self):
            # Only copy member variables.
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                setattr(
                    self,
                    attr,
                    recursive_apply(
                        getattr(self, attr),
                        apply_to,
                        device,
                        non_blocking=non_blocking,
                    ),
                )

        return self

    def pin_memory(self):
        """Copy `SampledSubgraph` to the pinned memory using reflection."""

        return self.to("pinned")

    def is_pinned(self) -> bool:
        """Check whether `SampledSubgraph` is pinned using reflection."""

        return is_object_pinned(self)


def _to_reverse_ids(node_pair, original_row_node_ids, original_column_node_ids):
    indptr = node_pair.indptr
    indices = node_pair.indices
    if original_row_node_ids is not None:
        indices = torch.index_select(
            original_row_node_ids, dim=0, index=indices
        )
    indptr = expand_indptr(
        indptr, indices.dtype, original_column_node_ids, len(indices)
    )
    return (indices, indptr)


def _relabel_two_arrays(lhs_array, rhs_array):
    """Relabel two arrays into a consecutive range starting from 0."""
    concated = torch.cat([lhs_array, rhs_array])
    _, mapping = torch.unique(concated, return_inverse=True)
    return mapping[: lhs_array.numel()], mapping[lhs_array.numel() :]


def _exclude_homo_edges(
    edges: Tuple[torch.Tensor, torch.Tensor],
    edges_to_exclude: torch.Tensor,
    assume_num_node_within_int32: bool,
    async_op: bool,
):
    """Return the indices of edges to be included."""
    if assume_num_node_within_int32:
        val = edges[0].long() << 32 | edges[1].long()
        edges_to_exclude_trans = edges_to_exclude.T
        val_to_exclude = (
            edges_to_exclude_trans[0].long() << 32
            | edges_to_exclude_trans[1].long()
        )
    else:
        # TODO: Add support for value > int32.
        raise NotImplementedError(
            "Values out of range int32 are not supported yet"
        )
    if async_op:
        return torch.ops.graphbolt.is_not_in_index_async(val, val_to_exclude)
    else:
        mask = ~isin(val, val_to_exclude)
        return torch.nonzero(mask, as_tuple=True)[0]


def _slice_subgraph(subgraph: SampledSubgraph, index: torch.Tensor):
    """Slice the subgraph according to the index."""

    def _index_select(obj, index):
        if obj is None:
            return None
        if index is None:
            return obj
        if isinstance(obj, CSCFormatBase):
            new_indices = obj.indices[index]
            new_indptr = torch.searchsorted(index, obj.indptr)
            return CSCFormatBase(
                indptr=new_indptr,
                indices=new_indices,
            )
        if isinstance(obj, torch.Tensor):
            return obj[index]
        # Handle the case when obj is a dictionary.
        assert isinstance(obj, dict)
        assert isinstance(index, dict)
        ret = {}
        for k, v in obj.items():
            ret[k] = _index_select(v, index[k])
        return ret

    return (
        _index_select(subgraph.sampled_csc, index),
        subgraph.original_column_node_ids,
        subgraph.original_row_node_ids,
        _index_select(subgraph.original_edge_ids, index),
    )
