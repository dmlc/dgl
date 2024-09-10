"""CSC format sampling graph."""

import textwrap

# pylint: disable= invalid-name
from typing import Dict, Optional, Union

import torch

from ..base import etype_str_to_tuple, etype_tuple_to_str, ORIGINAL_EDGE_ID
from ..internal_utils import gb_warning, is_wsl, recursive_apply
from ..sampling_graph import SamplingGraph
from .gpu_graph_cache import GPUGraphCache
from .sampled_subgraph_impl import CSCFormatBase, SampledSubgraphImpl


__all__ = [
    "FusedCSCSamplingGraph",
    "fused_csc_sampling_graph",
    "load_from_shared_memory",
    "from_dglgraph",
]


class _SampleNeighborsWaiter:
    def __init__(
        self, fn, future, seed_offsets, fetching_original_edge_ids_is_optional
    ):
        self.fn = fn
        self.future = future
        self.seed_offsets = seed_offsets
        self.fetching_original_edge_ids_is_optional = (
            fetching_original_edge_ids_is_optional
        )

    def wait(self):
        """Returns the stored value when invoked."""
        fn = self.fn
        C_sampled_subgraph = self.future.wait()
        seed_offsets = self.seed_offsets
        fetching_original_edge_ids_is_optional = (
            self.fetching_original_edge_ids_is_optional
        )
        # Ensure there is no memory leak.
        self.fn = self.future = self.seed_offsets = None
        self.fetching_original_edge_ids_is_optional = None
        return fn(
            C_sampled_subgraph,
            seed_offsets,
            fetching_original_edge_ids_is_optional,
        )


class FusedCSCSamplingGraph(SamplingGraph):
    r"""A sampling graph in CSC format."""

    def __repr__(self):
        final_str = (
            "{classname}(csc_indptr={csc_indptr},\n"
            "indices={indices},\n"
            "{metadata})"
        )

        classname_str = self.__class__.__name__
        csc_indptr_str = str(self.csc_indptr)
        indices_str = str(self.indices)
        meta_str = f"total_num_nodes={self.total_num_nodes}, num_edges={self.num_edges},"
        if self.node_type_offset is not None:
            meta_str += f"\nnode_type_offset={self.node_type_offset},"
        if self.type_per_edge is not None:
            meta_str += f"\ntype_per_edge={self.type_per_edge},"
        if self.node_type_to_id is not None:
            meta_str += f"\nnode_type_to_id={self.node_type_to_id},"
        if self.edge_type_to_id is not None:
            meta_str += f"\nedge_type_to_id={self.edge_type_to_id},"
        if self.node_attributes is not None:
            meta_str += f"\nnode_attributes={self.node_attributes},"
        if self.edge_attributes is not None:
            meta_str += f"\nedge_attributes={self.edge_attributes},"

        final_str = final_str.format(
            classname=classname_str,
            csc_indptr=csc_indptr_str,
            indices=indices_str,
            metadata=meta_str,
        )
        return textwrap.indent(
            final_str, " " * (len(classname_str) + 1)
        ).strip()

    def __init__(
        self,
        c_csc_graph: torch.ScriptObject,
    ):
        super().__init__()
        self._c_csc_graph = c_csc_graph

    def __del__(self):
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        if hasattr(self, "_is_inplace_pinned"):
            for tensor in self._is_inplace_pinned:
                assert self._inplace_unpinner(tensor.data_ptr()) == 0

    @property
    def total_num_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        int
            The number of rows in the dense format.
        """
        return self._c_csc_graph.num_nodes()

    @property
    def total_num_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
        return self._c_csc_graph.num_edges()

    @property
    def num_nodes(self) -> Union[int, Dict[str, int]]:
        """The number of nodes in the graph.
        - If the graph is homogenous, returns an integer.
        - If the graph is heterogenous, returns a dictionary.

        Returns
        -------
        Union[int, Dict[str, int]]
            The number of nodes. Integer indicates the total nodes number of a
            homogenous graph; dict indicates nodes number per node types of a
            heterogenous graph.

        Examples
        --------
        >>> import dgl.graphbolt as gb, torch
        >>> total_num_nodes = 5
        >>> total_num_edges = 12
        >>> ntypes = {"N0": 0, "N1": 1}
        >>> etypes = {"N0:R0:N0": 0, "N0:R1:N1": 1,
        ...     "N1:R2:N0": 2, "N1:R3:N1": 3}
        >>> indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
        >>> indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
        >>> node_type_offset = torch.LongTensor([0, 2, 5])
        >>> type_per_edge = torch.LongTensor(
        ...     [0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
        >>> graph = gb.fused_csc_sampling_graph(indptr, indices,
        ...     node_type_offset=node_type_offset,
        ...     type_per_edge=type_per_edge,
        ...     node_type_to_id=ntypes,
        ...     edge_type_to_id=etypes)
        >>> print(graph.num_nodes)
        {'N0': 2, 'N1': 3}
        """

        offset = self._node_type_offset_list

        # Homogenous.
        if offset is None or self.node_type_to_id is None:
            return self._c_csc_graph.num_nodes()

        # Heterogenous
        else:
            num_nodes_per_type = {
                _type: offset[_idx + 1] - offset[_idx]
                for _type, _idx in self.node_type_to_id.items()
            }

            return num_nodes_per_type

    @property
    def num_edges(self) -> Union[int, Dict[str, int]]:
        """The number of edges in the graph.
        - If the graph is homogenous, returns an integer.
        - If the graph is heterogenous, returns a dictionary.

        Returns
        -------
        Union[int, Dict[str, int]]
            The number of edges. Integer indicates the total edges number of a
            homogenous graph; dict indicates edges number per edge types of a
            heterogenous graph.

        Examples
        --------
        >>> import dgl.graphbolt as gb, torch
        >>> total_num_nodes = 5
        >>> total_num_edges = 12
        >>> ntypes = {"N0": 0, "N1": 1}
        >>> etypes = {"N0:R0:N0": 0, "N0:R1:N1": 1,
        ...     "N1:R2:N0": 2, "N1:R3:N1": 3}
        >>> indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
        >>> indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
        >>> node_type_offset = torch.LongTensor([0, 2, 5])
        >>> type_per_edge = torch.LongTensor(
        ...     [0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
        >>> metadata = gb.GraphMetadata(ntypes, etypes)
        >>> graph = gb.fused_csc_sampling_graph(indptr, indices, node_type_offset,
        ...     type_per_edge, None, metadata)
        >>> print(graph.num_edges)
        {'N0:R0:N0': 2, 'N0:R1:N1': 1, 'N1:R2:N0': 2, 'N1:R3:N1': 3}
        """

        type_per_edge = self.type_per_edge

        # Homogenous.
        if type_per_edge is None or self.edge_type_to_id is None:
            return self._c_csc_graph.num_edges()

        # Heterogenous
        bincount = torch.bincount(type_per_edge)
        num_edges_per_type = {}
        for etype, etype_id in self.edge_type_to_id.items():
            if etype_id < len(bincount):
                num_edges_per_type[etype] = bincount[etype_id].item()
            else:
                num_edges_per_type[etype] = 0
        return num_edges_per_type

    @property
    def csc_indptr(self) -> torch.tensor:
        """Returns the indices pointer in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices pointer in the CSC graph. An integer tensor with
            shape `(total_num_nodes+1,)`.
        """
        return self._c_csc_graph.csc_indptr()

    @csc_indptr.setter
    def csc_indptr(self, csc_indptr: torch.tensor) -> None:
        """Sets the indices pointer in the CSC graph."""
        self._c_csc_graph.set_csc_indptr(csc_indptr)

    @property
    def indices(self) -> torch.tensor:
        """Returns the indices in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices in the CSC graph. An integer tensor with shape
            `(total_num_edges,)`.

        Notes
        -------
        It is assumed that edges of each node are already sorted by edge type
        ids.
        """
        return self._c_csc_graph.indices()

    @indices.setter
    def indices(self, indices: torch.tensor) -> None:
        """Sets the indices in the CSC graph."""
        self._c_csc_graph.set_indices(indices)

    @property
    def node_type_offset(self) -> Optional[torch.Tensor]:
        """Returns the node type offset tensor if present. Do not modify the
        returned tensor in place.

        Returns
        -------
        torch.Tensor or None
            If present, returns a 1D integer tensor of shape
            `(num_node_types + 1,)`. The tensor is in ascending order as nodes
            of the same type have continuous IDs, and larger node IDs are
            paired with larger node type IDs. The first value is 0 and last
            value is the number of nodes. And nodes with IDs between
            `node_type_offset_[i]~node_type_offset_[i+1]` are of type id 'i'.

        """
        return self._c_csc_graph.node_type_offset()

    @property
    def _node_type_offset_list(self) -> Optional[list]:
        """Returns the node type offset list if present.

        Returns
        -------
        list or None
            If present, returns a 1D integer list of shape
            `(num_node_types + 1,)`. The list is in ascending order as nodes
            of the same type have continuous IDs, and larger node IDs are
            paired with larger node type IDs. The first value is 0 and last
            value is the number of nodes. And nodes with IDs between
            `node_type_offset_[i]~node_type_offset_[i+1]` are of type id 'i'.

        """
        if (
            not hasattr(self, "_node_type_offset_cached_list")
            or self._node_type_offset_cached_list is None
        ):
            self._node_type_offset_cached_list = self.node_type_offset
            if self._node_type_offset_cached_list is not None:
                self._node_type_offset_cached_list = (
                    self._node_type_offset_cached_list.tolist()
                )
        return self._node_type_offset_cached_list

    @node_type_offset.setter
    def node_type_offset(
        self, node_type_offset: Optional[torch.Tensor]
    ) -> None:
        """Sets the node type offset tensor if present."""
        self._c_csc_graph.set_node_type_offset(node_type_offset)
        self._node_type_offset_cached_list = None

    @property
    def _indptr_node_type_offset_list(self) -> Optional[list]:
        """Returns the indptr node type offset list which presents the column id
        space when it does not match the global id space. It is useful when we
        slice a subgraph from another FusedCSCSamplingGraph.

        Returns
        -------
        list or None
            If present, returns a 1D integer list of shape
            `(num_node_types + 1,)`. The list is in ascending order as nodes
            of the same type have continuous IDs, and larger node IDs are
            paired with larger node type IDs. The first value is 0 and last
            value is the number of nodes. And nodes with IDs between
            `node_type_offset_[i]~node_type_offset_[i+1]` are of type id 'i'.
        """
        return (
            self._indptr_node_type_offset_list_
            if hasattr(self, "_indptr_node_type_offset_list_")
            else None
        )

    @_indptr_node_type_offset_list.setter
    def _indptr_node_type_offset_list(
        self, indptr_node_type_offset_list: Optional[torch.Tensor]
    ):
        """Sets the indptr node type offset list if present."""
        self._indptr_node_type_offset_list_ = indptr_node_type_offset_list

    @property
    def _gpu_graph_cache(self) -> Optional[GPUGraphCache]:
        return (
            self._gpu_graph_cache_
            if hasattr(self, "_gpu_graph_cache_")
            else None
        )

    @property
    def type_per_edge(self) -> Optional[torch.Tensor]:
        """Returns the edge type tensor if present.

        Returns
        -------
        torch.Tensor or None
            If present, returns a 1D integer tensor of shape (total_num_edges,)
            containing the type of each edge in the graph.
        """
        return self._c_csc_graph.type_per_edge()

    @type_per_edge.setter
    def type_per_edge(self, type_per_edge: Optional[torch.Tensor]) -> None:
        """Sets the edge type tensor if present."""
        self._c_csc_graph.set_type_per_edge(type_per_edge)

    @property
    def node_type_to_id(self) -> Optional[Dict[str, int]]:
        """Returns the node type to id dictionary if present.

        Returns
        -------
        Dict[str, int] or None
            If present, returns a dictionary mapping node type to node type
            id.
        """
        return self._c_csc_graph.node_type_to_id()

    @node_type_to_id.setter
    def node_type_to_id(
        self, node_type_to_id: Optional[Dict[str, int]]
    ) -> None:
        """Sets the node type to id dictionary if present."""
        self._c_csc_graph.set_node_type_to_id(node_type_to_id)

    @property
    def edge_type_to_id(self) -> Optional[Dict[str, int]]:
        """Returns the edge type to id dictionary if present.

        Returns
        -------
        Dict[str, int] or None
            If present, returns a dictionary mapping edge type to edge type
            id.
        """
        return self._c_csc_graph.edge_type_to_id()

    @edge_type_to_id.setter
    def edge_type_to_id(
        self, edge_type_to_id: Optional[Dict[str, int]]
    ) -> None:
        """Sets the edge type to id dictionary if present."""
        self._c_csc_graph.set_edge_type_to_id(edge_type_to_id)

    @property
    def node_attributes(self) -> Optional[Dict[str, torch.Tensor]]:
        """Returns the node attributes dictionary.

        Returns
        -------
        Dict[str, torch.Tensor] or None
            If present, returns a dictionary of node attributes. Each key
            represents the attribute's name, while the corresponding value
            holds the attribute's specific value. The length of each value
            should match the total number of nodes."
        """
        return self._c_csc_graph.node_attributes()

    @node_attributes.setter
    def node_attributes(
        self, node_attributes: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        """Sets the node attributes dictionary."""
        self._c_csc_graph.set_node_attributes(node_attributes)

    @property
    def edge_attributes(self) -> Optional[Dict[str, torch.Tensor]]:
        """Returns the edge attributes dictionary.

        Returns
        -------
        Dict[str, torch.Tensor] or None
            If present, returns a dictionary of edge attributes. Each key
            represents the attribute's name, while the corresponding value
            holds the attribute's specific value. The length of each value
            should match the total number of edges."
        """
        return self._c_csc_graph.edge_attributes()

    @edge_attributes.setter
    def edge_attributes(
        self, edge_attributes: Optional[Dict[str, torch.Tensor]]
    ) -> None:
        """Sets the edge attributes dictionary."""
        self._c_csc_graph.set_edge_attributes(edge_attributes)

    def node_attribute(self, name: str) -> Optional[torch.Tensor]:
        """Returns the node attribute tensor by name.

        Parameters
        ----------
        name: str
            The name of the node attribute.

        Returns
        -------
        torch.Tensor or None
            If present, returns the node attribute tensor.
        """
        return self._c_csc_graph.node_attribute(name)

    def add_node_attribute(self, name: str, tensor: torch.Tensor) -> None:
        """Adds node attribute tensor by name.

        Parameters
        ----------
        name: str
            The name of the node attribute.
        tensor: torch.Tensor
            The node attribute tensor.
        """
        self._c_csc_graph.add_node_attribute(name, tensor)

    def edge_attribute(self, name: str) -> Optional[torch.Tensor]:
        """Returns the edge attribute tensor by name.

        Parameters
        ----------
        name: str
            The name of the edge attribute.

        Returns
        -------
        torch.Tensor or None
            If present, returns the edge attribute tensor.
        """
        return self._c_csc_graph.edge_attribute(name)

    def add_edge_attribute(self, name: str, tensor: torch.Tensor) -> None:
        """Adds edge attribute tensor by name.

        Parameters
        ----------
        name: str
            The name of the edge attribute.
        tensor: torch.Tensor
            The edge attribute tensor.
        """
        self._c_csc_graph.add_edge_attribute(name, tensor)

    def in_subgraph(
        self,
        nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
    ) -> SampledSubgraphImpl:
        """Return the subgraph induced on the inbound edges of the given nodes.

        An in subgraph is equivalent to creating a new graph using the incoming
        edges of the given nodes. Subgraph is compacted according to the order
        of passed-in `nodes`.

        Parameters
        ----------
        nodes: torch.Tensor or Dict[str, torch.Tensor]
            IDs of the given seed nodes.
              - If `nodes` is a tensor: It means the graph is homogeneous
                graph, and ids inside are homogeneous ids.
              - If `nodes` is a dictionary: The keys should be node type and
                ids inside are heterogeneous ids.

        Returns
        -------
        SampledSubgraphImpl
            The in subgraph.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> import torch
        >>> total_num_nodes = 5
        >>> total_num_edges = 12
        >>> ntypes = {"N0": 0, "N1": 1}
        >>> etypes = {
        ...     "N0:R0:N0": 0, "N0:R1:N1": 1, "N1:R2:N0": 2, "N1:R3:N1": 3}
        >>> indptr = torch.LongTensor([0, 3, 5, 7, 9, 12])
        >>> indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1, 1, 2, 0, 3, 4])
        >>> node_type_offset = torch.LongTensor([0, 2, 5])
        >>> type_per_edge = torch.LongTensor(
        ...     [0, 0, 2, 2, 2, 1, 1, 1, 3, 1, 3, 3])
        >>> graph = gb.fused_csc_sampling_graph(indptr, indices,
        ...     node_type_offset=node_type_offset,
        ...     type_per_edge=type_per_edge,
        ...     node_type_to_id=ntypes,
        ...     edge_type_to_id=etypes)
        >>> nodes = {"N0":torch.LongTensor([1]), "N1":torch.LongTensor([1, 2])}
        >>> in_subgraph = graph.in_subgraph(nodes)
        >>> print(in_subgraph.sampled_csc)
        {'N0:R0:N0': CSCFormatBase(indptr=tensor([0, 0]),
              indices=tensor([], dtype=torch.int64),
        ), 'N0:R1:N1': CSCFormatBase(indptr=tensor([0, 1, 2]),
                    indices=tensor([1, 0]),
        ), 'N1:R2:N0': CSCFormatBase(indptr=tensor([0, 2]),
                    indices=tensor([0, 1]),
        ), 'N1:R3:N1': CSCFormatBase(indptr=tensor([0, 1, 3]),
                    indices=tensor([0, 1, 2]),
        )}
        """
        if isinstance(nodes, dict):
            nodes, _ = self._convert_to_homogeneous_nodes(nodes)
        # Ensure nodes is 1-D tensor.
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."

        _in_subgraph = self._c_csc_graph.in_subgraph(nodes)
        return self._convert_to_sampled_subgraph(_in_subgraph)

    def _convert_to_homogeneous_nodes(
        self, nodes, timestamps=None, time_windows=None
    ):
        homogeneous_nodes = []
        homogeneous_node_offsets = [0]
        homogeneous_timestamps = []
        homogeneous_time_windows = []
        offset = self._node_type_offset_list
        for ntype, ntype_id in self.node_type_to_id.items():
            ids = nodes.get(ntype, [])
            if len(ids) > 0:
                homogeneous_nodes.append(ids + offset[ntype_id])
                if timestamps is not None:
                    homogeneous_timestamps.append(timestamps[ntype])
                if time_windows is not None:
                    homogeneous_time_windows.append(time_windows[ntype])
            homogeneous_node_offsets.append(
                homogeneous_node_offsets[-1] + len(ids)
            )
        if timestamps is not None:
            homogeneous_time_windows = (
                torch.cat(homogeneous_time_windows)
                if homogeneous_time_windows
                else None
            )
            return (
                torch.cat(homogeneous_nodes),
                homogeneous_node_offsets,
                torch.cat(homogeneous_timestamps),
                homogeneous_time_windows,
            )
        return torch.cat(homogeneous_nodes), homogeneous_node_offsets

    def _convert_to_sampled_subgraph(
        self,
        C_sampled_subgraph: torch.ScriptObject,
        seed_offsets: Optional[list] = None,
        fetching_original_edge_ids_is_optional: bool = False,
    ) -> SampledSubgraphImpl:
        """An internal function used to convert a fused homogeneous sampled
        subgraph to general struct 'SampledSubgraphImpl'."""
        indptr = C_sampled_subgraph.indptr
        indices = C_sampled_subgraph.indices
        type_per_edge = C_sampled_subgraph.type_per_edge
        column = C_sampled_subgraph.original_column_node_ids
        edge_ids_in_fused_csc_sampling_graph = (
            C_sampled_subgraph.original_edge_ids
        )
        etype_offsets = C_sampled_subgraph.etype_offsets
        if etype_offsets is not None:
            etype_offsets = etype_offsets.tolist()

        has_original_eids = (
            self.edge_attributes is not None
            and ORIGINAL_EDGE_ID in self.edge_attributes
        )
        original_edge_ids = (
            (
                torch.ops.graphbolt.index_select(
                    self.edge_attributes[ORIGINAL_EDGE_ID],
                    edge_ids_in_fused_csc_sampling_graph,
                )
                if not fetching_original_edge_ids_is_optional
                or not edge_ids_in_fused_csc_sampling_graph.is_cuda
                or not self.edge_attributes[ORIGINAL_EDGE_ID].is_pinned()
                else None
            )
            if has_original_eids
            else edge_ids_in_fused_csc_sampling_graph
        )
        if type_per_edge is None and etype_offsets is None:
            # The sampled graph is already a homogeneous graph.
            sampled_csc = CSCFormatBase(indptr=indptr, indices=indices)
            if indices is not None and original_edge_ids is not None:
                # Only needed to fetch indices or original_edge_ids.
                edge_ids_in_fused_csc_sampling_graph = None
        else:
            offset = self._node_type_offset_list

            original_hetero_edge_ids = {}
            sub_indices = {}
            sub_indptr = {}
            if etype_offsets is None:
                # UVA sampling requires us to move node_type_offset to GPU.
                self.node_type_offset = self.node_type_offset.to(column.device)
                # 1. Find node types for each nodes in column.
                node_types = (
                    torch.searchsorted(
                        self.node_type_offset, column, right=True
                    )
                    - 1
                )
                for ntype, ntype_id in self.node_type_to_id.items():
                    # Get all nodes of a specific node type in column.
                    nids = torch.nonzero(node_types == ntype_id).view(-1)
                    nids_original_indptr = indptr[nids + 1]
                    for etype, etype_id in self.edge_type_to_id.items():
                        src_ntype, _, dst_ntype = etype_str_to_tuple(etype)
                        if dst_ntype != ntype:
                            continue
                        # Get all edge ids of a specific edge type.
                        eids = torch.nonzero(type_per_edge == etype_id).view(-1)
                        src_ntype_id = self.node_type_to_id[src_ntype]
                        sub_indices[etype] = (
                            indices[eids] - offset[src_ntype_id]
                        )
                        cum_edges = torch.searchsorted(
                            eids, nids_original_indptr, right=False
                        )
                        sub_indptr[etype] = torch.cat(
                            (torch.tensor([0], device=indptr.device), cum_edges)
                        )
                        original_hetero_edge_ids[etype] = original_edge_ids[
                            eids
                        ]
                sampled_hetero_edge_ids_in_fused_csc_sampling_graph = None
            else:
                sampled_hetero_edge_ids_in_fused_csc_sampling_graph = {}
                edge_offsets = [0]
                for etype, etype_id in self.edge_type_to_id.items():
                    src_ntype, _, dst_ntype = etype_str_to_tuple(etype)
                    ntype_id = self.node_type_to_id[dst_ntype]
                    edge_offsets.append(
                        edge_offsets[-1]
                        + seed_offsets[ntype_id + 1]
                        - seed_offsets[ntype_id]
                        + 1
                    )
                for etype, etype_id in self.edge_type_to_id.items():
                    src_ntype, _, dst_ntype = etype_str_to_tuple(etype)
                    ntype_id = self.node_type_to_id[dst_ntype]
                    sub_indptr[etype] = indptr[
                        edge_offsets[etype_id] : edge_offsets[etype_id + 1]
                    ]
                    sub_indices[etype] = (
                        None
                        if indices is None
                        else indices[
                            etype_offsets[etype_id] : etype_offsets[
                                etype_id + 1
                            ]
                        ]
                    )
                    original_hetero_edge_ids[etype] = (
                        None
                        if original_edge_ids is None
                        else original_edge_ids[
                            etype_offsets[etype_id] : etype_offsets[
                                etype_id + 1
                            ]
                        ]
                    )
                    if indices is None or original_edge_ids is None:
                        # Only needed to fetch indices or original edge ids.
                        sampled_hetero_edge_ids_in_fused_csc_sampling_graph[
                            etype
                        ] = edge_ids_in_fused_csc_sampling_graph[
                            etype_offsets[etype_id] : etype_offsets[
                                etype_id + 1
                            ]
                        ]

            original_edge_ids = original_hetero_edge_ids
            edge_ids_in_fused_csc_sampling_graph = (
                sampled_hetero_edge_ids_in_fused_csc_sampling_graph
            )
            sampled_csc = {
                etype: CSCFormatBase(
                    indptr=sub_indptr[etype],
                    indices=sub_indices[etype],
                )
                for etype in self.edge_type_to_id.keys()
            }
        return SampledSubgraphImpl(
            sampled_csc=sampled_csc,
            original_edge_ids=original_edge_ids,
            _edge_ids_in_fused_csc_sampling_graph=edge_ids_in_fused_csc_sampling_graph,
        )

    def sample_neighbors(
        self,
        seeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
        returning_indices_and_original_edge_ids_are_optional: bool = False,
        async_op: bool = False,
    ) -> SampledSubgraphImpl:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph.

        Parameters
        ----------
        seeds: torch.Tensor or Dict[str, torch.Tensor]
            IDs of the given seed nodes.
              - If `nodes` is a tensor: It means the graph is homogeneous
                graph, and ids inside are homogeneous ids.
              - If `nodes` is a dictionary: The keys should be node type and
                ids inside are heterogeneous ids.
        fanouts: torch.Tensor
            The number of edges to be sampled for each node with or without
            considering edge types.
              - When the length is 1, it indicates that the fanout applies to
                all neighbors of the node as a collective, regardless of the
                edge type.
              - Otherwise, the length should equal to the number of edge
                types, and each fanout value corresponds to a specific edge
                type of the nodes.
            The value of each fanout should be >= 0 or = -1.
              - When the value is -1, all neighbors (with non-zero probability,
                if weighted) will be sampled once regardless of replacement. It
                is equivalent to selecting all neighbors with non-zero
                probability when the fanout is >= the number of neighbors (and
                replace is set to false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        probs_name: str, optional
            An optional string specifying the name of an edge attribute used.
            This attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        returning_indices_and_original_edge_ids_are_optional: bool
            Boolean indicating whether it is okay for the call to this function
            to leave the indices and the original edge ids tensors
            uninitialized. In this case, it is the user's responsibility to
            gather them using _edge_ids_in_fused_csc_sampling_graph if either is
            missing.
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

        Returns
        -------
        SampledSubgraphImpl
            The sampled subgraph.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> import torch
        >>> ntypes = {"n1": 0, "n2": 1}
        >>> etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
        >>> indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
        >>> indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
        >>> node_type_offset = torch.LongTensor([0, 2, 5])
        >>> type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
        >>> graph = gb.fused_csc_sampling_graph(indptr, indices,
        ...     node_type_offset=node_type_offset,
        ...     type_per_edge=type_per_edge,
        ...     node_type_to_id=ntypes,
        ...     edge_type_to_id=etypes)
        >>> nodes = {'n1': torch.LongTensor([0]), 'n2': torch.LongTensor([0])}
        >>> fanouts = torch.tensor([1, 1])
        >>> subgraph = graph.sample_neighbors(nodes, fanouts)
        >>> print(subgraph.sampled_csc)
        {'n1:e1:n2': CSCFormatBase(indptr=tensor([0, 1]),
                    indices=tensor([0]),
        ), 'n2:e2:n1': CSCFormatBase(indptr=tensor([0, 1]),
                    indices=tensor([2]),
        )}
        """
        seed_offsets = None
        if isinstance(seeds, dict):
            seeds, seed_offsets = self._convert_to_homogeneous_nodes(seeds)
        elif seeds is None:
            seed_offsets = self._indptr_node_type_offset_list
        probs_or_mask = self.edge_attributes[probs_name] if probs_name else None
        C_sampled_subgraph = self._sample_neighbors(
            seeds,
            seed_offsets,
            fanouts,
            replace=replace,
            probs_or_mask=probs_or_mask,
            returning_indices_is_optional=returning_indices_and_original_edge_ids_are_optional,
            async_op=async_op,
        )
        if async_op:
            return _SampleNeighborsWaiter(
                self._convert_to_sampled_subgraph,
                C_sampled_subgraph,
                seed_offsets,
                returning_indices_and_original_edge_ids_are_optional,
            )
        else:
            return self._convert_to_sampled_subgraph(
                C_sampled_subgraph,
                seed_offsets,
                returning_indices_and_original_edge_ids_are_optional,
            )

    def _check_sampler_arguments(self, nodes, fanouts, probs_or_mask):
        if nodes is not None:
            assert nodes.dim() == 1, "Nodes should be 1-D tensor."
            assert nodes.dtype == self.indices.dtype, (
                f"Data type of nodes must be consistent with "
                f"indices.dtype({self.indices.dtype}), but got {nodes.dtype}."
            )
        assert fanouts.dim() == 1, "Fanouts should be 1-D tensor."
        expected_fanout_len = 1
        if self.edge_type_to_id:
            expected_fanout_len = len(self.edge_type_to_id)
        assert len(fanouts) in [
            expected_fanout_len,
            1,
        ], "Fanouts should have the same number of elements as etypes or \
            should have a length of 1."
        if fanouts.size(0) > 1:
            assert (
                self.type_per_edge is not None
            ), "To perform sampling for each edge type (when the length of \
                `fanouts` > 1), the graph must include edge type information."
        assert torch.all(
            (fanouts >= 0) | (fanouts == -1)
        ), "Fanouts should consist of values that are either -1 or \
            greater than or equal to 0."
        if probs_or_mask is not None:
            assert probs_or_mask.dim() == 1, "Probs should be 1-D tensor."
            assert (
                probs_or_mask.size(0) == self.total_num_edges
            ), "Probs should have the same number of elements as the number \
                of edges."
            assert probs_or_mask.dtype in [
                torch.bool,
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ], "Probs should have a floating-point or boolean data type."

    def _sample_neighbors(
        self,
        seeds: torch.Tensor,
        seed_offsets: Optional[list],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_or_mask: Optional[torch.Tensor] = None,
        returning_indices_is_optional: bool = False,
        async_op: bool = False,
    ) -> torch.ScriptObject:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph.

        Parameters
        ----------
        seeds: torch.Tensor
            IDs of the given seed nodes.
        seeds_offsets: list, optional
            The offsets of the given seeds,
            seeds[seed_offsets[i]: seed_offsets[i + 1]] has node type i.
        fanouts: torch.Tensor
            The number of edges to be sampled for each node with or without
            considering edge types.
              - When the length is 1, it indicates that the fanout applies to
                all neighbors of the node as a collective, regardless of the
                edge type.
              - Otherwise, the length should equal to the number of edge
                types, and each fanout value corresponds to a specific edge
                type of the nodes.
            The value of each fanout should be >= 0 or = -1.
              - When the value is -1, all neighbors (with non-zero probability,
                if weighted) will be sampled once regardless of replacement. It
                is equivalent to selecting all neighbors with non-zero
                probability when the fanout is >= the number of neighbors (and
                replace is set to false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        probs_or_mask: torch.Tensor, optional
            An optional tensor of edge attribute for probability or masks. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        returning_indices_is_optional: bool
            Boolean indicating whether it is okay for the call to this function
            to leave the indices tensor uninitialized. In this case, it is the
            user's responsibility to gather it using the edge ids.
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

        Returns
        -------
        torch.classes.graphbolt.SampledSubgraph
            The sampled C subgraph.
        """
        # Ensure nodes is 1-D tensor.
        self._check_sampler_arguments(seeds, fanouts, probs_or_mask)
        sampling_fn = (
            self._c_csc_graph.sample_neighbors_async
            if async_op
            else self._c_csc_graph.sample_neighbors
        )
        return sampling_fn(
            seeds,
            seed_offsets,
            fanouts.tolist(),
            replace,
            False,  # is_labor
            returning_indices_is_optional,
            probs_or_mask,
            None,  # random_seed, labor parameter
            0,  # seed2_contribution, labor_parameter
        )

    def sample_layer_neighbors(
        self,
        seeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
        returning_indices_and_original_edge_ids_are_optional: bool = False,
        random_seed: torch.Tensor = None,
        seed2_contribution: float = 0.0,
        async_op: bool = False,
    ) -> SampledSubgraphImpl:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph via layer-neighbor sampling from the NeurIPS 2023 paper
        `Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
        <https://proceedings.neurips.cc/paper_files/paper/2023/file/51f9036d5e7ae822da8f6d4adda1fb39-Paper-Conference.pdf>`__

        Parameters
        ----------
        seeds: torch.Tensor or Dict[str, torch.Tensor]
            IDs of the given seed nodes.
              - If `nodes` is a tensor: It means the graph is homogeneous
                graph, and ids inside are homogeneous ids.
              - If `nodes` is a dictionary: The keys should be node type and
                ids inside are heterogeneous ids.
        fanouts: torch.Tensor
            The number of edges to be sampled for each node with or without
            considering edge types.
              - When the length is 1, it indicates that the fanout applies to
                all neighbors of the node as a collective, regardless of the
                edge type.
              - Otherwise, the length should equal to the number of edge
                types, and each fanout value corresponds to a specific edge
                type of the nodes.
            The value of each fanout should be >= 0 or = -1.
              - When the value is -1, all neighbors (with non-zero probability,
                if weighted) will be sampled once regardless of replacement. It
                is equivalent to selecting all neighbors with non-zero
                probability when the fanout is >= the number of neighbors (and
                replace is set to false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        probs_name: str, optional
            An optional string specifying the name of an edge attribute. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        returning_indices_and_original_edge_ids_are_optional: bool
            Boolean indicating whether it is okay for the call to this function
            to leave the indices and the original edge ids tensors
            uninitialized. In this case, it is the user's responsibility to
            gather them using _edge_ids_in_fused_csc_sampling_graph if either is
            missing.
        random_seed: torch.Tensor, optional
            An int64 tensor with one or two elements.

            The passed random_seed makes it so that for any seed node ``s`` and
            its neighbor ``t``, the rolled random variate ``r_t`` is the same
            for any call to this function with the same random seed. When
            sampling as part of the same batch, one would want identical seeds
            so that LABOR can globally sample. One example is that for
            heterogenous graphs, there is a single random seed passed for each
            edge type. This will sample much fewer nodes compared to having
            unique random seeds for each edge type. If one called this function
            individually for each edge type for a heterogenous graph with
            different random seeds, then it would run LABOR locally for each
            edge type, resulting into a larger number of nodes being sampled.

            If this function is called without a ``random_seed``, we get the
            random seed by getting a random number from GraphBolt. Use this
            argument with identical random_seed if multiple calls to this
            function are used to sample as part of a single batch.

            If given two numbers, then the ``seed2_contribution`` argument
            determines the interpolation between the two random seeds.
        seed2_contribution: float, optional
            A float value between [0, 1) that determines the contribution of the
            second random seed, ``random_seed[-1]``, to generate the random
            variates.
        async_op: bool
            Boolean indicating whether the call is asynchronous. If so, the
            result can be obtained by calling wait on the returned future.

        Returns
        -------
        SampledSubgraphImpl
            The sampled subgraph.

        Examples
        --------
        >>> import dgl.graphbolt as gb
        >>> import torch
        >>> ntypes = {"n1": 0, "n2": 1}
        >>> etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
        >>> indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
        >>> indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
        >>> node_type_offset = torch.LongTensor([0, 2, 5])
        >>> type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
        >>> graph = gb.fused_csc_sampling_graph(indptr, indices,
        ...     node_type_offset=node_type_offset,
        ...     type_per_edge=type_per_edge,
        ...     node_type_to_id=ntypes,
        ...     edge_type_to_id=etypes)
        >>> nodes = {'n1': torch.LongTensor([0]), 'n2': torch.LongTensor([0])}
        >>> fanouts = torch.tensor([1, 1])
        >>> subgraph = graph.sample_layer_neighbors(nodes, fanouts)
        >>> print(subgraph.sampled_csc)
        {'n1:e1:n2': CSCFormatBase(indptr=tensor([0, 1]),
                    indices=tensor([0]),
        ), 'n2:e2:n1': CSCFormatBase(indptr=tensor([0, 1]),
                    indices=tensor([2]),
        )}
        """
        if random_seed is not None:
            assert (
                1 <= len(random_seed) <= 2
            ), "There should be a 1 or 2 random seeds."
            if len(random_seed) == 2:
                assert (
                    0 <= seed2_contribution <= 1
                ), "seed2_contribution should be in [0, 1]."

        seed_offsets = None
        if isinstance(seeds, dict):
            seeds, seed_offsets = self._convert_to_homogeneous_nodes(seeds)
        elif seeds is None:
            seed_offsets = self._indptr_node_type_offset_list
        probs_or_mask = self.edge_attributes[probs_name] if probs_name else None
        self._check_sampler_arguments(seeds, fanouts, probs_or_mask)
        sampling_fn = (
            self._c_csc_graph.sample_neighbors_async
            if async_op
            else self._c_csc_graph.sample_neighbors
        )
        C_sampled_subgraph = sampling_fn(
            seeds,
            seed_offsets,
            fanouts.tolist(),
            replace,
            True,  # is_labor
            returning_indices_and_original_edge_ids_are_optional,
            probs_or_mask,
            random_seed,
            seed2_contribution,
        )
        if async_op:
            return _SampleNeighborsWaiter(
                self._convert_to_sampled_subgraph,
                C_sampled_subgraph,
                seed_offsets,
                returning_indices_and_original_edge_ids_are_optional,
            )
        else:
            return self._convert_to_sampled_subgraph(
                C_sampled_subgraph,
                seed_offsets,
                returning_indices_and_original_edge_ids_are_optional,
            )

    def temporal_sample_neighbors(
        self,
        seeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        seeds_timestamp: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        seeds_pre_time_window: Optional[
            Union[torch.Tensor, Dict[str, torch.Tensor]]
        ] = None,
        probs_name: Optional[str] = None,
        node_timestamp_attr_name: Optional[str] = None,
        edge_timestamp_attr_name: Optional[str] = None,
    ) -> torch.ScriptObject:
        """Temporally Sample neighboring edges of the given nodes and return the induced
        subgraph.

        If `node_timestamp_attr_name` or `edge_timestamp_attr_name` is given,
        the sampled neighbor or edge of an seed node must have a timestamp
        that is smaller than that of the seed node.

        Parameters
        ----------
        seeds: torch.Tensor
            IDs of the given seed nodes.
        seeds_timestamp: torch.Tensor
            Timestamps of the given seed nodes.
        fanouts: torch.Tensor
            The number of edges to be sampled for each node with or without
            considering edge types.
              - When the length is 1, it indicates that the fanout applies to
                all neighbors of the node as a collective, regardless of the
                edge type.
              - Otherwise, the length should equal to the number of edge
                types, and each fanout value corresponds to a specific edge
                type of the nodes.
            The value of each fanout should be >= 0 or = -1.
              - When the value is -1, all neighbors (with non-zero probability,
                if weighted) will be sampled once regardless of replacement. It
                is equivalent to selecting all neighbors with non-zero
                probability when the fanout is >= the number of neighbors (and
                replace is set to false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        seeds_pre_time_window: torch.Tensor
            The time window of the nodes represents a period of time before
            `seeds_timestamp`. If provided, only neighbors and related
            edges whose timestamps fall within `[seeds_timestamp -
            seeds_pre_time_window, seeds_timestamp]` will be filtered.
        probs_name: str, optional
            An optional string specifying the name of an edge attribute. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        node_timestamp_attr_name: str, optional
            An optional string specifying the name of an node attribute.
        edge_timestamp_attr_name: str, optional
            An optional string specifying the name of an edge attribute.

        Returns
        -------
        SampledSubgraphImpl
            The sampled subgraph.
        """
        seed_offsets = None
        if isinstance(seeds, dict):
            (
                seeds,
                seed_offsets,
                seeds_timestamp,
                seeds_pre_time_window,
            ) = self._convert_to_homogeneous_nodes(
                seeds, seeds_timestamp, seeds_pre_time_window
            )
        elif seeds is None:
            seed_offsets = self._indptr_node_type_offset_list

        # Ensure nodes is 1-D tensor.
        probs_or_mask = self.edge_attributes[probs_name] if probs_name else None
        self._check_sampler_arguments(seeds, fanouts, probs_or_mask)
        C_sampled_subgraph = self._c_csc_graph.temporal_sample_neighbors(
            seeds,
            seed_offsets,
            seeds_timestamp,
            fanouts.tolist(),
            replace,
            False,  # is_labor
            False,  # returning_indices_is_optional
            seeds_pre_time_window,
            probs_or_mask,
            node_timestamp_attr_name,
            edge_timestamp_attr_name,
            None,  # random_seed, labor parameter
            0,  # seed2_contribution, labor_parameter
        )
        return self._convert_to_sampled_subgraph(
            C_sampled_subgraph, seed_offsets
        )

    def temporal_sample_layer_neighbors(
        self,
        seeds: Union[torch.Tensor, Dict[str, torch.Tensor]],
        seeds_timestamp: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        seeds_pre_time_window: Optional[
            Union[torch.Tensor, Dict[str, torch.Tensor]]
        ] = None,
        probs_name: Optional[str] = None,
        node_timestamp_attr_name: Optional[str] = None,
        edge_timestamp_attr_name: Optional[str] = None,
        random_seed: torch.Tensor = None,
        seed2_contribution: float = 0.0,
    ) -> torch.ScriptObject:
        """Temporally Sample neighboring edges of the given nodes and return the induced
        subgraph via layer-neighbor sampling from the NeurIPS 2023 paper
        `Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
        <https://proceedings.neurips.cc/paper_files/paper/2023/file/51f9036d5e7ae822da8f6d4adda1fb39-Paper-Conference.pdf>`__

        If `node_timestamp_attr_name` or `edge_timestamp_attr_name` is given,
        the sampled neighbor or edge of an seed node must have a timestamp
        that is smaller than that of the seed node.

        Parameters
        ----------
        seeds: torch.Tensor
            IDs of the given seed nodes.
        seeds_timestamp: torch.Tensor
            Timestamps of the given seed nodes.
        fanouts: torch.Tensor
            The number of edges to be sampled for each node with or without
            considering edge types.
              - When the length is 1, it indicates that the fanout applies to
                all neighbors of the node as a collective, regardless of the
                edge type.
              - Otherwise, the length should equal to the number of edge
                types, and each fanout value corresponds to a specific edge
                type of the nodes.
            The value of each fanout should be >= 0 or = -1.
              - When the value is -1, all neighbors (with non-zero probability,
                if weighted) will be sampled once regardless of replacement. It
                is equivalent to selecting all neighbors with non-zero
                probability when the fanout is >= the number of neighbors (and
                replace is set to false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        seeds_pre_time_window: torch.Tensor
            The time window of the nodes represents a period of time before
            `seeds_timestamp`. If provided, only neighbors and related
            edges whose timestamps fall within `[seeds_timestamp -
            seeds_pre_time_window, seeds_timestamp]` will be
            filtered.
        probs_name: str, optional
            An optional string specifying the name of an edge attribute. This
            attribute tensor should contain (unnormalized) probabilities
            corresponding to each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor, with the number of elements
            equalling the total number of edges.
        node_timestamp_attr_name: str, optional
            An optional string specifying the name of an node attribute.
        edge_timestamp_attr_name: str, optional
            An optional string specifying the name of an edge attribute.
        random_seed: torch.Tensor, optional
            An int64 tensor with one or two elements.

            The passed random_seed makes it so that for any seed node ``s`` and
            its neighbor ``t``, the rolled random variate ``r_t`` is the same
            for any call to this function with the same random seed. When
            sampling as part of the same batch, one would want identical seeds
            so that LABOR can globally sample. One example is that for
            heterogenous graphs, there is a single random seed passed for each
            edge type. This will sample much fewer nodes compared to having
            unique random seeds for each edge type. If one called this function
            individually for each edge type for a heterogenous graph with
            different random seeds, then it would run LABOR locally for each
            edge type, resulting into a larger number of nodes being sampled.

            If this function is called without a ``random_seed``, we get the
            random seed by getting a random number from GraphBolt. Use this
            argument with identical random_seed if multiple calls to this
            function are used to sample as part of a single batch.

            If given two numbers, then the ``seed2_contribution`` argument
            determines the interpolation between the two random seeds.
        seed2_contribution: float, optional
            A float value between [0, 1) that determines the contribution of the
            second random seed, ``random_seed[-1]``, to generate the random
            variates.

        Returns
        -------
        SampledSubgraphImpl
            The sampled subgraph.
        """
        seed_offsets = None
        if isinstance(seeds, dict):
            (
                seeds,
                seed_offsets,
                seeds_timestamp,
                seeds_pre_time_window,
            ) = self._convert_to_homogeneous_nodes(
                seeds, seeds_timestamp, seeds_pre_time_window
            )
        elif seeds is None:
            seed_offsets = self._indptr_node_type_offset_list

        # Ensure nodes is 1-D tensor.
        probs_or_mask = self.edge_attributes[probs_name] if probs_name else None
        self._check_sampler_arguments(seeds, fanouts, probs_or_mask)
        C_sampled_subgraph = self._c_csc_graph.temporal_sample_neighbors(
            seeds,
            seed_offsets,
            seeds_timestamp,
            fanouts.tolist(),
            replace,
            True,  # is_labor
            False,  # returning_indices_is_optional
            seeds_pre_time_window,
            probs_or_mask,
            node_timestamp_attr_name,
            edge_timestamp_attr_name,
            random_seed,
            seed2_contribution,
        )
        return self._convert_to_sampled_subgraph(
            C_sampled_subgraph, seed_offsets
        )

    def sample_negative_edges_uniform(
        self, edge_type, node_pairs, negative_ratio
    ):
        """
        Sample negative edges by randomly choosing negative source-destination
        edges according to a uniform distribution. For each edge ``(u, v)``,
        it is supposed to generate `negative_ratio` pairs of negative edges
        ``(u, v')``, where ``v'`` is chosen uniformly from all the nodes in
        the graph. ``u`` is exactly same as the corresponding positive edges.
        It returns positive edges concatenated with negative edges. In
        negative edges, negative sources are constructed from the
        corresponding positive edges.

        Parameters
        ----------
        edge_type: str
            The type of edges in the provided node_pairs. Any negative edges
            sampled will also have the same type. If set to None, it will be
            considered as a homogeneous graph.
        node_pairs : torch.Tensor
            A 2D tensors that represent the N pairs of positive edges in
            source-destination format, with 'positive' indicating that these
            edges are present in the graph. It's important to note that within
            the context of a heterogeneous graph, the ids in these tensors
            signify heterogeneous ids.
        negative_ratio: int
            The ratio of the number of negative samples to positive samples.

        Returns
        -------
        torch.Tensor
            A 2D tensors represents the N pairs of positive and negative
            source-destination node pairs. In the context of a heterogeneous
            graph, both the input nodes and the selected nodes are represented
            by heterogeneous IDs, and the formed edges are of the input type
            `edge_type`. Note that negative refers to false negatives, which
            means the edge could be present or not present in the graph.
        """
        if edge_type:
            _, _, dst_ntype = etype_str_to_tuple(edge_type)
            max_node_id = self.num_nodes[dst_ntype]
        else:
            max_node_id = self.total_num_nodes
        pos_src = node_pairs[:, 0]
        num_negative = node_pairs.shape[0] * negative_ratio
        negative_seeds = (
            torch.cat(
                (
                    pos_src.repeat_interleave(negative_ratio),
                    torch.randint(
                        0,
                        max_node_id,
                        (num_negative,),
                        dtype=node_pairs.dtype,
                        device=node_pairs.device,
                    ),
                ),
            )
            .view(2, num_negative)
            .T
        )
        seeds = torch.cat((node_pairs, negative_seeds))
        return seeds

    def copy_to_shared_memory(self, shared_memory_name: str):
        """Copy the graph to shared memory.

        Parameters
        ----------
        shared_memory_name : str
            Name of the shared memory.

        Returns
        -------
        FusedCSCSamplingGraph
            The copied FusedCSCSamplingGraph object on shared memory.
        """
        return FusedCSCSamplingGraph(
            self._c_csc_graph.copy_to_shared_memory(shared_memory_name),
        )

    def _apply_to_members(self, fn):
        """Apply passed fn to all members of `FusedCSCSamplingGraph`."""
        self.csc_indptr = recursive_apply(self.csc_indptr, fn)
        self.indices = recursive_apply(self.indices, fn)
        self.node_type_offset = recursive_apply(self.node_type_offset, fn)
        self.type_per_edge = recursive_apply(self.type_per_edge, fn)
        self.node_attributes = recursive_apply(self.node_attributes, fn)
        self.edge_attributes = recursive_apply(self.edge_attributes, fn)

        return self

    def to(self, device: torch.device) -> None:  # pylint: disable=invalid-name
        """Copy `FusedCSCSamplingGraph` to the specified device."""

        def _to(x):
            return x.to(device) if hasattr(x, "to") else x

        def _pin(x):
            return x.pin_memory() if hasattr(x, "pin_memory") else x

        # Create a copy of self.
        self2 = fused_csc_sampling_graph(
            self.csc_indptr,
            self.indices,
            self.node_type_offset,
            self.type_per_edge,
            self.node_type_to_id,
            self.edge_type_to_id,
            self.node_attributes,
            self.edge_attributes,
        )
        return self2._apply_to_members(_pin if device == "pinned" else _to)

    def pin_memory_(self):
        """Copy `FusedCSCSamplingGraph` to the pinned memory in-place. Returns
        the same object modified in-place."""
        if is_wsl():
            gb_warning(
                "In place pinning is not supported on WSL. "
                "Returning the out of place pinned `FusedCSCSamplingGraph`."
            )
            return self.to("pinned")
        # torch.Tensor.pin_memory() is not an inplace operation. To make it
        # truly in-place, we need to use cudaHostRegister. Then, we need to use
        # cudaHostUnregister to unpin the tensor in the destructor.
        # https://github.com/pytorch/pytorch/issues/32167#issuecomment-753551842
        cudart = torch.cuda.cudart()
        if not hasattr(self, "_is_inplace_pinned"):
            self._is_inplace_pinned = set()

        def _pin(x):
            if hasattr(x, "pin_memory_"):
                x.pin_memory_()
            elif (
                isinstance(x, torch.Tensor)
                and not x.is_pinned()
                and x.device.type == "cpu"
            ):
                assert (
                    x.is_contiguous()
                ), "Tensor pinning is only supported for contiguous tensors."
                assert (
                    cudart.cudaHostRegister(
                        x.data_ptr(), x.numel() * x.element_size(), 0
                    )
                    == 0
                )

                self._is_inplace_pinned.add(x)
                self._inplace_unpinner = cudart.cudaHostUnregister

            return x

        return self._apply_to_members(_pin)

    def _initialize_gpu_graph_cache(
        self,
        num_gpu_cached_edges: int,
        gpu_cache_threshold: int,
        prob_name: Optional[str] = None,
    ):
        "Construct a GPUGraphCache given the cache parameters."
        num_gpu_cached_edges = min(num_gpu_cached_edges, self.total_num_edges)
        dtypes = [self.indices.dtype]
        if self.type_per_edge is not None:
            dtypes.append(self.type_per_edge.dtype)
        has_original_edge_ids = False
        if self.edge_attributes is not None:
            probs_or_mask = self.edge_attributes.get(prob_name, None)
            if probs_or_mask is not None:
                dtypes.append(probs_or_mask.dtype)
            original_edge_ids = self.edge_attributes.get(ORIGINAL_EDGE_ID, None)
            if original_edge_ids is not None:
                dtypes.append(original_edge_ids.dtype)
                has_original_edge_ids = True
        self._gpu_graph_cache_ = GPUGraphCache(
            num_gpu_cached_edges,
            gpu_cache_threshold,
            self.csc_indptr.dtype,
            dtypes,
            has_original_edge_ids,
        )


def fused_csc_sampling_graph(
    csc_indptr: torch.Tensor,
    indices: torch.Tensor,
    node_type_offset: Optional[torch.tensor] = None,
    type_per_edge: Optional[torch.tensor] = None,
    node_type_to_id: Optional[Dict[str, int]] = None,
    edge_type_to_id: Optional[Dict[str, int]] = None,
    node_attributes: Optional[Dict[str, torch.tensor]] = None,
    edge_attributes: Optional[Dict[str, torch.tensor]] = None,
) -> FusedCSCSamplingGraph:
    """Create a FusedCSCSamplingGraph object from a CSC representation.

    Parameters
    ----------
    csc_indptr : torch.Tensor
        Pointer to the start of each row in the `indices`. An integer tensor
        with shape `(total_num_nodes+1,)`.
    indices : torch.Tensor
        Column indices of the non-zero elements in the CSC graph. An integer
        tensor with shape `(total_num_edges,)`.
    node_type_offset : Optional[torch.tensor], optional
        Offset of node types in the graph, by default None.
    type_per_edge : Optional[torch.tensor], optional
        Type ids of each edge in the graph, by default None. If provided, it is
        required that the edge types in each vertex neighborhood are in sorted
        order. To be more precise, For each i in [0, csc_indptr.size(0) - 1),
        `type_per_edge[indptr[i]: indptr[i + 1]]` is expected to be
        monotonically nondecreasing.
    node_type_to_id : Optional[Dict[str, int]], optional
        Map node types to ids, by default None.
    edge_type_to_id : Optional[Dict[str, int]], optional
        Map edge types to ids, by default None.
    node_attributes: Optional[Dict[str, torch.tensor]], optional
        Node attributes of the graph, by default None.
    edge_attributes: Optional[Dict[str, torch.tensor]], optional
        Edge attributes of the graph, by default None.

    Returns
    -------
    FusedCSCSamplingGraph
        The created FusedCSCSamplingGraph object.

    Examples
    --------
    >>> ntypes = {'n1': 0, 'n2': 1, 'n3': 2}
    >>> etypes = {'n1:e1:n2': 0, 'n1:e2:n3': 1}
    >>> csc_indptr = torch.tensor([0, 2, 5, 7, 8])
    >>> indices = torch.tensor([1, 3, 0, 1, 2, 0, 3, 2])
    >>> node_type_offset = torch.tensor([0, 1, 2, 4])
    >>> type_per_edge = torch.tensor([0, 1, 0, 1, 1, 0, 0, 0])
    >>> graph = graphbolt.fused_csc_sampling_graph(csc_indptr, indices,
    ...         node_type_offset=node_type_offset,
    ...         type_per_edge=type_per_edge,
    ...         node_type_to_id=ntypes, edge_type_to_id=etypes,
    ...         node_attributes=None, edge_attributes=None,)
    >>> print(graph)
    FusedCSCSamplingGraph(csc_indptr=tensor([0, 2, 5, 7, 8]),
                          indices=tensor([1, 3, 0, 1, 2, 0, 3, 2]),
                          total_num_nodes=4, num_edges={'n1:e1:n2': 5, 'n1:e2:n3': 3},
                          node_type_offset=tensor([0, 1, 2, 4]),
                          type_per_edge=tensor([0, 1, 0, 1, 1, 0, 0, 0]),
                          node_type_to_id={'n1': 0, 'n2': 1, 'n3': 2},
                          edge_type_to_id={'n1:e1:n2': 0, 'n1:e2:n3': 1},)
    """
    if node_type_to_id is not None and edge_type_to_id is not None:
        node_types = list(node_type_to_id.keys())
        edge_types = list(edge_type_to_id.keys())
        node_type_ids = list(node_type_to_id.values())
        edge_type_ids = list(edge_type_to_id.values())

        # Validate node_type_to_id.
        assert all(
            isinstance(x, str) for x in node_types
        ), "Node type name should be string."
        assert all(
            isinstance(x, int) for x in node_type_ids
        ), "Node type id should be int."
        assert len(node_type_ids) == len(
            set(node_type_ids)
        ), "Multiple node types shoud not be mapped to a same id."
        # Validate edge_type_to_id.
        for edge_type in edge_types:
            src, edge, dst = etype_str_to_tuple(edge_type)
            assert isinstance(edge, str), "Edge type name should be string."
            assert (
                src in node_types
            ), f"Unrecognized node type {src} in edge type {edge_type}"
            assert (
                dst in node_types
            ), f"Unrecognized node type {dst} in edge type {edge_type}"
        assert all(
            isinstance(x, int) for x in edge_type_ids
        ), "Edge type id should be int."
        assert len(edge_type_ids) == len(
            set(edge_type_ids)
        ), "Multiple edge types shoud not be mapped to a same id."

        if node_type_offset is not None:
            assert len(node_type_to_id) + 1 == node_type_offset.size(
                0
            ), "node_type_offset length should be |ntypes| + 1."
    return FusedCSCSamplingGraph(
        torch.ops.graphbolt.fused_csc_sampling_graph(
            csc_indptr,
            indices,
            node_type_offset,
            type_per_edge,
            node_type_to_id,
            edge_type_to_id,
            node_attributes,
            edge_attributes,
        ),
    )


def load_from_shared_memory(
    shared_memory_name: str,
) -> FusedCSCSamplingGraph:
    """Load a FusedCSCSamplingGraph object from shared memory.

    Parameters
    ----------
    shared_memory_name : str
        Name of the shared memory.

    Returns
    -------
    FusedCSCSamplingGraph
        The loaded FusedCSCSamplingGraph object on shared memory.
    """
    return FusedCSCSamplingGraph(
        torch.ops.graphbolt.load_from_shared_memory(shared_memory_name),
    )


def from_dglgraph(
    DGLGraphInstance,
    is_homogeneous: bool = False,
    include_original_edge_id: bool = False,
) -> FusedCSCSamplingGraph:
    """Convert a DGLGraph to FusedCSCSamplingGraph."""
    from dgl.base import EID, ETYPE, NID, NTYPE
    from dgl.convert import to_homogeneous

    g = DGLGraphInstance

    homo_g, ntype_count, _ = to_homogeneous(
        g, ndata=g.ndata, edata=g.edata, return_count=True
    )

    if is_homogeneous:
        node_type_to_id = None
        edge_type_to_id = None
    else:
        # Initialize metadata.
        node_type_to_id = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
        edge_type_to_id = {
            etype_tuple_to_str(etype): g.get_etype_id(etype)
            for etype in g.canonical_etypes
        }

    # Obtain CSC matrix.
    indptr, indices, edge_ids = homo_g.adj_tensors("csc")
    ntype_count.insert(0, 0)
    node_type_offset = (
        None
        if is_homogeneous
        else torch.cumsum(torch.LongTensor(ntype_count), 0)
    )

    # Assign edge type according to the order of CSC matrix.
    type_per_edge = (
        None
        if is_homogeneous
        else torch.index_select(homo_g.edata[ETYPE], dim=0, index=edge_ids)
    )

    node_attributes = {}
    edge_attributes = {}
    for feat_name, feat_data in homo_g.ndata.items():
        if feat_name not in (NID, NTYPE):
            node_attributes[feat_name] = feat_data
    for feat_name, feat_data in homo_g.edata.items():
        if feat_name not in (EID, ETYPE):
            edge_attributes[feat_name] = feat_data
    if include_original_edge_id:
        # Assign edge attributes according to the original eids mapping.
        edge_attributes[ORIGINAL_EDGE_ID] = torch.index_select(
            homo_g.edata[EID], dim=0, index=edge_ids
        )

    return FusedCSCSamplingGraph(
        torch.ops.graphbolt.fused_csc_sampling_graph(
            indptr,
            indices,
            node_type_offset,
            type_per_edge,
            node_type_to_id,
            edge_type_to_id,
            node_attributes,
            edge_attributes,
        ),
    )
