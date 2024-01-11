"""CSC format sampling graph."""
# pylint: disable= invalid-name
from typing import Dict, Optional, Union

import torch

from dgl.utils import recursive_apply

from ...base import EID, ETYPE
from ...convert import to_homogeneous
from ...heterograph import DGLGraph
from ..base import etype_str_to_tuple, etype_tuple_to_str, ORIGINAL_EDGE_ID
from ..sampling_graph import SamplingGraph
from .sampled_subgraph_impl import CSCFormatBase, SampledSubgraphImpl


__all__ = [
    "FusedCSCSamplingGraph",
    "fused_csc_sampling_graph",
    "load_from_shared_memory",
    "from_dglgraph",
]


class FusedCSCSamplingGraph(SamplingGraph):
    r"""A sampling graph in CSC format."""

    def __repr__(self):
        return _csc_sampling_graph_str(self)

    def __init__(
        self,
        c_csc_graph: torch.ScriptObject,
    ):
        super().__init__()
        self._c_csc_graph = c_csc_graph

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

        offset = self.node_type_offset

        # Homogenous.
        if offset is None or self.node_type_to_id is None:
            return self._c_csc_graph.num_nodes()

        # Heterogenous
        else:
            num_nodes_per_type = {
                _type: (offset[_idx + 1] - offset[_idx]).item()
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
        """Returns the node type offset tensor if present.

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

    @node_type_offset.setter
    def node_type_offset(
        self, node_type_offset: Optional[torch.Tensor]
    ) -> None:
        """Sets the node type offset tensor if present."""
        self._c_csc_graph.set_node_type_offset(node_type_offset)

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
            nodes = self._convert_to_homogeneous_nodes(nodes)
        # Ensure nodes is 1-D tensor.
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."
        # Ensure that there are no duplicate nodes.
        assert len(torch.unique(nodes)) == len(
            nodes
        ), "Nodes cannot have duplicate values."

        _in_subgraph = self._c_csc_graph.in_subgraph(nodes)
        return self._convert_to_sampled_subgraph(_in_subgraph)

    def _convert_to_homogeneous_nodes(self, nodes, timestamps=None):
        homogeneous_nodes = []
        homogeneous_timestamps = []
        for ntype, ids in nodes.items():
            ntype_id = self.node_type_to_id[ntype]
            homogeneous_nodes.append(
                ids + self.node_type_offset[ntype_id].item()
            )
            if timestamps is not None:
                homogeneous_timestamps.append(timestamps[ntype])
        if timestamps is not None:
            return torch.cat(homogeneous_nodes), torch.cat(
                homogeneous_timestamps
            )
        return torch.cat(homogeneous_nodes)

    def _convert_to_sampled_subgraph(
        self,
        C_sampled_subgraph: torch.ScriptObject,
    ) -> SampledSubgraphImpl:
        """An internal function used to convert a fused homogeneous sampled
        subgraph to general struct 'SampledSubgraphImpl'."""
        indptr = C_sampled_subgraph.indptr
        indices = C_sampled_subgraph.indices
        type_per_edge = C_sampled_subgraph.type_per_edge
        column = C_sampled_subgraph.original_column_node_ids
        original_edge_ids = C_sampled_subgraph.original_edge_ids

        has_original_eids = (
            self.edge_attributes is not None
            and ORIGINAL_EDGE_ID in self.edge_attributes
        )
        if has_original_eids:
            original_edge_ids = torch.index_select(
                self.edge_attributes[ORIGINAL_EDGE_ID],
                dim=0,
                index=original_edge_ids,
            )
        if type_per_edge is None:
            # The sampled graph is already a homogeneous graph.
            sampled_csc = CSCFormatBase(indptr=indptr, indices=indices)
        else:
            self.node_type_offset = self.node_type_offset.to(column.device)
            # 1. Find node types for each nodes in column.
            node_types = (
                torch.searchsorted(self.node_type_offset, column, right=True)
                - 1
            )

            original_hetero_edge_ids = {}
            sub_indices = {}
            sub_indptr = {}
            # 2. For loop each node type.
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
                        indices[eids] - self.node_type_offset[src_ntype_id]
                    )
                    cum_edges = torch.searchsorted(
                        eids, nids_original_indptr, right=False
                    )
                    sub_indptr[etype] = torch.cat(
                        (torch.tensor([0], device=indptr.device), cum_edges)
                    )
                    if has_original_eids:
                        original_hetero_edge_ids[etype] = original_edge_ids[
                            eids
                        ]
            if has_original_eids:
                original_edge_ids = original_hetero_edge_ids
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
        )

    def sample_neighbors(
        self,
        nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
    ) -> SampledSubgraphImpl:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph.

        Parameters
        ----------
        nodes: torch.Tensor or Dict[str, torch.Tensor]
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
        if isinstance(nodes, dict):
            nodes = self._convert_to_homogeneous_nodes(nodes)

        C_sampled_subgraph = self._sample_neighbors(
            nodes, fanouts, replace, probs_name
        )
        return self._convert_to_sampled_subgraph(C_sampled_subgraph)

    def _check_sampler_arguments(self, nodes, fanouts, probs_name):
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
        if probs_name:
            assert (
                probs_name in self.edge_attributes
            ), f"Unknown edge attribute '{probs_name}'."
            probs_or_mask = self.edge_attributes[probs_name]
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
        nodes: torch.Tensor,
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
    ) -> torch.ScriptObject:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph.

        Parameters
        ----------
        nodes: torch.Tensor
            IDs of the given seed nodes.
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

        Returns
        -------
        torch.classes.graphbolt.SampledSubgraph
            The sampled C subgraph.
        """
        # Ensure nodes is 1-D tensor.
        self._check_sampler_arguments(nodes, fanouts, probs_name)
        has_original_eids = (
            self.edge_attributes is not None
            and ORIGINAL_EDGE_ID in self.edge_attributes
        )
        return self._c_csc_graph.sample_neighbors(
            nodes,
            fanouts.tolist(),
            replace,
            False,
            has_original_eids,
            probs_name,
        )

    def sample_layer_neighbors(
        self,
        nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
    ) -> SampledSubgraphImpl:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph via layer-neighbor sampling from the NeurIPS 2023 paper
        `Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs
        <https://arxiv.org/abs/2210.13339>`__

        Parameters
        ----------
        nodes: torch.Tensor or Dict[str, torch.Tensor]
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
        if isinstance(nodes, dict):
            nodes = self._convert_to_homogeneous_nodes(nodes)

        self._check_sampler_arguments(nodes, fanouts, probs_name)
        has_original_eids = (
            self.edge_attributes is not None
            and ORIGINAL_EDGE_ID in self.edge_attributes
        )
        C_sampled_subgraph = self._c_csc_graph.sample_neighbors(
            nodes,
            fanouts.tolist(),
            replace,
            True,
            has_original_eids,
            probs_name,
        )
        return self._convert_to_sampled_subgraph(C_sampled_subgraph)

    def temporal_sample_neighbors(
        self,
        nodes: torch.Tensor,
        input_nodes_timestamp: torch.Tensor,
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
        node_timestamp_attr_name: Optional[str] = None,
        edge_timestamp_attr_name: Optional[str] = None,
    ) -> torch.ScriptObject:
        """Temporally Sample neighboring edges of the given nodes and return the induced
        subgraph.

        If `node_timestamp_attr_name` or `edge_timestamp_attr_name` is given,
        the sampled neighbors or edges of an input node must have a timestamp
        that is no later than that of the input node.

        Parameters
        ----------
        nodes: torch.Tensor
            IDs of the given seed nodes.
        input_nodes_timestamp: torch.Tensor
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
        if isinstance(nodes, dict):
            nodes, input_nodes_timestamp = self._convert_to_homogeneous_nodes(
                nodes, input_nodes_timestamp
            )

        # Ensure nodes is 1-D tensor.
        self._check_sampler_arguments(nodes, fanouts, probs_name)
        has_original_eids = (
            self.edge_attributes is not None
            and ORIGINAL_EDGE_ID in self.edge_attributes
        )
        C_sampled_subgraph = self._c_csc_graph.temporal_sample_neighbors(
            nodes,
            input_nodes_timestamp,
            fanouts.tolist(),
            replace,
            has_original_eids,
            probs_name,
            node_timestamp_attr_name,
            edge_timestamp_attr_name,
        )
        return self._convert_to_sampled_subgraph(C_sampled_subgraph)

    def sample_negative_edges_uniform(
        self, edge_type, node_pairs, negative_ratio
    ):
        """
        Sample negative edges by randomly choosing negative source-destination
        pairs according to a uniform distribution. For each edge ``(u, v)``,
        it is supposed to generate `negative_ratio` pairs of negative edges
        ``(u, v')``, where ``v'`` is chosen uniformly from all the nodes in
        the graph. As ``u`` is exactly same as the corresponding positive edges,
        it returns None for negative sources.

        Parameters
        ----------
        edge_type: str
            The type of edges in the provided node_pairs. Any negative edges
            sampled will also have the same type. If set to None, it will be
            considered as a homogeneous graph.
        node_pairs : Tuple[Tensor, Tensor]
            A tuple of two 1D tensors that represent the source and destination
            of positive edges, with 'positive' indicating that these edges are
            present in the graph. It's important to note that within the
            context of a heterogeneous graph, the ids in these tensors signify
            heterogeneous ids.
        negative_ratio: int
            The ratio of the number of negative samples to positive samples.

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tuple consisting of two 1D tensors represents the source and
            destination of negative edges. In the context of a heterogeneous
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
        pos_src, _ = node_pairs
        num_negative = pos_src.size(0) * negative_ratio
        return (
            None,
            torch.randint(
                0,
                max_node_id,
                (num_negative,),
                dtype=pos_src.dtype,
                device=pos_src.device,
            ),
        )

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

        return self._apply_to_members(_to)

    def pin_memory_(self):
        """Copy `FusedCSCSamplingGraph` to the pinned memory in-place."""

        def _pin(x):
            return x.pin_memory() if hasattr(x, "pin_memory") else x

        self._apply_to_members(_pin)


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
        Type ids of each edge in the graph, by default None.
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
    >>> csc_indptr = torch.tensor([0, 2, 5, 7])
    >>> indices = torch.tensor([1, 3, 0, 1, 2, 0, 3])
    >>> node_type_offset = torch.tensor([0, 1, 2, 3])
    >>> type_per_edge = torch.tensor([0, 1, 0, 1, 1, 0, 0])
    >>> graph = graphbolt.fused_csc_sampling_graph(csc_indptr, indices,
    ...             node_type_offset=node_type_offset,
    ...             type_per_edge=type_per_edge,
    ...             node_type_to_id=ntypes, edge_type_to_id=etypes,
    ...             node_attributes=None, edge_attributes=None,)
    >>> print(graph)
    FusedCSCSamplingGraph(csc_indptr=tensor([0, 2, 5, 7]),
                     indices=tensor([1, 3, 0, 1, 2, 0, 3]),
                     total_num_nodes=3, total_num_edges=7)
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


def _csc_sampling_graph_str(graph: FusedCSCSamplingGraph) -> str:
    """Internal function for converting a csc sampling graph to string
    representation.
    """
    csc_indptr_str = str(graph.csc_indptr)
    indices_str = str(graph.indices)
    meta_str = f"num_nodes={graph.total_num_nodes}, num_edges={graph.num_edges}"
    if graph.node_type_offset is not None:
        meta_str += f", node_type_offset={graph.node_type_offset}"
    if graph.type_per_edge is not None:
        meta_str += f", type_per_edge={graph.type_per_edge}"
    if graph.node_type_to_id is not None:
        meta_str += f", node_type_to_id={graph.node_type_to_id}"
    if graph.edge_type_to_id is not None:
        meta_str += f", edge_type_to_id={graph.edge_type_to_id}"
    if graph.node_attributes is not None:
        meta_str += f", node_attributes={graph.node_attributes}"
    if graph.edge_attributes is not None:
        meta_str += f", edge_attributes={graph.edge_attributes}"

    prefix = f"{type(graph).__name__}("

    def _add_indent(_str, indent):
        lines = _str.split("\n")
        lines = [lines[0]] + [" " * indent + line for line in lines[1:]]
        return "\n".join(lines)

    final_str = (
        "csc_indptr="
        + _add_indent(csc_indptr_str, len("csc_indptr="))
        + ",\n"
        + "indices="
        + _add_indent(indices_str, len("indices="))
        + ",\n"
        + meta_str
        + ")"
    )

    final_str = prefix + _add_indent(final_str, len(prefix))
    return final_str


def from_dglgraph(
    g: DGLGraph,
    is_homogeneous: bool = False,
    include_original_edge_id: bool = False,
) -> FusedCSCSamplingGraph:
    """Convert a DGLGraph to FusedCSCSamplingGraph."""

    homo_g, ntype_count, _ = to_homogeneous(g, return_count=True)

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
