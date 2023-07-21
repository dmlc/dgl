"""CSC format sampling graph."""
# pylint: disable= invalid-name
import os
import tarfile
import tempfile
from collections import defaultdict
from typing import Dict, Optional, Tuple, Union

import torch

from ...base import ETYPE
from ...convert import to_homogeneous
from ...heterograph import DGLGraph
from .sampled_subgraph_impl import SampledSubgraphImpl


class GraphMetadata:
    r"""Class for metadata of csc sampling graph."""

    def __init__(
        self,
        node_type_to_id: Dict[str, int],
        edge_type_to_id: Dict[Tuple[str, str, str], int],
    ):
        """Initialize the GraphMetadata object.

        Parameters
        ----------
        node_type_to_id : Dict[str, int]
            Dictionary from node types to node type IDs.
        edge_type_to_id : Dict[Tuple[str, str, str], int]
            Dictionary from edge types to edge type IDs.

        Raises
        ------
        AssertionError
            If any of the assertions fail.
        """

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
            src, edge, dst = edge_type
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

        self.node_type_to_id = node_type_to_id
        self.edge_type_to_id = edge_type_to_id


class CSCSamplingGraph:
    r"""Class for CSC sampling graph."""

    def __repr__(self):
        return _csc_sampling_graph_str(self)

    def __init__(
        self, c_csc_graph: torch.ScriptObject, metadata: Optional[GraphMetadata]
    ):
        self._c_csc_graph = c_csc_graph
        self._metadata = metadata

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        int
            The number of rows in the dense format.
        """
        return self._c_csc_graph.num_nodes()

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph.
        """
        return self._c_csc_graph.num_edges()

    @property
    def csc_indptr(self) -> torch.tensor:
        """Returns the indices pointer in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices pointer in the CSC graph. An integer tensor with
            shape `(num_nodes+1,)`.
        """
        return self._c_csc_graph.csc_indptr()

    @property
    def indices(self) -> torch.tensor:
        """Returns the indices in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices in the CSC graph. An integer tensor with shape
            `(num_edges,)`.

        Notes
        -------
        It is assumed that edges of each node are already sorted by edge type
        ids.
        """
        return self._c_csc_graph.indices()

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

    @property
    def type_per_edge(self) -> Optional[torch.Tensor]:
        """Returns the edge type tensor if present.

        Returns
        -------
        torch.Tensor or None
            If present, returns a 1D integer tensor of shape (num_edges,)
            containing the type of each edge in the graph.
        """
        return self._c_csc_graph.type_per_edge()

    @property
    def edge_attributes(self) -> Optional[Dict[str, torch.Tensor]]:
        """Returns the edge attributes dictionary.

        Returns
        -------
        torch.Tensor or None
            If present, returns a dictionary of edge attributes. Each key
            represents the attribute's name, while the corresponding value
            holds the attribute's specific value. The length of each value
            should match the total number of edges."
        """
        return self._c_csc_graph.edge_attributes()

    @property
    def metadata(self) -> Optional[GraphMetadata]:
        """Returns the metadata of the graph.

        Returns
        -------
        GraphMetadata or None
            If present, returns the metadata of the graph.
        """
        return self._metadata

    def in_subgraph(self, nodes: torch.Tensor) -> torch.ScriptObject:
        """Return the subgraph induced on the inbound edges of the given nodes.

        An in subgraph is equivalent to creating a new graph using the incoming
        edges of the given nodes.

        Parameters
        ----------
        nodes : torch.Tensor
            The nodes to form the subgraph which are type agnostic.

        Returns
        -------
        torch.classes.graphbolt.SampledSubgraph
            The in subgraph.
        """
        # Ensure nodes is 1-D tensor.
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."
        # Ensure that there are no duplicate nodes.
        assert len(torch.unique(nodes)) == len(
            nodes
        ), "Nodes cannot have duplicate values."
        # TODO: change the result to 'SampledSubgraphImpl'.
        return self._c_csc_graph.in_subgraph(nodes)

    def _convert_to_sampled_subgraph(
        self,
        C_sampled_subgraph: torch.ScriptObject,
    ):
        """An internal function used to convert a fused homogeneous sampled
        subgraph to general struct 'SampledSubgraphImpl'."""
        column_num = (
            C_sampled_subgraph.indptr[1:] - C_sampled_subgraph.indptr[:-1]
        )
        column = C_sampled_subgraph.reverse_column_node_ids.repeat_interleave(
            column_num
        )
        row = C_sampled_subgraph.indices
        type_per_edge = C_sampled_subgraph.type_per_edge
        if type_per_edge is None:
            # The sampled graph is already a homogeneous graph.
            node_pairs = (row, column)
        else:
            # The sampled graph is a fused homogenized graph, which need to be
            # converted to heterogeneous graphs.
            node_pairs = defaultdict(list)
            for etype, etype_id in self.metadata.edge_type_to_id.items():
                src_ntype, _, dst_ntype = etype
                src_ntype_id = self.metadata.node_type_to_id[src_ntype]
                dst_ntype_id = self.metadata.node_type_to_id[dst_ntype]
                mask = type_per_edge == etype_id
                hetero_row = row[mask] - self.node_type_offset[src_ntype_id]
                hetero_column = (
                    column[mask] - self.node_type_offset[dst_ntype_id]
                )
                node_pairs[etype] = (hetero_row, hetero_column)
        return SampledSubgraphImpl(node_pairs=node_pairs)

    def _convert_to_homogeneous_nodes(self, nodes):
        homogeneous_nodes = []
        for ntype, ids in nodes.items():
            ntype_id = self.metadata.node_type_to_id[ntype]
            homogeneous_nodes.append(ids + self.node_type_offset[ntype_id])
        return torch.cat(homogeneous_nodes)

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
              - When the value is -1, all neighbors will be chosen for
                sampling. It is equivalent to selecting all neighbors when
                the fanout is >= the number of neighbors (and replace is set to
                false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        probs_name: str, optional
            An optional string specifying the name of an edge attribute used a. This
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
        >>> ntypes = {'n1': 0, 'n2': 1, 'n3': 2}
        >>> etypes = {('n1', 'e1', 'n2'): 0, ('n1', 'e2', 'n3'): 1}
        >>> metadata = gb.GraphMetadata(ntypes, etypes)
        >>> indptr = torch.LongTensor([0, 3, 4, 5, 7])
        >>> indices = torch.LongTensor([0, 1, 3, 2, 3, 0, 1])
        >>> node_type_offset = torch.LongTensor([0, 2, 3, 4])
        >>> type_per_edge = torch.LongTensor([0, 0, 1, 0, 1, 0, 1])
        >>> graph = gb.from_csc(indptr, indices, type_per_edge=type_per_edge,
        ... node_type_offset=node_type_offset, metadata=metadata)
        >>> nodes = {'n1': torch.LongTensor([1]), 'n2': torch.LongTensor([0])}
        >>> fanouts = torch.tensor([1, 1])
        >>> subgraph = graph.sample_neighbors(nodes, fanouts)
        >>> print(subgraph.node_pairs)
        defaultdict(<class 'list'>, {('n1', 'e1', 'n2'): (tensor([2]), \
        tensor([1])), ('n1', 'e2', 'n3'): (tensor([3]), tensor([2]))})
        """
        if isinstance(nodes, dict):
            nodes = self._convert_to_homogeneous_nodes(nodes)

        C_sampled_subgraph = self._sample_neighbors(
            nodes, fanouts, replace, False, probs_name
        )

        return self._convert_to_sampled_subgraph(C_sampled_subgraph)

    def _check_sampler_arguments(self, nodes, fanouts, probs_name):
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."
        assert fanouts.dim() == 1, "Fanouts should be 1-D tensor."
        expected_fanout_len = 1
        if self.metadata and self.metadata.edge_type_to_id:
            expected_fanout_len = len(self.metadata.edge_type_to_id)
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
                probs_or_mask.size(0) == self.num_edges
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
        return_eids: bool = False,
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
              - When the value is -1, all neighbors will be chosen for
                sampling. It is equivalent to selecting all neighbors when
                the fanout is >= the number of neighbors (and replace is set to
                false).
              - When the value is a non-negative integer, it serves as a
                minimum threshold for selecting neighbors.
        replace: bool
            Boolean indicating whether the sample is preformed with or
            without replacement. If True, a value can be selected multiple
            times. Otherwise, each value can be selected only once.
        return_eids: bool
            Boolean indicating whether the edge IDs of sampled edges,
            represented as a 1D tensor, should be returned. This is
            typically used when edge features are required.
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
        return self._c_csc_graph.sample_neighbors(
            nodes, fanouts.tolist(), replace, False, return_eids, probs_name
        )

    def sample_layer_neighbors(
        self,
        nodes: Union[torch.Tensor, Dict[str, torch.Tensor]],
        fanouts: torch.Tensor,
        replace: bool = False,
        probs_name: Optional[str] = None,
    ) -> SampledSubgraphImpl:
        """Sample neighboring edges of the given nodes and return the induced
        subgraph via layer-neighbor sampling from arXiv:2210.13339:
        "Layer-Neighbor Sampling -- Defusing Neighborhood Explosion in GNNs"

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
              - When the value is -1, all neighbors will be chosen for
                sampling. It is equivalent to selecting all neighbors when
                the fanout is >= the number of neighbors (and replace is set to
                false).
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
        TODO: Provide typical examples.
        """
        if isinstance(nodes, dict):
            nodes = self._convert_to_homogeneous_nodes(nodes)

        self._check_sampler_arguments(nodes, fanouts, probs_name)
        C_sampled_subgraph = self._c_csc_graph.sample_neighbors(
            nodes, fanouts.tolist(), replace, True, False, probs_name
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
        the graph.

        Parameters
        ----------
        edge_type: Tuple[str]
            The type of edges in the provided node_pairs. Any negative edges
            sampled will also have the same type. If set to None, it will be
            considered as a homogeneous graph.
        node_pairs : Tuple[Tensor]
            A tuple of two 1D tensors that represent the source and destination
            of positive edges, with 'positive' indicating that these edges are
            present in the graph. It's important to note that within the
            context of a heterogeneous graph, the ids in these tensors signify
            heterogeneous ids.
        negative_ratio: int
            The ratio of the number of negative samples to positive samples.

        Returns
        -------
        Tuple[Tensor]
            A tuple consisting of two 1D tensors represents the source and
            destination of negative edges. In the context of a heterogeneous
            graph, both the input nodes and the selected nodes are represented
            by heterogeneous IDs, and the formed edges are of the input type
            `edge_type`. Note that negative refers to false negatives, which
            means the edge could be present or not present in the graph.
        """
        if edge_type:
            assert (
                self.node_type_offset is not None
            ), "The 'node_type_offset' array is necessary for performing \
                negative sampling by edge type."
            _, _, dst_node_type = edge_type
            dst_node_type_id = self.metadata.node_type_to_id[dst_node_type]
            max_node_id = (
                self.node_type_offset[dst_node_type_id + 1]
                - self.node_type_offset[dst_node_type_id]
            )
        else:
            max_node_id = self.num_nodes
        return self._c_csc_graph.sample_negative_edges_uniform(
            node_pairs,
            negative_ratio,
            max_node_id,
        )

    def copy_to_shared_memory(self, shared_memory_name: str):
        """Copy the graph to shared memory.

        Parameters
        ----------
        shared_memory_name : str
            Name of the shared memory.

        Returns
        -------
        CSCSamplingGraph
            The copied CSCSamplingGraph object on shared memory.
        """
        return CSCSamplingGraph(
            self._c_csc_graph.copy_to_shared_memory(shared_memory_name),
            self._metadata,
        )


def from_csc(
    csc_indptr: torch.Tensor,
    indices: torch.Tensor,
    node_type_offset: Optional[torch.tensor] = None,
    type_per_edge: Optional[torch.tensor] = None,
    edge_attributes: Optional[Dict[str, torch.tensor]] = None,
    metadata: Optional[GraphMetadata] = None,
) -> CSCSamplingGraph:
    """Create a CSCSamplingGraph object from a CSC representation.

    Parameters
    ----------
    csc_indptr : torch.Tensor
        Pointer to the start of each row in the `indices`. An integer tensor
        with shape `(num_nodes+1,)`.
    indices : torch.Tensor
        Column indices of the non-zero elements in the CSC graph. An integer
        tensor with shape `(num_edges,)`.
    node_type_offset : Optional[torch.tensor], optional
        Offset of node types in the graph, by default None.
    type_per_edge : Optional[torch.tensor], optional
        Type ids of each edge in the graph, by default None.
    edge_attributes: Optional[Dict[str, torch.tensor]], optional
        Edge attributes of the graph, by default None.
    metadata: Optional[GraphMetadata], optional
        Metadata of the graph, by default None.
    Returns
    -------
    CSCSamplingGraph
        The created CSCSamplingGraph object.

    Examples
    --------
    >>> ntypes = {'n1': 0, 'n2': 1, 'n3': 2}
    >>> etypes = {('n1', 'e1', 'n2'): 0, ('n1', 'e2', 'n3'): 1}
    >>> metadata = graphbolt.GraphMetadata(ntypes, etypes)
    >>> csc_indptr = torch.tensor([0, 2, 5, 7])
    >>> indices = torch.tensor([1, 3, 0, 1, 2, 0, 3])
    >>> node_type_offset = torch.tensor([0, 1, 2, 3])
    >>> type_per_edge = torch.tensor([0, 1, 0, 1, 1, 0, 0])
    >>> graph = graphbolt.from_csc(csc_indptr, indices, node_type_offset, \
    >>>                            type_per_edge, None, metadata)
    >>> print(graph)
    CSCSamplingGraph(csc_indptr=tensor([0, 2, 5, 7]),
                     indices=tensor([1, 3, 0, 1, 2, 0, 3]),
                     num_nodes=3, num_edges=7)
    """
    if metadata and metadata.node_type_to_id and node_type_offset is not None:
        assert len(metadata.node_type_to_id) + 1 == node_type_offset.size(
            0
        ), "node_type_offset length should be |ntypes| + 1."
    return CSCSamplingGraph(
        torch.ops.graphbolt.from_csc(
            csc_indptr,
            indices,
            node_type_offset,
            type_per_edge,
            edge_attributes,
        ),
        metadata,
    )


def load_from_shared_memory(
    shared_memory_name: str,
    metadata: Optional[GraphMetadata] = None,
) -> CSCSamplingGraph:
    """Load a CSCSamplingGraph object from shared memory.

    Parameters
    ----------
    shared_memory_name : str
        Name of the shared memory.

    Returns
    -------
    CSCSamplingGraph
        The loaded CSCSamplingGraph object on shared memory.
    """
    return CSCSamplingGraph(
        torch.ops.graphbolt.load_from_shared_memory(shared_memory_name),
        metadata,
    )


def _csc_sampling_graph_str(graph: CSCSamplingGraph) -> str:
    """Internal function for converting a csc sampling graph to string
    representation.
    """
    csc_indptr_str = str(graph.csc_indptr)
    indices_str = str(graph.indices)
    meta_str = f"num_nodes={graph.num_nodes}, num_edges={graph.num_edges}"
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


def load_csc_sampling_graph(filename):
    """Load CSCSamplingGraph from tar file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(filename, "r") as archive:
            archive.extractall(temp_dir)
        graph_filename = os.path.join(temp_dir, "csc_sampling_graph.pt")
        metadata_filename = os.path.join(temp_dir, "metadata.pt")
        return CSCSamplingGraph(
            torch.ops.graphbolt.load_csc_sampling_graph(graph_filename),
            torch.load(metadata_filename),
        )


def save_csc_sampling_graph(graph, filename):
    """Save CSCSamplingGraph to tar file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        graph_filename = os.path.join(temp_dir, "csc_sampling_graph.pt")
        torch.ops.graphbolt.save_csc_sampling_graph(
            graph._c_csc_graph, graph_filename
        )
        metadata_filename = os.path.join(temp_dir, "metadata.pt")
        torch.save(graph.metadata, metadata_filename)
        with tarfile.open(filename, "w") as archive:
            archive.add(
                graph_filename, arcname=os.path.basename(graph_filename)
            )
            archive.add(
                metadata_filename, arcname=os.path.basename(metadata_filename)
            )
    print(f"CSCSamplingGraph has been saved to {filename}.")


def from_dglgraph(g: DGLGraph) -> CSCSamplingGraph:
    """Convert a DGLGraph to CSCSamplingGraph."""
    homo_g, ntype_count, _ = to_homogeneous(g, return_count=True)
    # Initialize metadata.
    node_type_to_id = {ntype: g.get_ntype_id(ntype) for ntype in g.ntypes}
    edge_type_to_id = {
        etype: g.get_etype_id(etype) for etype in g.canonical_etypes
    }
    metadata = GraphMetadata(node_type_to_id, edge_type_to_id)

    # Obtain CSC matrix.
    indptr, indices, _ = homo_g.adj_tensors("csc")
    ntype_count.insert(0, 0)
    node_type_offset = torch.cumsum(torch.LongTensor(ntype_count), 0)
    type_per_edge = homo_g.edata[ETYPE]

    return CSCSamplingGraph(
        torch.ops.graphbolt.from_csc(
            indptr,
            indices,
            node_type_offset,
            type_per_edge,
            None,
        ),
        metadata,
    )
