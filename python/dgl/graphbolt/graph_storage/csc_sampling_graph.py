"""CSC format sampling graph."""
# pylint: disable= invalid-name
import os
import tarfile
import tempfile
from typing import Dict, Optional, Tuple

import torch


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
        SampledSubgraph
            The in subgraph.
        """
        # Ensure nodes is 1-D tensor.
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."
        # Ensure that there are no duplicate nodes.
        assert len(torch.unique(nodes)) == len(
            nodes
        ), "Nodes cannot have duplicate values."
        return self._c_csc_graph.in_subgraph(nodes)

    def sample_neighbors(
        self,
        nodes: torch.Tensor,
        fanouts: torch.Tensor,
        replace: bool = False,
        return_eids: bool = False,
        probs_or_mask: Optional[torch.Tensor] = None,
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
        probs_or_mask: torch.Tensor, optional
            Optional tensor containing the (unnormalized) probabilities
            associated with each neighboring edge of a node. It must be a 1D
            floating-point or boolean tensor with the number of elements equal
            to the number of edges.
        Returns
        -------
        SampledSubgraph
            The sampled subgraph.

        Examples
        --------
        >>> indptr = torch.LongTensor([0, 3, 5, 7])
        >>> indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 1])
        >>> type_per_edge = torch.LongTensor([0, 0, 1, 0, 1, 0, 1])
        >>> graph = gb.from_csc(indptr, indices, type_per_edge=type_per_edge)
        >>> nodes = torch.LongTensor([1, 2])
        >>> fanouts = torch.tensor([1, 1])
        >>> subgraph = graph.sample_neighbors(nodes, fanouts, return_eids=True)
        >>> print(subgraph.indptr)
        tensor([0, 2, 4])
        >>> print(subgraph.indices)
        tensor([2, 3, 0, 1])
        >>> print(subgraph.reverse_column_node_ids)
        tensor([1, 2])
        >>> print(subgraph.reverse_edge_ids)
        tensor([3, 4, 5, 6])
        >>> print(subgraph.type_per_edge)
        tensor([0, 1, 0, 1])
        """
        # Ensure nodes is 1-D tensor.
        assert nodes.dim() == 1, "Nodes should be 1-D tensor."
        assert fanouts.dim() == 1, "Fanouts should be 1-D tensor."
        if fanouts.size(0) > 1:
            assert (
                self.type_per_edge is not None
            ), "To perform sampling for each edge type (when the length of \
                `fanouts` > 1), the graph must include edge type information."
        assert torch.all(
            (fanouts >= 0) | (fanouts == -1)
        ), "Fanouts should consist of values that are either -1 or \
            greater than or equal to 0."
        if self.metadata and self.metadata.edge_type_to_id:
            assert len(self.metadata.edge_type_to_id) == fanouts.size(
                0
            ), "Fanouts should have the same number of elements as etypes."
        if probs_or_mask is not None:
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
        return self._c_csc_graph.sample_neighbors(
            nodes, fanouts.tolist(), replace, return_eids, probs_or_mask
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
    >>>                            type_per_edge, metadata)
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
            csc_indptr, indices, node_type_offset, type_per_edge
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
