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
