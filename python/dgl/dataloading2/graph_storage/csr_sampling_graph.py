"""Csr format sampling graph."""
# pylint: disable= invalid-name
from typing import List, Optional, Tuple

import numpy as np
import torch

from .base import Graph

HeteroInfo = Tuple[List[str], List[str], torch.tensor, torch.tensor]


class CSRSamplingGraph(Graph):
    r"""Class for Csr sampling graph."""

    def __repr__(self):
        pass

    def __init__(self, c_csr_graph: torch.ScriptObject):
        self.c_csr_graph = c_csr_graph

    @property
    def num_rows(self) -> int:
        """Returns the number of rows in the dense format.

        Returns
        -------
        int
            The number of rows in the dense format
        """
        return self.c_csr_graph.num_rows()

    @property
    def num_cols(self) -> int:
        """Returns the number of columns in the dense format.

        Returns
        -------
        int
            The number of columns in the dense format
        """
        return self.c_csr_graph.num_cols()

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the Graph.

        Returns
        -------
        int
            The number of rows in the dense format
        """
        return self.c_csr_graph.num_nodes()

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the Graph.

        Returns
        -------
        int
            The number of columns in the dense format
        """
        return self.c_csr_graph.num_edges()

    @property
    def indptr(self) -> torch.tensor:
        """Returns the indices pointer in the Csr.

        Returns
        -------
        int
            The indices pointer in the Csr
        """
        return self.c_csr_graph.indptr()

    @property
    def indices(self) -> torch.tensor:
        """Returns the indices in the Csr.

        Returns
        -------
        int
            The indices in the Csr
        """
        return self.c_csr_graph.indices()

    @property
    def is_heterogeneous(self) -> bool:
        """Returns if the graph is heterogeneous.

        Returns
        -------
        bool
            True if the graph is heterogeneous, False otherwise.
        """
        return self.c_csr_graph.is_heterogeneous()

    @property
    def node_types(self) -> Optional[torch.Tensor]:
        """Returns the type of each node in the graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size num_nodes,
            containing the type of each node in the graph. If the graph is homogeneous,
            returns None.
        """
        return (
            self.c_csr_graph.node_types()
            if self.c_csr_graph.is_heterogeneous()
            else None
        )

    @property
    def edge_types(self) -> Optional[torch.Tensor]:
        """Returns the type of each edge in the graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size num_edges,
            containing the type of each edge in the graph. If the graph is homogeneous,
            returns None.
        """
        return (
            self.c_csr_graph.edge_types()
            if self.c_csr_graph.is_heterogeneous()
            else None
        )

    @property
    def node_type_offset(self) -> Optional[torch.Tensor]:
        """Returns the node type offset tensor for a heterogeneous graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size num_node_types + 1,
            where num_node_types is the number of unique node types in the graph.
            The i-th element of the tensor indicates the index in the node type tensor
            where nodes of type i start. If the graph is homogeneous, returns None.
        """
        return (
            self.c_csr_graph.node_type_offset()
            if self.c_csr_graph.is_heterogeneous()
            else None
        )

    @property
    def per_edge_type(self) -> Optional[torch.Tensor]:
        """Returns the edge type tensor for a heterogeneous graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size num_edges,
            containing the type of each edge in the graph. If the graph is homogeneous,
            returns None.
        """
        return (
            self.c_csr_graph.per_edge_type()
            if self.c_csr_graph.is_heterogeneous()
            else None
        )

    def _set_hetero_info(
        self,
        hetero_info: HeteroInfo,
    ):
        self.c_csr_graph.set_hetero_info(*hetero_info)


def from_csr(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    shape: Optional[Tuple[int, int]] = None,
    hetero_info: Optional[HeteroInfo] = None,
) -> CSRSamplingGraph:
    """
    Create a CSRSamplingGraph object from a CSR sparse matrix representation.

    Parameters
    ----------
    indptr : torch.Tensor
        Pointer to the start of each row in indices.
    indices : torch.Tensor
        Column indices of the non-zero elements in the CSR matrix.
    shape : Optional[Tuple[int, int]], optional
        Shape of the matrix (number of rows, number of columns), by default None.
    hetero_info : Optional[HeteroInfo], optional
        Heterogeneous graph metadata, by default None.

    Returns
    -------
    CSRSamplingGraph
        The created CSRSamplingGraph object.
    """
    if shape is None:
        shape = (indptr.shape[0] - 1, torch.max(indices).item())

    graph = CSRSamplingGraph(
        torch.ops.graphbolt.from_csr(shape, indptr, indices)
    )
    if hetero_info is not None:
        graph._set_hetero_info(hetero_info)

    return graph


def from_coo(
    row: torch.Tensor,
    col: torch.Tensor,
    hetero_info: Optional[HeteroInfo] = None,
) -> CSRSamplingGraph:
    """
    Create a CSRSamplingGraph object from a COO sparse matrix representation.

    Parameters
    ----------
    row : torch.Tensor
        Row indices of the non-zero elements in the COO matrix.
    col : torch.Tensor
        Column indices of the non-zero elements in the COO matrix.
    hetero_info : Optional[HeteroInfo], optional
        Heterogeneous graph metadata, by default None.

    Returns
    -------
    CSRSamplingGraph
        The created CSRSamplingGraph object.
    """
    assert row.dim() == 1
    assert col.dim() == 1
    assert row.size(0) == col.size(0)

    _, _, _, type_per_edge = hetero_info
    if type_per_edge is not None:
        assert type_per_edge.size(0) == row.size(0)
        indices = np.lexsort((row.numpy(), type_per_edge.numpy()))
        indices = torch.from_numpy(indices)
        row, col, type_per_edge = (
            row[indices],
            col[indices],
            type_per_edge[indices],
        )

    else:
        row, indices = torch.sort(row)
        col = col[indices]

    indptr = torch.cumsum(torch.bincount(row), dim=0)

    return from_csr(indptr, col, hetero_info=hetero_info)
