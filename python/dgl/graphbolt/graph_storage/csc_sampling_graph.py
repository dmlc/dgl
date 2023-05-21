"""CSC format sampling graph."""
# pylint: disable= invalid-name
from typing import List, Optional, Tuple

import torch


HeteroInfo = Tuple[List[str], List[str], torch.tensor, torch.tensor]


class CSCSamplingGraph:
    r"""Class for CSC sampling graph."""

    def __repr__(self):
        return _csc_sampling_graph_str(self)

    def __init__(self, c_csc_graph: torch.ScriptObject):
        self.c_csc_graph = c_csc_graph

    @property
    def num_nodes(self) -> int:
        """Returns the number of nodes in the graph.

        Returns
        -------
        int
            The number of rows in the dense format
        """
        return self.c_csc_graph.num_nodes()

    @property
    def num_edges(self) -> int:
        """Returns the number of edges in the graph.

        Returns
        -------
        int
            The number of edges in the graph
        """
        return self.c_csc_graph.num_edges()

    @property
    def csc_indptr(self) -> torch.tensor:
        """Returns the indices pointer in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices pointer in the CSC graph
        """
        return self.c_csc_graph.csc_indptr()

    @property
    def indices(self) -> torch.tensor:
        """Returns the indices in the CSC graph.

        Returns
        -------
        torch.tensor
            The indices in the CSC graph
        """
        return self.c_csc_graph.indices()

    @property
    def is_heterogeneous(self) -> bool:
        """Returns if the graph is heterogeneous.

        Returns
        -------
        bool
            True if the graph is heterogeneous, False otherwise.
        """
        return self.c_csc_graph.is_heterogeneous()

    @property
    def node_types(self) -> Optional[List[str]]:
        """Returns all node types in the graph.

        Returns
        -------
        List[str] or None
            If the graph is heterogeneous, returns a string list of size
            the number of node types, containing all node types in the graph.
            If the graph is homogeneous, returns None.
        """
        return (
            self.c_csc_graph.node_types()
            if self.c_csc_graph.is_heterogeneous()
            else None
        )

    @property
    def edge_types(self) -> Optional[List[str]]:
        """Returns all edge types in the graph.

        Returns
        -------
        List[str] or None
            If the graph is heterogeneous, returns a string triplet list tensor
            of size the number of edge types, containing all edge types in the
            graph. If the graph is homogeneous, returns None.
        """
        return (
            self.c_csc_graph.edge_types()
            if self.c_csc_graph.is_heterogeneous()
            else None
        )

    @property
    def node_type_offset(self) -> Optional[torch.Tensor]:
        """Returns the node type offset tensor for a heterogeneous graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size
            num_node_types + 1, where num_node_types is the number of unique
            node types in the graph. The i-th element of the tensor indicates
            the index in the node type tensor where nodes of type i start.
            If the graph is homogeneous, returns None.
        """
        return (
            self.c_csc_graph.node_type_offset()
            if self.c_csc_graph.is_heterogeneous()
            else None
        )

    @property
    def type_per_edge(self) -> Optional[torch.Tensor]:
        """Returns the edge type tensor for a heterogeneous graph.

        Returns
        -------
        torch.Tensor or None
            If the graph is heterogeneous, returns a 1D tensor of size
            the number of edges, containing the type of each edge in the graph.
            If the graph is homogeneous, returns None.
        """
        return (
            self.c_csc_graph.type_per_edge()
            if self.c_csc_graph.is_heterogeneous()
            else None
        )


def from_csc(
    csc_indptr: torch.Tensor,
    indices: torch.Tensor,
    etype_sorted: bool = True,
    num_nodes: Optional[int] = None,
    hetero_info: Optional[HeteroInfo] = None,
) -> CSCSamplingGraph:
    """
    Create a CSCSamplingGraph object from a CSC representation.

    Parameters
    ----------
    csc_indptr : torch.Tensor
        Pointer to the start of each row in the `indices`.
    indices : torch.Tensor
        Column indices of the non-zero elements in the CSC graph.
    etype_sorted : bool
        A hint telling whether neighbors of each node are sorted by edge type
        ID, it is only meaningful when graph is heterogeneous, by default True.
    num_nodes : int, optional
        Number of nodes in the graph, by default None.
    hetero_info : Optional[HeteroInfo], optional
        Heterogeneous graph metadata, by default None.
        Indicates whether the graph is homogeneous or heterogeneous.

    Returns
    -------
    CSCSamplingGraph
        The created CSCSamplingGraph object.

    Examples
    --------
    >>> csc_indptr = torch.tensor([0, 2, 5, 7])
    >>> indices = torch.tensor([1, 3, 0, 1, 2, 0, 3])
    >>> num_nodes = 3
    >>> ntypes = ['n1', 'n2', 'n3']
    >>> etypes = [('n1', 'e1', 'n2'), ('n1', 'e2', 'n3')]
    >>> type_per_edge = torch.tensor([0, 1, 0, 1, 1, 0, 0])
    >>> hetero_info = (ntypes, etypes, torch.tensor([0, 1, 2]), type_per_edge)
    >>> graph = from_csc(csc_indptr, indices, num_nodes, hetero_info)
    >>> print(graph)
    CSCSamplingGraph(csc_indptr=tensor([0, 2, 5, 7]),
                    indices=tensor([1, 3, 0, 1, 2, 0, 3]),
                    num_nodes=3, num_edges=7, is_heterogeneous=True)
    """
    if num_nodes is None:
        num_nodes = csc_indptr.shape[0] - 1

    assert csc_indptr.shape[0] == num_nodes + 1
    assert torch.max(indices) < num_nodes
    if hetero_info is None:
        return CSCSamplingGraph(
            torch.ops.graphbolt.from_csc(num_nodes, csc_indptr, indices)
        )
    else:
        if etype_sorted is False:
            ntypes, etypes, node_type_offset, type_per_edge = hetero_info
            # sort neighbors by edge type id for each node
            sorted_etype = []
            sorted_indices = []
            for s, e in zip(csc_indptr[:-1], csc_indptr[1:]):
                neighbor_etype, index = torch.sort(type_per_edge[s:e])
                sorted_etype.append(neighbor_etype)
                sorted_indices.append(indices[s:e][index])
            type_per_edge = torch.cat(sorted_etype, dim=0)
            indices = torch.cat(sorted_indices, dim=0)
            hetero_info = (ntypes, etypes, node_type_offset, type_per_edge)

        return CSCSamplingGraph(
            torch.ops.graphbolt.from_csc_with_hetero_info(
                num_nodes, csc_indptr, indices, *hetero_info
            )
        )


def from_coo(
    coo: torch.Tensor,
    num_nodes: Optional[int] = None,
    hetero_info: Optional[HeteroInfo] = None,
) -> CSCSamplingGraph:
    """
    Create a CSCSamplingGraph object from a COO representation.

    Parameters
    ----------
    coo : torch.Tensor
        COO format graph.
    num_nodes : int, optional
        Number of nodes in the graph, by default None.
    hetero_info : Optional[HeteroInfo], optional
        Heterogeneous graph metadata, by default None. Indicates whether
        the graph is homogeneous or heterogeneous.

    Returns
    -------
    CSCSamplingGraph
        The created CSCSamplingGraph object.

    Examples
    --------
    >>> coo = torch.tensor([[0, 1, 2, 2, 3, 3],
    ...                     [1, 2, 0, 3, 1, 3]])
    >>> ntypes, etypes = ['n1', 'n2', 'n3']
    >>> etypes = [('n1', 'e1', 'n2'), ('n1', 'e2', 'n3')]
    >>> type_per_edge = torch.tensor([0, 1, 0, 1, 0, 1])
    >>> hetero_info = (ntypes, etypes, torch.tensor([0, 2, 5]), type_per_edge)
    >>> graph = dataloading2.from_coo(coo, hetero_info)
    >>> print(graph)
    CSCSamplingGraph(csc_indptr=tensor([0, 1, 3, 4, 6]),
                    indices=tensor([2, 0, 3, 1, 2, 3]),
                    num_nodes=4, num_edges=6, is_heterogeneous=True)
    """
    assert coo.dim() == 2

    if num_nodes is None:
        num_nodes = torch.max(coo) + (1 if coo.size(1) > 0 else 0)
    rol = coo[1]
    _, indices = torch.sort(rol)

    row, col = torch.index_select(coo, 1, indices)
    csc_indptr = torch.cumsum(torch.bincount(col, minlength=num_nodes), dim=0)
    csc_indptr = torch.cat(
        (torch.zeros((1,), dtype=csc_indptr.dtype), csc_indptr), dim=0
    )
    if hetero_info is not None:
        ntypes, etypes, node_type_offset, type_per_edge = hetero_info
        type_per_edge = type_per_edge[indices]
        hetero_info = (ntypes, etypes, node_type_offset, type_per_edge)

    return from_csc(
        csc_indptr,
        row,
        etype_sorted=False,
        num_nodes=num_nodes,
        hetero_info=hetero_info,
    )


def _csc_sampling_graph_str(graph: CSCSamplingGraph) -> str:
    """Internal function for converting a csc sampling graph to string
    representation.
    """
    csc_indptr_str = str(graph.csc_indptr)
    indices_str = str(graph.indices)
    meta_str = (
        f"num_nodes={graph.num_nodes}, num_edges={graph.num_edges}"
        f", is_heterogeneous={graph.is_heterogeneous}"
    )
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
