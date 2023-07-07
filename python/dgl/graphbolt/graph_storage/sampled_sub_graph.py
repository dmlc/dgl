"""Heterogeneous sampled subgraph."""
# pylint: disable= invalid-name
from collections.abc import Mapping


class SampledSubGraph:
    r"""Class for sampled subgraph."""

    def __init__(
        self,
        node_pairs,
        reverse_column_node_ids=None,
        reverse_row_node_ids=None,
        reverse_edge_ids=None,
    ):
        """ ""Initialize the SampledSubGraph object. In the context of a
        heterogeneous graph, each field should be of `Dict` type. For
        homogeneous graphs, each field should correspond to its respective
        value type."

        Parameters
        ----------
        node_pairs :  Tuple[Tensor] or Dict[(str, str, str), Tuple[Tensor]]
            Node pairs representing source-destination edges.
            - If `node_pairs` is a tuple: It should be in the format ('u', 'v')
                representing source and destination pairs.
            - If `node_pairs` is a dictionary: The keys should be edge type and
                the values should be corresponding node pairs. The ids inside
                is heterogeneous ids.
        reverse_column_node_ids : Optional[Tensor or Dict[str, Tensor]]
            Column's reverse node ids in the original graph. A graph structure
            can be treated as a coordinated row and column pair, and this is
            the mapped ids of the column.
            - If `reverse_column_node_ids` is a tensor: It represents the
                original node ids.
            - If `reverse_column_node_ids` is a dictionary: The keys should be
                node type and the values should be corresponding original
                heterogeneous node ids.
        reverse_row_node_ids : Optional[Tensor or Dict[str, Tensor]]
            Row's reverse node ids in the original graph. A graph structure
            can be treated as a coordinated row and column pair, and this is
            the mapped ids of the row.
            - If `reverse_row_node_ids` is a tensor: It represents the
                original node ids.
            - If `reverse_row_node_ids` is a dictionary: The keys should be
                node type and the values should be corresponding original
                heterogeneous node ids.
        reverse_edge_ids: Optional[Tensor or Dict[(str, str, str), Tensor]]
            Reverse edge ids in the original graph.This is useful when edge
            features are needed.
            - If `reverse_edge_ids` is a tensor: It represents the
                original edge ids.
            - If `reverse_edge_ids` is a dictionary: The keys should be
                edge type and the values should be corresponding original
                heterogeneous edge ids.

        Raises
        ------
        AssertionError
            If any of the assertions fail.
        """
        if isinstance(node_pairs, Mapping):
            for etype, pair in node_pairs.items():
                assert (
                    isinstance(etype, tuple) and len(etype) == 3
                ), "Edge type should be a triplet of strings (str, str, str)."
                assert all(
                    isinstance(item, str) for item in etype
                ), "Edge type should be a triplet of strings (str, str, str)."
                assert (
                    isinstance(pair, tuple) and len(pair) == 2
                ), "Node pair should be a source-destination tuple (u, v)."
                assert all(
                    isinstance(item, str) for item in pair
                ), "Nodes in pairs should be of type string."
        else:
            assert (
                isinstance(node_pairs, tuple) and len(node_pairs) == 2
            ), "Node pair should be a source-destination tuple (u, v)."
            assert all(
                isinstance(item, str) for item in node_pairs
            ), "Nodes in pairs should be of type string."

        self._node_pairs = node_pairs
        self._reverse_column_node_ids = reverse_column_node_ids
        self._reverse_row_node_ids = reverse_row_node_ids
        self._reverse_edge_ids = reverse_edge_ids

    @property
    def node_pairs(self):
        """Returns the node pairs in the sub-graph.

        Returns
        -------
        Tuple[Tensor] or Dict[(str, str, str), Tuple[Tensor]]
            Node pairs representing source-destination edges.
            - If `node_pairs` is a tuple: It should be in the format ('u', 'v')
                representing source and destination pairs.
            - If `node_pairs` is a dictionary: The keys should be edge type and
                the values should be corresponding node pairs. The ids inside
                is heterogeneous ids.
        """
        return self._node_pairs

    @property
    def reverse_column_node_ids(self):
        """Returns corresponding reverse column node ids the original graph.

        Returns
        -------
        Optional[Tensor or Dict[str, Tensor]]
            Column's reverse node ids in the original graph. A graph structure
            can be treated as a coordinated row and column pair, and this is
            the mapped ids of the column.
            - If `reverse_column_node_ids` is a tensor: It represents the
                original node ids.
            - If `reverse_column_node_ids` is a dictionary: The keys should be
                node type and the values should be corresponding original
                heterogeneous node ids.
        """
        return self._reverse_column_node_ids

    @property
    def reverse_row_node_ids(self):
        """Returns corresponding reverse row node ids the original graph.

        Returns
        -------
        Optional[Tensor or Dict[str, Tensor]]
            Row's reverse node ids in the original graph. A graph structure
            can be treated as a coordinated row and column pair, and this is
            the mapped ids of the row.
            - If `reverse_row_node_ids` is a tensor: It represents the
                original node ids.
            - If `reverse_row_node_ids` is a dictionary: The keys should be
                node type and the values should be corresponding original
                heterogeneous node ids.
        """
        return self._reverse_column_node_ids

    @property
    def reverse_edge_ids(self):
        """Returns corresponding reverse edge ids the original graph.

        Returns
        -------
        Optional[Tensor or Dict[str, Tensor]]
            Reverse edge ids in the original graph.This is useful when edge
            features are needed.
            - If `reverse_edge_ids` is a tensor: It represents the
                original edge ids.
            - If `reverse_edge_ids` is a dictionary: The keys should be
                edge type and the values should be corresponding original
                heterogeneous edge ids.
        """
        return self._reverse_edge_ids
