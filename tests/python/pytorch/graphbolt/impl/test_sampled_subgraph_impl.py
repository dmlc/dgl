import pytest
import torch
from dgl.graphbolt.impl.sampled_subgraph_impl import (
    exclude_edges,
    SampledSubgraphImpl,
)


def _assert_container_equal(lhs, rhs):
    if isinstance(lhs, torch.Tensor):
        assert isinstance(rhs, torch.Tensor)
        assert torch.equal(lhs, rhs)
    elif isinstance(lhs, tuple):
        assert isinstance(rhs, tuple)
        assert len(lhs) == len(rhs)
        for l, r in zip(lhs, rhs):
            _assert_container_equal(l, r)
    elif isinstance(lhs, dict):
        assert isinstance(rhs, dict)
        assert len(lhs) == len(rhs)
        for key, value in lhs.items():
            assert key in rhs
            _assert_container_equal(value, rhs[key])


@pytest.mark.parametrize("is_reverse", [True, False])
def test_exclude_edges_homo(is_reverse):
    node_pairs = (torch.tensor([0, 2, 3]), torch.tensor([1, 4, 2]))
    if is_reverse:
        reverse_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        reverse_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        reverse_row_node_ids = reverse_column_node_ids = None
    reverse_edge_ids = torch.Tensor([5, 9, 10])
    subgraph = SampledSubgraphImpl(
        node_pairs,
        reverse_column_node_ids,
        reverse_row_node_ids,
        reverse_edge_ids,
    )
    if is_reverse:
        excluded_edges = (torch.tensor([11]), torch.tensor([9]))
    else:
        excluded_edges = (torch.tensor([2]), torch.tensor([4]))
    result = exclude_edges(subgraph, excluded_edges)
    expected_node_pairs = (torch.tensor([0, 3]), torch.tensor([1, 2]))
    if is_reverse:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 10])

    _assert_container_equal(result.node_pairs, expected_node_pairs)
    _assert_container_equal(
        result.reverse_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.reverse_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.reverse_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("is_reverse", [True, False])
def test_exclude_edges_hetero(is_reverse):
    node_pairs = {
        ("A", "relation", "B"): (
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 1, 0]),
        )
    }
    if is_reverse:
        reverse_row_node_ids = {
            "A": torch.tensor([13, 14, 15]),
        }
        reverse_column_node_ids = {
            "B": torch.tensor([10, 11, 12]),
        }
    else:
        reverse_row_node_ids = reverse_column_node_ids = None
    reverse_edge_ids = {("A", "relation", "B"): torch.tensor([19, 20, 21])}
    subgraph = SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )

    if is_reverse:
        excluded_edges = {
            ("A", "relation", "B"): (
                torch.tensor([15, 13]),
                torch.tensor([10, 12]),
            )
        }
    else:
        excluded_edges = {
            ("A", "relation", "B"): (torch.tensor([2, 0]), torch.tensor([0, 2]))
        }
    result = exclude_edges(subgraph, excluded_edges)
    expected_node_pairs = {
        ("A", "relation", "B"): (
            torch.tensor([1]),
            torch.tensor([1]),
        )
    }
    if is_reverse:
        expected_row_node_ids = {
            "A": torch.tensor([13, 14, 15]),
        }
        expected_column_node_ids = {
            "B": torch.tensor([10, 11, 12]),
        }
    else:
        expected_row_node_ids = expected_column_node_ids = None
    expected_edge_ids = {("A", "relation", "B"): torch.tensor([20])}

    _assert_container_equal(result.node_pairs, expected_node_pairs)
    _assert_container_equal(
        result.reverse_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.reverse_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.reverse_edge_ids, expected_edge_ids)
