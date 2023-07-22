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


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_homo(reverse_row, reverse_column):
    node_pairs = (torch.tensor([0, 2, 3]), torch.tensor([1, 4, 2]))
    if reverse_row:
        reverse_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        src_to_exclude = torch.tensor([11])
    else:
        reverse_row_node_ids = None
        src_to_exclude = torch.tensor([2])

    if reverse_column:
        reverse_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
        dst_to_exclude = torch.tensor([9])
    else:
        reverse_column_node_ids = None
        dst_to_exclude = torch.tensor([4])
    reverse_edge_ids = torch.Tensor([5, 9, 10])
    subgraph = SampledSubgraphImpl(
        node_pairs,
        reverse_column_node_ids,
        reverse_row_node_ids,
        reverse_edge_ids,
    )
    edges_to_exclude = (src_to_exclude, dst_to_exclude)
    result = exclude_edges(subgraph, edges_to_exclude)
    expected_node_pairs = (torch.tensor([0, 3]), torch.tensor([1, 2]))
    if reverse_row:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 10])

    _assert_container_equal(result.node_pairs, expected_node_pairs)
    _assert_container_equal(
        result.reverse_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.reverse_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.reverse_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero(reverse_row, reverse_column):
    node_pairs = {
        ("A", "relation", "B"): (
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 1, 0]),
        )
    }
    if reverse_row:
        reverse_row_node_ids = {
            "A": torch.tensor([13, 14, 15]),
        }
        src_to_exclude = torch.tensor([15, 13])
    else:
        reverse_row_node_ids = None
        src_to_exclude = torch.tensor([2, 0])
    if reverse_column:
        reverse_column_node_ids = {
            "B": torch.tensor([10, 11, 12]),
        }
        dst_to_exclude = torch.tensor([10, 12])
    else:
        reverse_column_node_ids = None
        dst_to_exclude = torch.tensor([0, 2])
    reverse_edge_ids = {("A", "relation", "B"): torch.tensor([19, 20, 21])}
    subgraph = SampledSubgraphImpl(
        node_pairs=node_pairs,
        reverse_column_node_ids=reverse_column_node_ids,
        reverse_row_node_ids=reverse_row_node_ids,
        reverse_edge_ids=reverse_edge_ids,
    )

    edges_to_exclude = {
        ("A", "relation", "B"): (
            src_to_exclude,
            dst_to_exclude,
        )
    }
    result = exclude_edges(subgraph, edges_to_exclude)
    expected_node_pairs = {
        ("A", "relation", "B"): (
            torch.tensor([1]),
            torch.tensor([1]),
        )
    }
    if reverse_row:
        expected_row_node_ids = {
            "A": torch.tensor([13, 14, 15]),
        }
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = {
            "B": torch.tensor([10, 11, 12]),
        }
    else:
        expected_column_node_ids = None
    expected_edge_ids = {("A", "relation", "B"): torch.tensor([20])}

    _assert_container_equal(result.node_pairs, expected_node_pairs)
    _assert_container_equal(
        result.reverse_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.reverse_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.reverse_edge_ids, expected_edge_ids)
