import unittest

import backend as F
import pytest
import torch

from dgl.graphbolt.impl.sampled_subgraph_impl import FusedSampledSubgraphImpl


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
        original_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        src_to_exclude = torch.tensor([11])
    else:
        original_row_node_ids = None
        src_to_exclude = torch.tensor([2])

    if reverse_column:
        original_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
        dst_to_exclude = torch.tensor([9])
    else:
        original_column_node_ids = None
        dst_to_exclude = torch.tensor([4])
    original_edge_ids = torch.Tensor([5, 9, 10])
    subgraph = FusedSampledSubgraphImpl(
        node_pairs,
        original_column_node_ids,
        original_row_node_ids,
        original_edge_ids,
    )
    edges_to_exclude = (src_to_exclude, dst_to_exclude)
    result = subgraph.exclude_edges(edges_to_exclude)
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
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero(reverse_row, reverse_column):
    node_pairs = {
        "A:relation:B": (
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 1, 0]),
        )
    }
    if reverse_row:
        original_row_node_ids = {
            "A": torch.tensor([13, 14, 15]),
        }
        src_to_exclude = torch.tensor([15, 13])
    else:
        original_row_node_ids = None
        src_to_exclude = torch.tensor([2, 0])
    if reverse_column:
        original_column_node_ids = {
            "B": torch.tensor([10, 11, 12]),
        }
        dst_to_exclude = torch.tensor([10, 12])
    else:
        original_column_node_ids = None
        dst_to_exclude = torch.tensor([0, 2])
    original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
    subgraph = FusedSampledSubgraphImpl(
        node_pairs=node_pairs,
        original_column_node_ids=original_column_node_ids,
        original_row_node_ids=original_row_node_ids,
        original_edge_ids=original_edge_ids,
    )

    edges_to_exclude = {
        "A:relation:B": (
            src_to_exclude,
            dst_to_exclude,
        )
    }
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_node_pairs = {
        "A:relation:B": (
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
    expected_edge_ids = {"A:relation:B": torch.tensor([20])}

    _assert_container_equal(result.node_pairs, expected_node_pairs)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="`to` function needs GPU to test.",
)
def test_sampled_subgraph_to_device():
    # Initialize data.
    node_pairs = {
        "A:relation:B": (
            torch.tensor([0, 1, 2]),
            torch.tensor([2, 1, 0]),
        )
    }
    original_row_node_ids = {
        "A": torch.tensor([13, 14, 15]),
    }
    src_to_exclude = torch.tensor([15, 13])
    original_column_node_ids = {
        "B": torch.tensor([10, 11, 12]),
    }
    dst_to_exclude = torch.tensor([10, 12])
    original_edge_ids = {"A:relation:B": torch.tensor([19, 20, 21])}
    subgraph = FusedSampledSubgraphImpl(
        node_pairs=node_pairs,
        original_column_node_ids=original_column_node_ids,
        original_row_node_ids=original_row_node_ids,
        original_edge_ids=original_edge_ids,
    )
    edges_to_exclude = {
        "A:relation:B": (
            src_to_exclude,
            dst_to_exclude,
        )
    }
    graph = subgraph.exclude_edges(edges_to_exclude)

    # Copy to device.
    graph = graph.to("cuda")

    # Check.
    for key in graph.node_pairs:
        assert graph.node_pairs[key][0].device.type == "cuda"
        assert graph.node_pairs[key][1].device.type == "cuda"
    for key in graph.original_column_node_ids:
        assert graph.original_column_node_ids[key].device.type == "cuda"
    for key in graph.original_row_node_ids:
        assert graph.original_row_node_ids[key].device.type == "cuda"
    for key in graph.original_edge_ids:
        assert graph.original_edge_ids[key].device.type == "cuda"
