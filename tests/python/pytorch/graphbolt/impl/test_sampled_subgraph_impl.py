import unittest

import backend as F

import dgl.graphbolt as gb
import pytest
import torch
from dgl.graphbolt.impl.sampled_subgraph_impl import SampledSubgraphImpl


def _assert_container_equal(lhs, rhs):
    if isinstance(lhs, torch.Tensor):
        assert isinstance(rhs, torch.Tensor)
        assert torch.equal(lhs, rhs)
    elif isinstance(lhs, tuple):
        assert isinstance(rhs, tuple)
        assert len(lhs) == len(rhs)
        for l, r in zip(lhs, rhs):
            _assert_container_equal(l, r)
    elif isinstance(lhs, gb.CSCFormatBase):
        assert isinstance(rhs, gb.CSCFormatBase)
        assert len(lhs.indptr) == len(rhs.indptr)
        assert len(lhs.indices) == len(rhs.indices)
        _assert_container_equal(lhs.indptr, rhs.indptr)
        _assert_container_equal(lhs.indices, rhs.indices)
    elif isinstance(lhs, dict):
        assert isinstance(rhs, dict)
        assert len(lhs) == len(rhs)
        for key, value in lhs.items():
            assert key in rhs
            _assert_container_equal(value, rhs[key])


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_homo_deduplicated(reverse_row, reverse_column):
    csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 2, 2, 3]), indices=torch.tensor([0, 3, 2])
    )
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
    subgraph = SampledSubgraphImpl(
        csc_formats,
        original_column_node_ids,
        original_row_node_ids,
        original_edge_ids,
    )
    edges_to_exclude = (src_to_exclude, dst_to_exclude)
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 2, 2, 2]), indices=torch.tensor([0, 3])
    )
    if reverse_row:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 9])

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_homo_duplicated(reverse_row, reverse_column):
    csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 3, 3, 5]),
        indices=torch.tensor([0, 3, 3, 2, 2]),
    )
    if reverse_row:
        original_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        src_to_exclude = torch.tensor([24])
    else:
        original_row_node_ids = None
        src_to_exclude = torch.tensor([3])

    if reverse_column:
        original_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
        dst_to_exclude = torch.tensor([11])
    else:
        original_column_node_ids = None
        dst_to_exclude = torch.tensor([2])
    original_edge_ids = torch.Tensor([5, 9, 9, 10, 10])
    subgraph = SampledSubgraphImpl(
        csc_formats,
        original_column_node_ids,
        original_row_node_ids,
        original_edge_ids,
    )
    edges_to_exclude = (src_to_exclude, dst_to_exclude)
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 1, 1, 3]), indices=torch.tensor([0, 2, 2])
    )
    if reverse_row:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 10, 10])
    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero_deduplicated(reverse_row, reverse_column):
    csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 2, 3]),
            indices=torch.tensor([2, 1, 0]),
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
    subgraph = SampledSubgraphImpl(
        sampled_csc=csc_formats,
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
    expected_csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 0, 1, 1]),
            indices=torch.tensor([1]),
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

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero_duplicated(reverse_row, reverse_column):
    csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 5]),
            indices=torch.tensor([2, 2, 1, 1, 0]),
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
    original_edge_ids = {"A:relation:B": torch.tensor([19, 19, 20, 20, 21])}
    subgraph = SampledSubgraphImpl(
        sampled_csc=csc_formats,
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
    expected_csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 0, 2, 2]),
            indices=torch.tensor([1, 1]),
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
    expected_edge_ids = {"A:relation:B": torch.tensor([20, 20])}

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_homo_deduplicated_tensor(reverse_row, reverse_column):
    csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 2, 2, 3]), indices=torch.tensor([0, 3, 2])
    )
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
    subgraph = SampledSubgraphImpl(
        csc_formats,
        original_column_node_ids,
        original_row_node_ids,
        original_edge_ids,
    )
    edges_to_exclude = torch.cat((src_to_exclude, dst_to_exclude)).view(1, -1)
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 2, 2, 2]), indices=torch.tensor([0, 3])
    )
    if reverse_row:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 9])

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_homo_duplicated_tensor(reverse_row, reverse_column):
    csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 3, 3, 5]),
        indices=torch.tensor([0, 3, 3, 2, 2]),
    )
    if reverse_row:
        original_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
        src_to_exclude = torch.tensor([24])
    else:
        original_row_node_ids = None
        src_to_exclude = torch.tensor([3])

    if reverse_column:
        original_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
        dst_to_exclude = torch.tensor([11])
    else:
        original_column_node_ids = None
        dst_to_exclude = torch.tensor([2])
    original_edge_ids = torch.Tensor([5, 9, 9, 10, 10])
    subgraph = SampledSubgraphImpl(
        csc_formats,
        original_column_node_ids,
        original_row_node_ids,
        original_edge_ids,
    )
    edges_to_exclude = torch.cat((src_to_exclude, dst_to_exclude)).view(1, -1)
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = gb.CSCFormatBase(
        indptr=torch.tensor([0, 0, 1, 1, 1, 3]), indices=torch.tensor([0, 2, 2])
    )
    if reverse_row:
        expected_row_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_row_node_ids = None
    if reverse_column:
        expected_column_node_ids = torch.tensor([10, 15, 11, 24, 9])
    else:
        expected_column_node_ids = None
    expected_edge_ids = torch.Tensor([5, 10, 10])
    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero_deduplicated_tensor(reverse_row, reverse_column):
    csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 2, 3]),
            indices=torch.tensor([2, 1, 0]),
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
    subgraph = SampledSubgraphImpl(
        sampled_csc=csc_formats,
        original_column_node_ids=original_column_node_ids,
        original_row_node_ids=original_row_node_ids,
        original_edge_ids=original_edge_ids,
    )

    edges_to_exclude = {
        "A:relation:B": torch.cat((src_to_exclude, dst_to_exclude))
        .view(2, -1)
        .T
    }
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 0, 1, 1]),
            indices=torch.tensor([1]),
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

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
    _assert_container_equal(
        result.original_column_node_ids, expected_column_node_ids
    )
    _assert_container_equal(result.original_row_node_ids, expected_row_node_ids)
    _assert_container_equal(result.original_edge_ids, expected_edge_ids)


@pytest.mark.parametrize("reverse_row", [True, False])
@pytest.mark.parametrize("reverse_column", [True, False])
def test_exclude_edges_hetero_duplicated_tensor(reverse_row, reverse_column):
    csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 5]),
            indices=torch.tensor([2, 2, 1, 1, 0]),
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
    original_edge_ids = {"A:relation:B": torch.tensor([19, 19, 20, 20, 21])}
    subgraph = SampledSubgraphImpl(
        sampled_csc=csc_formats,
        original_column_node_ids=original_column_node_ids,
        original_row_node_ids=original_row_node_ids,
        original_edge_ids=original_edge_ids,
    )

    edges_to_exclude = {
        "A:relation:B": torch.cat((src_to_exclude, dst_to_exclude))
        .view(2, -1)
        .T
    }
    result = subgraph.exclude_edges(edges_to_exclude)
    expected_csc_formats = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 0, 2, 2]),
            indices=torch.tensor([1, 1]),
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
    expected_edge_ids = {"A:relation:B": torch.tensor([20, 20])}

    _assert_container_equal(result.sampled_csc, expected_csc_formats)
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
    csc_format = {
        "A:relation:B": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 2, 3]),
            indices=torch.tensor([0, 1, 2]),
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
    subgraph = SampledSubgraphImpl(
        sampled_csc=csc_format,
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
    for key in graph.sampled_csc:
        assert graph.sampled_csc[key].indices.device.type == "cuda"
        assert graph.sampled_csc[key].indptr.device.type == "cuda"
    for key in graph.original_column_node_ids:
        assert graph.original_column_node_ids[key].device.type == "cuda"
    for key in graph.original_row_node_ids:
        assert graph.original_row_node_ids[key].device.type == "cuda"
    for key in graph.original_edge_ids:
        assert graph.original_edge_ids[key].device.type == "cuda"


def test_sampled_subgraph_impl_representation_homo():
    sampled_subgraph_impl = SampledSubgraphImpl(
        sampled_csc=gb.CSCFormatBase(
            indptr=torch.arange(0, 101, 10),
            indices=torch.arange(10, 110),
        ),
        original_column_node_ids=torch.arange(0, 10),
        original_row_node_ids=torch.arange(0, 110),
        original_edge_ids=None,
    )
    expected_result = str(
        """SampledSubgraphImpl(sampled_csc=CSCFormatBase(indptr=tensor([  0,  10,  20,  30,  40,  50,  60,  70,  80,  90, 100]),
                                             indices=tensor([ 10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,
                                                              24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,
                                                              38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
                                                              52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
                                                              66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
                                                              80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,
                                                              94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
                                                             108, 109]),
                               ),
                   original_row_node_ids=tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                                                  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
                                                  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                                                  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
                                                  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                                                  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
                                                  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
                                                  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109]),
                   original_edge_ids=None,
                   original_column_node_ids=tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
)"""
    )
    assert str(sampled_subgraph_impl) == expected_result, print(
        sampled_subgraph_impl
    )


def test_sampled_subgraph_impl_representation_hetero():
    sampled_subgraph_impl = SampledSubgraphImpl(
        sampled_csc={
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([4, 5, 6, 7]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4, 6, 8]),
                indices=torch.tensor([2, 3, 4, 5, 6, 7, 8, 9]),
            ),
        },
        original_column_node_ids={
            "n1": torch.tensor([1, 0, 0, 1]),
            "n2": torch.tensor([1, 2]),
        },
        original_row_node_ids={
            "n1": torch.tensor([1, 0, 0, 1, 1, 0, 0, 1]),
            "n2": torch.tensor([1, 2, 0, 1, 0, 2, 0, 2, 0, 1]),
        },
        original_edge_ids=None,
    )
    expected_result = str(
        """SampledSubgraphImpl(sampled_csc={'n1:e1:n2': CSCFormatBase(indptr=tensor([0, 2, 4]),
                                             indices=tensor([4, 5, 6, 7]),
                               ), 'n2:e2:n1': CSCFormatBase(indptr=tensor([0, 2, 4, 6, 8]),
                                             indices=tensor([2, 3, 4, 5, 6, 7, 8, 9]),
                               )},
                   original_row_node_ids={'n1': tensor([1, 0, 0, 1, 1, 0, 0, 1]), 'n2': tensor([1, 2, 0, 1, 0, 2, 0, 2, 0, 1])},
                   original_edge_ids=None,
                   original_column_node_ids={'n1': tensor([1, 0, 0, 1]), 'n2': tensor([1, 2])},
)"""
    )
    assert str(sampled_subgraph_impl) == expected_result, print(
        sampled_subgraph_impl
    )
