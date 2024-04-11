import backend as F
import dgl.graphbolt as gb
import pytest
import torch


def test_find_reverse_edges_homo():
    edges = torch.tensor([[1, 3, 5], [2, 4, 5]]).T
    edges = gb.add_reverse_edges(edges)
    expected_edges = torch.tensor([[1, 3, 5, 2, 4, 5], [2, 4, 5, 1, 3, 5]]).T
    assert torch.equal(edges, expected_edges)
    assert torch.equal(edges[1], expected_edges[1])


def test_find_reverse_edges_hetero():
    edges = {
        "A:r:B": torch.tensor([[1, 5], [2, 5]]).T,
        "B:rr:A": torch.tensor([[3], [3]]).T,
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"})
    expected_edges = {
        "A:r:B": torch.tensor([[1, 5], [2, 5]]).T,
        "B:rr:A": torch.tensor([[3, 2, 5], [3, 1, 5]]).T,
    }
    assert torch.equal(edges["A:r:B"], expected_edges["A:r:B"])
    assert torch.equal(edges["B:rr:A"], expected_edges["B:rr:A"])


def test_find_reverse_edges_bi_reverse_types():
    edges = {
        "A:r:B": torch.tensor([[1, 5], [2, 5]]).T,
        "B:rr:A": torch.tensor([[3], [3]]).T,
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A", "B:rr:A": "A:r:B"})
    expected_edges = {
        "A:r:B": torch.tensor([[1, 5, 3], [2, 5, 3]]).T,
        "B:rr:A": torch.tensor([[3, 2, 5], [3, 1, 5]]).T,
    }
    assert torch.equal(edges["A:r:B"], expected_edges["A:r:B"])
    assert torch.equal(edges["B:rr:A"], expected_edges["B:rr:A"])


def test_find_reverse_edges_circual_reverse_types():
    edges = {
        "A:r1:B": torch.tensor([[1, 1]]),
        "B:r2:C": torch.tensor([[2, 2]]),
        "C:r3:A": torch.tensor([[3, 3]]),
    }
    edges = gb.add_reverse_edges(
        edges, {"A:r1:B": "B:r2:C", "B:r2:C": "C:r3:A", "C:r3:A": "A:r1:B"}
    )
    expected_edges = {
        "A:r1:B": torch.tensor([[1, 3], [1, 3]]).T,
        "B:r2:C": torch.tensor([[2, 1], [2, 1]]).T,
        "C:r3:A": torch.tensor([[3, 2], [3, 2]]).T,
    }
    assert torch.equal(edges["A:r1:B"], expected_edges["A:r1:B"])
    assert torch.equal(edges["B:r2:C"], expected_edges["B:r2:C"])
    assert torch.equal(edges["A:r1:B"], expected_edges["A:r1:B"])
    assert torch.equal(edges["C:r3:A"], expected_edges["C:r3:A"])
