import backend as F
import dgl.graphbolt as gb
import pytest
import torch


def test_find_reverse_edges_homo():
    edges = (torch.tensor([1, 3, 5]), torch.tensor([2, 4, 5]))
    edges = gb.add_reverse_edges(edges)
    expected_edges = (
        torch.tensor([1, 3, 5, 2, 4, 5]),
        torch.tensor([2, 4, 5, 1, 3, 5]),
    )
    assert torch.equal(edges[0], expected_edges[0])
    assert torch.equal(edges[1], expected_edges[1])


def test_find_reverse_edges_hetero():
    edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A"})
    expected_edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3, 2, 5]), torch.tensor([3, 1, 5])),
    }
    assert torch.equal(edges["A:r:B"][0], expected_edges["A:r:B"][0])
    assert torch.equal(edges["A:r:B"][1], expected_edges["A:r:B"][1])
    assert torch.equal(edges["B:rr:A"][0], expected_edges["B:rr:A"][0])
    assert torch.equal(edges["B:rr:A"][1], expected_edges["B:rr:A"][1])


def test_find_reverse_edges_bi_reverse_types():
    edges = {
        "A:r:B": (torch.tensor([1, 5]), torch.tensor([2, 5])),
        "B:rr:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(edges, {"A:r:B": "B:rr:A", "B:rr:A": "A:r:B"})
    expected_edges = {
        "A:r:B": (torch.tensor([1, 5, 3]), torch.tensor([2, 5, 3])),
        "B:rr:A": (torch.tensor([3, 2, 5]), torch.tensor([3, 1, 5])),
    }
    assert torch.equal(edges["A:r:B"][0], expected_edges["A:r:B"][0])
    assert torch.equal(edges["A:r:B"][1], expected_edges["A:r:B"][1])
    assert torch.equal(edges["B:rr:A"][0], expected_edges["B:rr:A"][0])
    assert torch.equal(edges["B:rr:A"][1], expected_edges["B:rr:A"][1])


def test_find_reverse_edges_circual_reverse_types():
    edges = {
        "A:r1:B": (torch.tensor([1]), torch.tensor([1])),
        "B:r2:C": (torch.tensor([2]), torch.tensor([2])),
        "C:r3:A": (torch.tensor([3]), torch.tensor([3])),
    }
    edges = gb.add_reverse_edges(
        edges, {"A:r1:B": "B:r2:C", "B:r2:C": "C:r3:A", "C:r3:A": "A:r1:B"}
    )
    expected_edges = {
        "A:r1:B": (torch.tensor([1, 3]), torch.tensor([1, 3])),
        "B:r2:C": (torch.tensor([2, 1]), torch.tensor([2, 1])),
        "C:r3:A": (torch.tensor([3, 2]), torch.tensor([3, 2])),
    }
    assert torch.equal(edges["A:r1:B"][0], expected_edges["A:r1:B"][0])
    assert torch.equal(edges["A:r1:B"][1], expected_edges["A:r1:B"][1])
    assert torch.equal(edges["B:r2:C"][0], expected_edges["B:r2:C"][0])
    assert torch.equal(edges["B:r2:C"][1], expected_edges["B:r2:C"][1])
    assert torch.equal(edges["A:r1:B"][0], expected_edges["A:r1:B"][0])
    assert torch.equal(edges["A:r1:B"][1], expected_edges["A:r1:B"][1])
    assert torch.equal(edges["C:r3:A"][0], expected_edges["C:r3:A"][0])
    assert torch.equal(edges["C:r3:A"][1], expected_edges["C:r3:A"][1])
