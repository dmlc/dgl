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


def test_unique_and_compact_hetero():
    device = torch.device(
        "cpu" if F._default_context_str == "cpu" else "cuda:0"
    )

    N1 = torch.tensor(
        [0, 5, 2, 7, 12, 7, 9, 5, 6, 2, 3, 4, 1, 0, 9], device=device
    )
    N2 = torch.tensor([0, 3, 3, 5, 2, 7, 2, 8, 4, 9, 2, 3], device=device)
    N3 = torch.tensor([1, 2, 6, 6, 1, 8, 3, 6, 3, 2], device=device)
    expected_unique = {
        "n1": torch.tensor([0, 5, 2, 7, 12, 9, 6, 3, 4, 1], device=device),
        "n2": torch.tensor([0, 3, 5, 2, 7, 8, 4, 9], device=device),
        "n3": torch.tensor([1, 2, 6, 8, 3], device=device),
    }
    if N1.is_cuda:
        expected_reverse_id = {
            k: v.sort()[1] for k, v in expected_unique.items()
        }
        expected_unique = {k: v.sort()[0] for k, v in expected_unique.items()}
    else:
        expected_reverse_id = {
            k: torch.arange(0, v.shape[0], device=device)
            for k, v in expected_unique.items()
        }
    nodes_dict = {
        "n1": N1.split(5),
        "n2": N2.split(4),
        "n3": N3.split(2),
    }
    expected_nodes_dict = {
        "n1": [
            torch.tensor([0, 1, 2, 3, 4], device=device),
            torch.tensor([3, 5, 1, 6, 2], device=device),
            torch.tensor([7, 8, 9, 0, 5], device=device),
        ],
        "n2": [
            torch.tensor([0, 1, 1, 2], device=device),
            torch.tensor([3, 4, 3, 5], device=device),
            torch.tensor([6, 7, 3, 1], device=device),
        ],
        "n3": [
            torch.tensor([0, 1], device=device),
            torch.tensor([2, 2], device=device),
            torch.tensor([0, 3], device=device),
            torch.tensor([4, 2], device=device),
            torch.tensor([4, 1], device=device),
        ],
    }

    unique, compacted = gb.unique_and_compact(nodes_dict)
    for ntype, nodes in unique.items():
        expected_nodes = expected_unique[ntype]
        assert torch.equal(nodes, expected_nodes)

    for ntype, nodes in compacted.items():
        expected_nodes = expected_nodes_dict[ntype]
        assert isinstance(nodes, list)
        for expected_node, node in zip(expected_nodes, nodes):
            node = expected_reverse_id[ntype][node]
            assert torch.equal(expected_node, node)


def test_unique_and_compact_homo():
    device = torch.device(
        "cpu" if F._default_context_str == "cpu" else "cuda:0"
    )

    N = torch.tensor(
        [0, 5, 2, 7, 12, 7, 9, 5, 6, 2, 3, 4, 1, 0, 9], device=device
    )
    expected_unique_N = torch.tensor(
        [0, 5, 2, 7, 12, 9, 6, 3, 4, 1], device=device
    )
    if N.is_cuda:
        expected_reverse_id_N = expected_unique_N.sort()[1]
        expected_unique_N = expected_unique_N.sort()[0]
    else:
        expected_reverse_id_N = torch.arange(
            0, expected_unique_N.shape[0], device=device
        )
    nodes_list = N.split(5)
    expected_nodes_list = [
        torch.tensor([0, 1, 2, 3, 4], device=device),
        torch.tensor([3, 5, 1, 6, 2], device=device),
        torch.tensor([7, 8, 9, 0, 5], device=device),
    ]

    unique, compacted = gb.unique_and_compact(nodes_list)

    assert torch.equal(unique, expected_unique_N)
    assert isinstance(compacted, list)
    for expected_node, node in zip(expected_nodes_list, compacted):
        node = expected_reverse_id_N[node]
        assert torch.equal(expected_node, node)


def test_unique_and_compact_node_pairs_hetero():
    device = torch.device(
        "cpu" if F._default_context_str == "cpu" else "cuda:0"
    )

    node_pairs = {
        "n1:e1:n2": (
            torch.tensor([1, 3, 4, 6, 2, 7, 9, 4, 2, 6], device=device),
            torch.tensor([2, 2, 2, 4, 1, 1, 1, 3, 3, 3], device=device),
        ),
        "n1:e2:n3": (
            torch.tensor([5, 2, 6, 4, 7, 2, 8, 1, 3, 0], device=device),
            torch.tensor([1, 3, 3, 3, 2, 2, 2, 7, 7, 7], device=device),
        ),
        "n2:e3:n3": (
            torch.tensor([2, 5, 4, 1, 4, 3, 6, 0], device=device),
            torch.tensor([1, 1, 3, 3, 2, 2, 7, 7], device=device),
        ),
    }

    expected_unique_nodes = {
        "n1": torch.tensor([1, 3, 4, 6, 2, 7, 9, 5, 8, 0], device=device),
        "n2": torch.tensor([1, 2, 3, 4, 5, 6, 0], device=device),
        "n3": torch.tensor([1, 2, 3, 7], device=device),
    }
    if expected_unique_nodes["n1"].is_cuda:
        expected_reverse_id = {
            "n1": expected_unique_nodes["n1"].sort()[1],
            "n2": torch.tensor([0, 1, 2, 3, 6, 4, 5], device=device),
            "n3": expected_unique_nodes["n3"].sort()[1],
        }
        expected_unique_nodes = {
            "n1": expected_unique_nodes["n1"].sort()[0],
            "n2": torch.tensor([1, 2, 3, 4, 0, 5, 6], device=device),
            "n3": expected_unique_nodes["n3"].sort()[0],
        }
    else:
        expected_reverse_id = {
            k: torch.arange(0, v.shape[0], device=device)
            for k, v in expected_unique_nodes.items()
        }
    expected_node_pairs = {
        "n1:e1:n2": (
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 2, 4, 3], device=device),
            torch.tensor([1, 1, 1, 3, 0, 0, 0, 2, 2, 2], device=device),
        ),
        "n1:e2:n3": (
            torch.tensor([7, 4, 3, 2, 5, 4, 8, 0, 1, 9], device=device),
            torch.tensor([0, 2, 2, 2, 1, 1, 1, 3, 3, 3], device=device),
        ),
        "n2:e3:n3": (
            torch.tensor([1, 4, 3, 0, 3, 2, 5, 6], device=device),
            torch.tensor([0, 0, 2, 2, 1, 1, 3, 3], device=device),
        ),
    }

    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    for ntype, nodes in unique_nodes.items():
        expected_nodes = expected_unique_nodes[ntype]
        assert torch.equal(nodes, expected_nodes)
    for etype, pair in compacted_node_pairs.items():
        u, v = pair
        ntype1, _, ntype2 = etype.split(":")
        u = expected_reverse_id[ntype1][u]
        v = expected_reverse_id[ntype2][v]
        expected_u, expected_v = expected_node_pairs[etype]
        assert torch.equal(u, expected_u)
        assert torch.equal(v, expected_v)


def test_unique_and_compact_node_pairs_homo():
    device = torch.device(
        "cpu" if F._default_context_str == "cpu" else "cuda:0"
    )

    dst_nodes = torch.tensor([1, 1, 3, 3, 5, 5, 2, 6, 6, 6, 6], device=device)
    src_nodes = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6], device=device)
    node_pairs = (src_nodes, dst_nodes)

    expected_unique_nodes = torch.tensor([1, 2, 3, 5, 6, 4], device=device)
    expected_dst_nodes = torch.tensor(
        [0, 0, 2, 2, 3, 3, 1, 4, 4, 4, 4], device=device
    )
    expected_src_ndoes = torch.tensor(
        [1, 2, 0, 5, 3, 1, 3, 0, 5, 5, 4], device=device
    )
    unique_nodes, compacted_node_pairs = gb.unique_and_compact_node_pairs(
        node_pairs
    )
    assert torch.equal(unique_nodes, expected_unique_nodes)

    u, v = compacted_node_pairs
    assert torch.equal(u, expected_src_ndoes)
    assert torch.equal(v, expected_dst_nodes)
    assert torch.equal(
        unique_nodes[:5], torch.tensor([1, 2, 3, 5, 6], device=device)
    )


def test_incomplete_unique_dst_nodes_():
    device = torch.device(
        "cpu" if F._default_context_str == "cpu" else "cuda:0"
    )

    node_pairs = (
        torch.arange(0, 50, device=device),
        torch.arange(100, 150, device=device),
    )
    unique_dst_nodes = torch.arange(150, 200, device=device)
    with pytest.raises(IndexError):
        gb.unique_and_compact_node_pairs(node_pairs, unique_dst_nodes)


def test_unique_and_compact_csc_formats_hetero():
    dst_nodes = {
        "n2": torch.tensor([2, 4, 1, 3]),
        "n3": torch.tensor([1, 3, 2, 7]),
    }
    csc_formats = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0, 3, 4, 7, 10]),
            indices=torch.tensor([1, 3, 4, 6, 2, 7, 9, 4, 2, 6]),
        ),
        "n1:e2:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 4, 7, 10]),
            indices=torch.tensor([5, 2, 6, 4, 7, 2, 8, 1, 3, 0]),
        ),
        "n2:e3:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 6, 8]),
            indices=torch.tensor([2, 5, 4, 1, 4, 3, 6, 0]),
        ),
    }

    expected_unique_nodes = {
        "n1": torch.tensor([1, 3, 4, 6, 2, 7, 9, 5, 8, 0]),
        "n2": torch.tensor([2, 4, 1, 3, 5, 6, 0]),
        "n3": torch.tensor([1, 3, 2, 7]),
    }
    expected_csc_formats = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0, 3, 4, 7, 10]),
            indices=torch.tensor([0, 1, 2, 3, 4, 5, 6, 2, 4, 3]),
        ),
        "n1:e2:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 4, 7, 10]),
            indices=torch.tensor([7, 4, 3, 2, 5, 4, 8, 0, 1, 9]),
        ),
        "n2:e3:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 6, 8]),
            indices=torch.tensor([0, 4, 1, 2, 1, 3, 5, 6]),
        ),
    }

    unique_nodes, compacted_csc_formats = gb.unique_and_compact_csc_formats(
        csc_formats, dst_nodes
    )

    for ntype, nodes in unique_nodes.items():
        expected_nodes = expected_unique_nodes[ntype]
        assert torch.equal(nodes, expected_nodes)
    for etype, pair in compacted_csc_formats.items():
        indices = pair.indices
        indptr = pair.indptr
        expected_indices = expected_csc_formats[etype].indices
        expected_indptr = expected_csc_formats[etype].indptr
        assert torch.equal(indices, expected_indices)
        assert torch.equal(indptr, expected_indptr)


def test_unique_and_compact_csc_formats_homo():
    seeds = torch.tensor([1, 3, 5, 2, 6])
    indptr = torch.tensor([0, 2, 4, 6, 7, 10, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6])
    csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)

    expected_unique_nodes = torch.tensor([1, 3, 5, 2, 6, 4])
    expected_indptr = indptr
    expected_indices = torch.tensor([3, 1, 0, 5, 2, 3, 2, 0, 5, 5, 4])

    unique_nodes, compacted_csc_formats = gb.unique_and_compact_csc_formats(
        csc_formats, seeds
    )

    indptr = compacted_csc_formats.indptr
    indices = compacted_csc_formats.indices
    assert torch.equal(indptr, expected_indptr)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(unique_nodes, expected_unique_nodes)


def test_compact_csc_format_hetero():
    dst_nodes = {
        "n2": torch.tensor([2, 4, 1, 3]),
        "n3": torch.tensor([1, 3, 2, 7]),
    }
    csc_formats = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0, 3, 4, 7, 10]),
            indices=torch.tensor([1, 3, 4, 6, 2, 7, 9, 4, 2, 6]),
        ),
        "n1:e2:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 4, 7, 10]),
            indices=torch.tensor([5, 2, 6, 4, 7, 2, 8, 1, 3, 0]),
        ),
        "n2:e3:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 6, 8]),
            indices=torch.tensor([2, 5, 4, 1, 4, 3, 6, 0]),
        ),
    }

    expected_original_row_ids = {
        "n1": torch.tensor(
            [1, 3, 4, 6, 2, 7, 9, 4, 2, 6, 5, 2, 6, 4, 7, 2, 8, 1, 3, 0]
        ),
        "n2": torch.tensor([2, 4, 1, 3, 2, 5, 4, 1, 4, 3, 6, 0]),
        "n3": torch.tensor([1, 3, 2, 7]),
    }
    expected_csc_formats = {
        "n1:e1:n2": gb.CSCFormatBase(
            indptr=torch.tensor([0, 3, 4, 7, 10]),
            indices=torch.arange(0, 10),
        ),
        "n1:e2:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 1, 4, 7, 10]),
            indices=torch.arange(0, 10) + 10,
        ),
        "n2:e3:n3": gb.CSCFormatBase(
            indptr=torch.tensor([0, 2, 4, 6, 8]),
            indices=torch.arange(0, 8) + 4,
        ),
    }
    original_row_ids, compacted_csc_formats = gb.compact_csc_format(
        csc_formats, dst_nodes
    )

    for ntype, nodes in original_row_ids.items():
        expected_nodes = expected_original_row_ids[ntype]
        assert torch.equal(nodes, expected_nodes)
    for etype, csc_format in compacted_csc_formats.items():
        indptr = csc_format.indptr
        indices = csc_format.indices
        expected_indptr = expected_csc_formats[etype].indptr
        expected_indices = expected_csc_formats[etype].indices
        assert torch.equal(indptr, expected_indptr)
        assert torch.equal(indices, expected_indices)


def test_compact_csc_format_homo():
    seeds = torch.tensor([1, 3, 5, 2, 6])
    indptr = torch.tensor([0, 2, 4, 6, 7, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6])
    csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)

    expected_original_row_ids = torch.tensor(
        [1, 3, 5, 2, 6, 2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6]
    )
    expected_indptr = indptr
    expected_indices = torch.arange(0, len(indices)) + 5

    original_row_ids, compacted_csc_formats = gb.compact_csc_format(
        csc_formats, seeds
    )

    indptr = compacted_csc_formats.indptr
    indices = compacted_csc_formats.indices

    assert torch.equal(indptr, expected_indptr)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(original_row_ids, expected_original_row_ids)
