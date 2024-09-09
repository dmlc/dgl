import backend as F
import dgl.graphbolt as gb
import pytest
import torch


def test_unique_and_compact_hetero():
    N1 = torch.tensor(
        [0, 5, 2, 7, 12, 7, 9, 5, 6, 2, 3, 4, 1, 0, 9], device=F.ctx()
    )
    N2 = torch.tensor([0, 3, 3, 5, 2, 7, 2, 8, 4, 9, 2, 3], device=F.ctx())
    N3 = torch.tensor([1, 2, 6, 6, 1, 8, 3, 6, 3, 2], device=F.ctx())
    expected_unique = {
        "n1": torch.tensor([0, 5, 2, 7, 12, 9, 6, 3, 4, 1], device=F.ctx()),
        "n2": torch.tensor([0, 3, 5, 2, 7, 8, 4, 9], device=F.ctx()),
        "n3": torch.tensor([1, 2, 6, 8, 3], device=F.ctx()),
    }
    if N1.is_cuda and torch.cuda.get_device_capability()[0] < 7:
        expected_reverse_id = {
            k: v.sort()[1] for k, v in expected_unique.items()
        }
        expected_unique = {k: v.sort()[0] for k, v in expected_unique.items()}
    else:
        expected_reverse_id = {
            k: torch.arange(0, v.shape[0], device=F.ctx())
            for k, v in expected_unique.items()
        }
    nodes_dict = {
        "n1": N1.split(5),
        "n2": N2.split(4),
        "n3": N3.split(2),
    }
    expected_nodes_dict = {
        "n1": [
            torch.tensor([0, 1, 2, 3, 4], device=F.ctx()),
            torch.tensor([3, 5, 1, 6, 2], device=F.ctx()),
            torch.tensor([7, 8, 9, 0, 5], device=F.ctx()),
        ],
        "n2": [
            torch.tensor([0, 1, 1, 2], device=F.ctx()),
            torch.tensor([3, 4, 3, 5], device=F.ctx()),
            torch.tensor([6, 7, 3, 1], device=F.ctx()),
        ],
        "n3": [
            torch.tensor([0, 1], device=F.ctx()),
            torch.tensor([2, 2], device=F.ctx()),
            torch.tensor([0, 3], device=F.ctx()),
            torch.tensor([4, 2], device=F.ctx()),
            torch.tensor([4, 1], device=F.ctx()),
        ],
    }

    unique, compacted, _ = gb.unique_and_compact(nodes_dict)
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
    N = torch.tensor(
        [0, 5, 2, 7, 12, 7, 9, 5, 6, 2, 3, 4, 1, 0, 9], device=F.ctx()
    )
    expected_unique_N = torch.tensor(
        [0, 5, 2, 7, 12, 9, 6, 3, 4, 1], device=F.ctx()
    )
    if N.is_cuda and torch.cuda.get_device_capability()[0] < 7:
        expected_reverse_id_N = expected_unique_N.sort()[1]
        expected_unique_N = expected_unique_N.sort()[0]
    else:
        expected_reverse_id_N = torch.arange(
            0, expected_unique_N.shape[0], device=F.ctx()
        )
    nodes_list = N.split(5)
    expected_nodes_list = [
        torch.tensor([0, 1, 2, 3, 4], device=F.ctx()),
        torch.tensor([3, 5, 1, 6, 2], device=F.ctx()),
        torch.tensor([7, 8, 9, 0, 5], device=F.ctx()),
    ]

    unique, compacted, _ = gb.unique_and_compact(nodes_list)

    assert torch.equal(unique, expected_unique_N)
    assert isinstance(compacted, list)
    for expected_node, node in zip(expected_nodes_list, compacted):
        node = expected_reverse_id_N[node]
        assert torch.equal(expected_node, node)


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

    unique_nodes, compacted_csc_formats, _ = gb.unique_and_compact_csc_formats(
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
    indptr = torch.tensor([0, 2, 4, 6, 7, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6])
    csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)

    expected_unique_nodes = torch.tensor([1, 3, 5, 2, 6, 4])
    expected_indptr = indptr
    expected_indices = torch.tensor([3, 1, 0, 5, 2, 3, 2, 0, 5, 5, 4])

    unique_nodes, compacted_csc_formats, _ = gb.unique_and_compact_csc_formats(
        csc_formats, seeds
    )

    indptr = compacted_csc_formats.indptr
    indices = compacted_csc_formats.indices
    assert torch.equal(indptr, expected_indptr)
    assert torch.equal(indices, expected_indices)
    assert torch.equal(unique_nodes, expected_unique_nodes)


def test_unique_and_compact_incorrect_indptr():
    seeds = torch.tensor([1, 3, 5, 2, 6, 7])
    indptr = torch.tensor([0, 2, 4, 6, 7, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6])
    csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)

    # The number of seeds is not corresponding to indptr.
    with pytest.raises(AssertionError):
        gb.unique_and_compact_csc_formats(csc_formats, seeds)


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


def test_compact_incorrect_indptr():
    seeds = torch.tensor([1, 3, 5, 2, 6, 7])
    indptr = torch.tensor([0, 2, 4, 6, 7, 11])
    indices = torch.tensor([2, 3, 1, 4, 5, 2, 5, 1, 4, 4, 6])
    csc_formats = gb.CSCFormatBase(indptr=indptr, indices=indices)

    # The number of seeds is not corresponding to indptr.
    with pytest.raises(AssertionError):
        gb.compact_csc_format(csc_formats, seeds)
