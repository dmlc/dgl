import unittest

import backend as F

import numpy as np
import pytest
import torch

from dgl.graphbolt import *

torch.manual_seed(42)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_empty_graph_from_coo(num_nodes):
    coo = torch.empty((2, 0), dtype=int)
    graph = from_coo(coo, num_nodes)
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert graph.indices.numel() == 0


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_empty_graph_from_csc(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    graph = from_csc(csc_indptr, indices)
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_empty_hetero_graph_from_csc(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    ntypes, etypes = get_ntypes_and_etypes(num_ntypes=3, num_etypes=5)
    # Some node types have no nodes.
    if num_nodes == 0:
        node_type_offset = torch.zeros((3,), dtype=int)
    else:
        node_type_offset = torch.sort(torch.randint(0, num_nodes, (3,)))[0]
        node_type_offset[0] = 0
    type_per_edge = torch.tensor([])
    graph = from_csc(
        csc_indptr,
        indices,
        hetero_info=HeteroInfo(ntypes, etypes, node_type_offset, type_per_edge),
    )
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)
    assert graph.node_types == ntypes
    assert graph.edge_types == etypes
    assert torch.equal(graph.node_type_offset, node_type_offset)
    assert torch.equal(graph.type_per_edge, type_per_edge)


def get_ntypes_and_etypes(num_ntypes, num_etypes):
    ntypes = [f"n{i}" for i in range(num_ntypes)]
    etypes = []
    count = 0
    for n1 in range(num_ntypes):
        for n2 in range(n1, num_ntypes):
            if count >= num_etypes:
                break
            etypes.append((f"n{n1}", f"e{count}", f"n{n2}"))
            count += 1
    return (ntypes, etypes)


def random_heterogeneous_graph_from_csc(num_nodes, num_ntypes, num_etypes):
    # assume each node has 0 ~ 10 neighbors
    csc_indptr = torch.randint(0, 10, (num_nodes,))
    csc_indptr = torch.cumsum(csc_indptr, dim=0)
    csc_indptr = torch.cat((torch.tensor([0]), csc_indptr), dim=0)
    num_edges = csc_indptr[-1].item()
    indices = torch.randint(0, num_nodes, (num_edges,))
    ntypes, etypes = get_ntypes_and_etypes(num_ntypes, num_etypes)
    # random get node type split point
    node_type_offset = torch.sort(torch.randint(0, num_nodes, (num_ntypes,)))[0]
    node_type_offset[0] = 0
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    return (
        from_csc(
            csc_indptr,
            indices,
            etype_sorted=False,
            hetero_info=HeteroInfo(
                ntypes, etypes, node_type_offset, type_per_edge
            ),
        ),
        csc_indptr,
        indices,
        type_per_edge,
    )


def random_heterogeneous_graph_from_coo(
    num_nodes, num_edges, num_ntypes, num_etypes
):
    coo = torch.randint(0, num_nodes, (2, num_edges))
    ntypes, etypes = get_ntypes_and_etypes(num_ntypes, num_etypes)
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    node_type_offset = torch.sort(torch.randint(0, num_nodes, (num_ntypes,)))[0]
    node_type_offset[0] = 0
    return (
        from_coo(
            coo,
            num_nodes,
            HeteroInfo(ntypes, etypes, node_type_offset, type_per_edge),
        ),
        torch.cat([coo, type_per_edge.unsqueeze(dim=0)], dim=0),
    )


def sort_coo(coo: torch.tensor):
    row, col = coo
    indices = torch.from_numpy(np.lexsort((row.numpy(), col.numpy())))
    return torch.index_select(coo, 1, indices)


def sort_coo_with_hetero(coo: torch.tensor):
    row, col, etype = coo
    indices = torch.from_numpy(
        np.lexsort((etype.numpy(), row.numpy(), col.numpy()))
    )
    return torch.index_select(coo, 1, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_ntypes, ntype_offset",
    [
        (1, torch.tensor([0], dtype=int)),
        (1, None),
        (3, torch.tensor([0, 2, 5], dtype=int)),
    ],
)
@pytest.mark.parametrize(
    "num_etypes, per_etype",
    [(1, torch.tensor([0, 0, 0], dtype=int)), (1, None)],
)
def test_hetero_info(num_ntypes, num_etypes, ntype_offset, per_etype):
    ntypes, etypes = get_ntypes_and_etypes(num_ntypes, num_etypes)
    hetero_info = HeteroInfo(ntypes, etypes, ntype_offset, per_etype)
    node_types, edge_types, node_type_offset, typer_per_edge = hetero_info
    assert ntypes == node_types
    assert etypes == edge_types
    if torch.is_tensor(ntype_offset):
        assert torch.equal(ntype_offset, node_type_offset)
    if torch.is_tensor(per_etype):
        assert torch.equal(per_etype, typer_per_edge)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "ntypes, ntype_offset",
    [
        (["n1"], torch.tensor([0, 2], dtype=int)),
        (["n1", "n2"], None),
        (["n1", "n2"], torch.tensor([2, 0], dtype=int)),
        (["n1", "n1"], torch.tensor([0, 2], dtype=int)),
    ],
)
@pytest.mark.parametrize(
    "etypes, per_etype",
    [
        (["e1"], torch.tensor([0], dtype=int)),
        ([("n1", "e1", "n2"), ("n1", "e2", "n3")], None),
        (
            [("n1", "e2", "n3"), ("n1", "e2", "n3")],
            torch.tensor([0, 1], dtype=int),
        ),
    ],
)
def test_hetero_info_with_exception(ntypes, etypes, ntype_offset, per_etype):
    try:
        HeteroInfo(ntypes, etypes, ntype_offset, per_etype)
        raise
    except:
        # Expceted
        pass


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(10, 0), (1, 1), (10, 50), (1000, 50000)]
)
def test_homogeneous_graph(num_nodes, num_edges):
    orig_coo = torch.randint(0, num_nodes, (2, num_edges))
    graph = from_coo(orig_coo, num_nodes)
    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges
    assert graph.is_heterogeneous == False

    csc_indptr = graph.csc_indptr
    col = csc_indptr[1:] - csc_indptr[:-1]
    col_indices = torch.nonzero(col).squeeze()
    col = col_indices.repeat_interleave(col[col_indices])
    coo = torch.stack([graph.indices, col], dim=0)
    orig_coo, coo = sort_coo(orig_coo), sort_coo(coo)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_ntypes, num_etypes", [(1, 1), (2, 1), (3, 3), (5, 1), (3, 5)]
)
@pytest.mark.parametrize("num_nodes", [10, 50, 1000, 10000])
def test_from_csc(num_nodes, num_ntypes, num_etypes):
    (
        graph,
        orig_csc_indptr,
        orig_indices,
        orig_type_per_edge,
    ) = random_heterogeneous_graph_from_csc(num_nodes, num_ntypes, num_etypes)

    csc_indptr = graph.csc_indptr
    indices = graph.indices
    type_per_edge = graph.type_per_edge

    assert torch.equal(csc_indptr, orig_csc_indptr)
    assert torch.equal(
        torch.sort(type_per_edge).values, torch.sort(orig_type_per_edge).values
    )
    assert torch.equal(
        torch.sort(indices).values, torch.sort(orig_indices).values
    )
    # check if the etype is sorted for each node
    for s, e in zip(csc_indptr[:-1], csc_indptr[1:]):
        etype = type_per_edge[s:e]
        assert torch.equal(etype, torch.sort(etype).values)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_ntypes, num_etypes", [(1, 1), (2, 1), (3, 3), (5, 1), (3, 5)]
)
@pytest.mark.parametrize(
    "num_nodes,num_edges", [(10, 20), (50, 300), (1000, 500000)]
)
def test_from_coo(num_nodes, num_edges, num_ntypes, num_etypes):
    graph, orig_coo = random_heterogeneous_graph_from_coo(
        num_nodes, num_edges, num_ntypes, num_etypes
    )

    csc_indptr = graph.csc_indptr
    indices = graph.indices
    type_per_edge = graph.type_per_edge

    col = csc_indptr[1:] - csc_indptr[:-1]
    col_indices = torch.nonzero(col).squeeze()
    col = col_indices.repeat_interleave(col[col_indices])
    coo = torch.stack([indices, col, type_per_edge], dim=0)
    orig_coo, coo = sort_coo_with_hetero(orig_coo), sort_coo_with_hetero(coo)

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges
    assert coo.shape == orig_coo.shape
    assert torch.equal(orig_coo, coo)


if __name__ == "__main__":
    test_from_csc(100000, 3, 5)
    test_from_coo(1000, 50000, 3, 5)
