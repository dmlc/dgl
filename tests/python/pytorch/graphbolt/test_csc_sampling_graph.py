import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch

torch.manual_seed(3407)


def get_metadata(num_ntypes, num_etypes):
    ntypes = {f"n{i}": i for i in range(num_ntypes)}
    etypes = {}
    count = 0
    for n1 in range(num_ntypes):
        for n2 in range(n1, num_ntypes):
            if count >= num_etypes:
                break
            etypes.update({(f"n{n1}", f"e{count}", f"n{n2}"): count})
            count += 1
    return gb.GraphMetadata(ntypes, etypes)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_empty_graph(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    graph = gb.from_csc(csc_indptr, indices)
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [0, 1, 10, 100, 1000])
def test_hetero_empty_graph(num_nodes):
    csc_indptr = torch.zeros((num_nodes + 1,), dtype=int)
    indices = torch.tensor([])
    metadata = get_metadata(num_ntypes=3, num_etypes=5)
    # Some node types have no nodes.
    if num_nodes == 0:
        node_type_offset = torch.zeros((4,), dtype=int)
    else:
        node_type_offset = torch.sort(torch.randint(0, num_nodes, (4,)))[0]
        node_type_offset[0] = 0
        node_type_offset[-1] = num_nodes
    type_per_edge = torch.tensor([])
    graph = gb.from_csc(
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    )
    assert graph.num_edges == 0
    assert graph.num_nodes == num_nodes
    assert torch.equal(graph.csc_indptr, csc_indptr)
    assert torch.equal(graph.indices, indices)
    assert graph.metadata.node_type_to_id == metadata.node_type_to_id
    assert graph.metadata.edge_type_to_id == metadata.edge_type_to_id
    assert torch.equal(graph.node_type_offset, node_type_offset)
    assert torch.equal(graph.type_per_edge, type_per_edge)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "ntypes", [{"n1": 1, "n2": 1}, {5: 1, "n2": 2}, {"n1": 1.5, "n2": 2.0}]
)
def test_metadata_with_ntype_exception(ntypes):
    with pytest.raises(Exception):
        gb.GraphMetadata(ntypes, {("n1", "e1", "n2"): 1})


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "etypes",
    [
        {("n1", 5, "n12"): 1},
        {"e1": 1},
        {("n1", "e1"): 1},
        {("n1", "e1", 10): 1},
        {("n1", "e1", "n2"): 1, ("n1", "e2", "n3"): 1},
        {("n1", "e1", "n2"): 1, ("n1", "e1", "n3"): 2},
        {("n1", "e1", "n10"): 1},
        {("n1", "e1", "n2"): 1.5},
    ],
)
def test_metadata_with_etype_exception(etypes):
    with pytest.raises(Exception):
        gb.GraphMetadata({"n1": 0, "n2": 1, "n3": 2}, etypes)


def random_homo_graph(num_nodes, num_edges):
    csc_indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    csc_indptr = torch.sort(csc_indptr)[0]
    csc_indptr[0] = 0
    csc_indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return csc_indptr, indices


def random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    metadata = get_metadata(num_ntypes, num_etypes)
    # Randomly get node type split point.
    node_type_offset = torch.sort(
        torch.randint(0, num_nodes, (num_ntypes + 1,))
    )[0]
    node_type_offset[0] = 0
    node_type_offset[-1] = num_nodes

    type_per_edge = []
    for i in range(num_nodes):
        num = csc_indptr[i + 1] - csc_indptr[i]
        type_per_edge.append(
            torch.sort(torch.randint(0, num_etypes, (num,)))[0]
        )
    type_per_edge = torch.cat(type_per_edge, dim=0)
    return (csc_indptr, indices, node_type_offset, type_per_edge, metadata)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
def test_homo_graph(num_nodes, num_edges):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    graph = gb.from_csc(csc_indptr, indices)

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)

    assert graph.metadata is None
    assert graph.node_type_offset is None
    assert graph.type_per_edge is None


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "num_nodes, num_edges", [(1, 1), (100, 1), (10, 50), (1000, 50000)]
)
@pytest.mark.parametrize("num_ntypes, num_etypes", [(1, 1), (3, 5), (100, 1)])
def test_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        metadata,
    ) = random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes)
    graph = gb.from_csc(
        csc_indptr, indices, node_type_offset, type_per_edge, metadata
    )

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges

    assert torch.equal(csc_indptr, graph.csc_indptr)
    assert torch.equal(indices, graph.indices)
    assert torch.equal(node_type_offset, graph.node_type_offset)
    assert torch.equal(type_per_edge, graph.type_per_edge)
    assert metadata.node_type_to_id == graph.metadata.node_type_to_id
    assert metadata.edge_type_to_id == graph.metadata.edge_type_to_id


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "node_type_offset",
    [
        torch.tensor([0, 1]),
        torch.tensor([0, 1, 5, 6, 10]),
        torch.tensor([0, 1, 10]),
    ],
)
def test_node_type_offset_wrong_legnth(node_type_offset):
    num_ntypes = 3
    csc_indptr, indices, _, type_per_edge, metadata = random_hetero_graph(
        10, 50, num_ntypes, 5
    )
    with pytest.raises(Exception):
        gb.from_csc(
            csc_indptr, indices, node_type_offset, type_per_edge, metadata
        )


if __name__ == "__main__":
    test_empty_graph(10)
    test_node_type_offset_wrong_legnth(torch.tensor([0, 1, 5]))
    test_hetero_graph(10, 50, 3, 5)
