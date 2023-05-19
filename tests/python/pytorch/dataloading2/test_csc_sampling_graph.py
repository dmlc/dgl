import unittest

import backend as F

import numpy as np
import pytest
import torch
import tempfile
import os

from dgl.dataloading2 import *


def random_heterogeneous_graph_from_csc(num_nodes, num_ntypes, num_etypes):
    # assume each node has 0 ~ 10 neighbors
    csc_indptr = torch.randint(0, 10, (num_nodes,))
    csc_indptr = torch.cumsum(csc_indptr, dim=0)
    csc_indptr = torch.cat((torch.tensor([0]), csc_indptr), dim=0)
    num_edges = csc_indptr[-1].item()
    indices = torch.randint(0, num_nodes, (num_edges,))
    ntypes = [f"n{i}" for i in range(num_ntypes)]
    etypes = [f"e{i}" for i in range(num_etypes)]
    # random get node type split point
    node_type_offset = torch.sort(torch.randperm(num_nodes)[:num_ntypes])[0]
    node_type_offset[0] = 0
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    return (
        from_csc(
            csc_indptr,
            indices,
            etype_sorted=False,
            hetero_info=(ntypes, etypes, node_type_offset, type_per_edge),
        ),
        csc_indptr,
        indices,
        type_per_edge,
    )


def random_heterogeneous_graph_from_coo(
    num_nodes, num_edges, num_ntypes, num_etypes
):
    coo = torch.randint(0, num_nodes, (2, num_edges))
    ntypes = [f"n{i}" for i in range(num_ntypes)]
    etypes = [f"e{i}" for i in range(num_etypes)]
    type_per_edge = torch.randint(0, num_etypes, (num_edges,))
    node_type_offset = torch.sort(torch.randperm(num_nodes)[:num_ntypes])[0]
    node_type_offset[0] = 0
    return (
        from_coo(
            coo,
            hetero_info=(ntypes, etypes, node_type_offset, type_per_edge),
        ),
        torch.cat([coo, type_per_edge.unsqueeze(dim=0)], dim=0),
    )


def sort_coo(coo: torch.tensor):
    row, col, etype = coo
    indices = torch.from_numpy(
        np.lexsort((etype.numpy(), row.numpy(), col.numpy()))
    )
    return torch.index_select(coo, 1, indices)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
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
    "num_nodes,num_edges", [(10, 20), (50, 300), (1000, 500000)]
)
def test_from_coo(num_nodes, num_edges, num_ntypes=3, num_etypes=5):
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
    orig_coo, coo = sort_coo(orig_coo), sort_coo(coo)

    assert graph.num_nodes == num_nodes
    assert graph.num_edges == num_edges
    assert coo.shape == orig_coo.shape
    assert torch.equal(orig_coo, coo)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes,num_ntypes,num_etypes", [(10, 1, 1), (50, 2, 2), (1000, 3, 5), (10000, 8, 10)])
def test_load_save_graph(num_nodes, num_ntypes, num_etypes):
    graph, _, _, _ = random_heterogeneous_graph_from_csc(num_nodes, num_ntypes, num_etypes)

    with tempfile.TemporaryDirectory() as test_dir:
        filename = os.path.join(test_dir, "csc_sampling_graph.pt")
        save_csc_sampling_graph(graph, filename)
        graph2 = load_csc_sampling_graph(filename)

        assert graph.num_nodes == graph2.num_nodes
        assert torch.equal(graph.csc_indptr, graph2.csc_indptr)
        assert torch.equal(graph.indices, graph2.indices)
        assert graph.is_heterogeneous == graph2.is_heterogeneous
        assert graph.node_types == graph2.node_types
        assert graph.edge_types == graph2.edge_types
        assert torch.equal(graph.node_type_offset, graph2.node_type_offset)
        assert torch.equal(graph.type_per_edge, graph2.type_per_edge)
