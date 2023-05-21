import unittest

import backend as F

import numpy as np
import pytest
import torch

from dgl.graphbolt import *


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
def test_from_csc(num_nodes, num_ntypes=3, num_etypes=5):
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


if __name__ == "__main__":
    test_from_csc(100000)
    test_from_coo(1000, 50000)
