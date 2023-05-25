import unittest

import backend as F

import dgl.graphbolt as gb

import pytest
import torch

# from .utils import random_hetero_graph, csc_to_coo
from .utils import random_graph_with_fixed_neighbors, random_hetero_graph


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "seed_nodes, fanouts",
    [
        (torch.tensor([]), torch.tensor([2, 2])),
        (torch.tensor([0, 2]), torch.tensor([0, 0])),
        (torch.tensor([], dtype=int), torch.tensor([0, 0])),
    ],
)
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize("return_eids", [True, False])
def test_return_empty_graph(seed_nodes, fanouts, replace, return_eids):
    data = random_hetero_graph(6, 10, 3, 2)
    graph = gb.from_csc(*data)
    sub_graph = graph.sample_etype_neighbors(seed_nodes, fanouts, replace=replace, return_eids=return_eids)
    num_nodes = seed_nodes.size(0)
    assert torch.equal(sub_graph.indptr, torch.zeros((num_nodes+1,), dtype=int))
    assert torch.equal(sub_graph.indices, torch.tensor([]))
    assert torch.equal(sub_graph.reverse_row_node_ids, seed_nodes)
    if return_eids:
        assert torch.equal(sub_graph.reverse_edge_ids, torch.tensor([]))
    else:
        assert sub_graph.reverse_edge_ids is None

def test_sample_basic(num_nodes, num_etypes, replace, neighbors_per_etype, fanout_expand_ratio=1):
    seed_num = num_nodes // 5 + 1
    fanouts = torch.randint(0, neighbors_per_etype * fanout_expand_ratio, (num_etypes,))
    graph = random_graph_with_fixed_neighbors(
        num_nodes, neighbors_per_etype, 5, num_etypes
    )
    seed_nodes = torch.randint(0, num_nodes, (seed_num,))
    sub_graph = graph.sample_etype_neighbors(
        seed_nodes, fanouts, replace=replace, return_eids=True
    )

    eid = sub_graph.reverse_edge_ids

    assert torch.equal(seed_nodes, sub_graph.reverse_row_node_ids)
    assert sub_graph.indptr.is_contiguous()

    expected_indices = torch.index_select(graph.indices, 0, eid)
    assert torch.equal(sub_graph.indices, expected_indices)
    expected_etypes = torch.index_select(graph.type_per_edge, 0, eid)
    assert torch.equal(sub_graph.type_per_edge, expected_etypes)

    return sub_graph, fanouts

@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [1, 10, 50, 1000])
@pytest.mark.parametrize("num_etypes", [1, 2, 5, 10])
@pytest.mark.parametrize("fanout_expand_ratio", [1, 2, 5])
def test_sample_etype_without_replace(
    num_nodes, num_etypes, fanout_expand_ratio
):
    neighbors_per_etype = 5
    sub_graph, fanouts = test_sample_basic(num_nodes, num_etypes, False, neighbors_per_etype, fanout_expand_ratio)
    num_per_node = sub_graph.indptr[1:] - sub_graph.indptr[:-1]
    sampled_etypes = torch.stack(
        torch.split(sub_graph.type_per_edge, num_per_node.tolist())
    )
    # Check sample number.
    for etype in range(num_etypes):
        mask = sampled_etypes == etype
        etype_neighbor_num = torch.sum(mask, dim=1)
        num_pick = fanouts[etype].item()
        if num_pick == -1 or num_pick >= neighbors_per_etype:
            assert torch.all(etype_neighbor_num == neighbors_per_etype)
        else:
            assert torch.all(etype_neighbor_num == num_pick)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [1, 10, 50, 1000])
@pytest.mark.parametrize("num_etypes", [1, 2, 5, 10])
@pytest.mark.parametrize("fanout_expand_ratio", [1, 2, 10, 100])
def test_sample_etype_with_replace(
    num_nodes, num_etypes, fanout_expand_ratio
):
    neighbors_per_etype = 5
    sub_graph, fanouts = test_sample_basic(num_nodes, num_etypes, True, neighbors_per_etype, fanout_expand_ratio)
    num_per_node = sub_graph.indptr[1:] - sub_graph.indptr[:-1]
    sampled_etypes = torch.stack(
        torch.split(sub_graph.type_per_edge, num_per_node.tolist())
    )
    for etype in range(num_etypes):
        mask = sampled_etypes == etype
        etype_neighbor_num = torch.sum(mask, dim=1)
        num_pick = fanouts[etype].item()
        assert torch.all(etype_neighbor_num == num_pick)

if __name__ == "__main__":
    fanouts = torch.tensor([5, 5, 5, 5, 5], dtype=int)
    # test_sample_etype_without_replace(0, 50, 5, 3, 5, fanouts)
    test_return_empty_graph(torch.tensor([0, 2]), torch.tensor([0, 0]), False, False)
