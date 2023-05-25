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
    sub_graph = graph.sample_etype_neighbors(
        seed_nodes, fanouts, replace=replace, return_eids=return_eids
    )
    num_nodes = seed_nodes.size(0)
    assert torch.equal(
        sub_graph.indptr, torch.zeros((num_nodes + 1,), dtype=int)
    )
    assert torch.equal(sub_graph.indices, torch.tensor([]))
    assert torch.equal(sub_graph.reverse_row_node_ids, seed_nodes)
    if return_eids:
        assert torch.equal(sub_graph.reverse_edge_ids, torch.tensor([]))
    else:
        assert sub_graph.reverse_edge_ids is None


def check_sample_basic(
    num_nodes, num_etypes, replace, neighbors_per_etype, fanouts
):
    seed_num = num_nodes // 5 + 1
    graph = random_graph_with_fixed_neighbors(
        num_nodes, neighbors_per_etype, 5, num_etypes
    )
    seed_nodes = torch.randint(0, num_nodes, (seed_num,))
    sub_graph = graph.sample_etype_neighbors(
        seed_nodes, fanouts, replace=replace, return_eids=True
    )

    # Dtype check
    assert sub_graph.indptr.dtype == graph.csc_indptr.dtype
    assert sub_graph.indices.dtype == graph.indices.dtype
    assert sub_graph.type_per_edge.dtype == graph.type_per_edge.dtype
    assert sub_graph.reverse_edge_ids.dtype == graph.csc_indptr.dtype

    eid = sub_graph.reverse_edge_ids

    assert torch.equal(seed_nodes, sub_graph.reverse_row_node_ids)
    assert sub_graph.indptr.is_contiguous()
    assert sub_graph.indptr[-1].item() == sub_graph.indices.size(0)

    expected_indices = torch.index_select(graph.indices, 0, eid)
    assert torch.equal(sub_graph.indices, expected_indices)
    expected_etypes = torch.index_select(graph.type_per_edge, 0, eid)
    assert torch.equal(sub_graph.type_per_edge, expected_etypes)

    return sub_graph


def check_sample_number_without_replace(
    sub_graph, fanouts, neighbors_per_etype
):
    num_etypes = fanouts.size(0)
    num_per_node = sub_graph.indptr[1:] - sub_graph.indptr[:-1]
    sampled_etypes = torch.stack(
        torch.split(sub_graph.type_per_edge, num_per_node.tolist())
    )
    # Check sample number.is expected
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
@pytest.mark.parametrize("fanout_expand_ratio", [1, 2, 5])
def test_sample_etype_without_replace(
    num_nodes, num_etypes, fanout_expand_ratio
):
    neighbors_per_etype = 5
    fanouts = torch.randint(
        0, neighbors_per_etype * fanout_expand_ratio, (num_etypes,)
    )
    sub_graph = check_sample_basic(
        num_nodes, num_etypes, False, neighbors_per_etype, fanouts
    )
    check_sample_number_without_replace(sub_graph, fanouts, neighbors_per_etype)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_nodes", [1, 10, 50, 1000])
@pytest.mark.parametrize("num_etypes", [1, 2, 5, 10])
@pytest.mark.parametrize("fanout_expand_ratio", [1, 2, 10, 100])
def test_sample_etype_with_replace(num_nodes, num_etypes, fanout_expand_ratio):
    neighbors_per_etype = 5
    fanouts = torch.randint(
        0, neighbors_per_etype * fanout_expand_ratio, (num_etypes,)
    )
    sub_graph = check_sample_basic(
        num_nodes, num_etypes, True, neighbors_per_etype, fanouts
    )
    num_per_node = sub_graph.indptr[1:] - sub_graph.indptr[:-1]
    sampled_etypes = torch.stack(
        torch.split(sub_graph.type_per_edge, num_per_node.tolist())
    )
    for etype in range(num_etypes):
        mask = sampled_etypes == etype
        etype_neighbor_num = torch.sum(mask, dim=1)
        num_pick = fanouts[etype].item()
        assert torch.all(etype_neighbor_num == num_pick)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "fanouts",
    [
        torch.tensor([2, 2, 3]),
        torch.tensor([-1, -1, -1]),
        torch.tensor([-1, 2, -1]),
        torch.tensor([-1, 10, 10]),
        torch.tensor([-1, 1, 10]),
        torch.tensor([10, 20, 10]),
        torch.tensor([10, 0, 2]),
    ],
)
def test_fanouts(fanouts):
    neighbors_per_etype = 5
    sub_graph = check_sample_basic(10, 3, False, neighbors_per_etype, fanouts)
    check_sample_number_without_replace(sub_graph, fanouts, neighbors_per_etype)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "fanouts",
    [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([0, 1]),
        torch.tensor([[1], [1], [2]]),
        torch.tensor([-2, 1, -1]),
        [1, 2, 3],
    ],
)
def test_sample_with_wrong_fanouts(fanouts):
    with pytest.raises(Exception):
        check_sample_basic(10, 3, False, 5, fanouts)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize(
    "seed_nodes",
    [
        torch.tensor([-1, 1, 2]),
        torch.tensor([1, 2, 1000]),
        torch.tensor([[1], [2]]),
        [1, 2, 3],
    ],
)
def test_sample_with_wrong_seed_nodes(seed_nodes):
    with pytest.raises(Exception):
        data = random_hetero_graph(6, 10, 3, 2)
        graph = gb.from_csc(*data)
        graph.sample_etype_neighbors(
            seed_nodes, torch.tensor([2,2]), False, return_eids=False
        )

@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("replace", [True, False])
@pytest.mark.parametrize(
    "probs",
    [
        torch.tensor([2.5, 0, 8.4, 0, 0.4, 1.2]),
        torch.tensor([True, True, False, True, True, False]),
    ],
)
def test_sample_with_probs(replace, probs):
    num_edges =6
    data = random_hetero_graph(5, num_edges, 3, 2)
    graph = gb.from_csc(*data)
    fanouts = torch.tensor([-1, -1])
    seed_nodes = torch.arange(5)
    sub_graph = graph.sample_etype_neighbors(
            seed_nodes, fanouts, probs, replace, return_eids=True
        )
    eid = sub_graph.reverse_edge_ids
    expected_eid = torch.nonzero(probs).squeeze(dim=1)
    assert torch.equal(torch.sort(eid)[0], expected_eid)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Graph is CPU only at present.",
)
@pytest.mark.parametrize("num_etypes", [3, 5, 10, 20])
def test_sample_with_part_type_edges_missing(num_etypes):
    num_nodes = 100
    neighbor_per_etype = 3
    num_edges = num_nodes * neighbor_per_etype * num_etypes
    csc_indptr, indices, node_type_offset, type_per_edge, metadata = random_hetero_graph(num_nodes, num_edges, 10, num_etypes)
    #  Make sure all ndoes have all etyes neighbors.
    csc_indptr = torch.arange(num_nodes + 1) * neighbor_per_etype * num_etypes
    type_per_edge = torch.arange(num_etypes).repeat_interleave(neighbor_per_etype).repeat(num_nodes)
    # Replace etype 1 edges with 2.
    type_per_edge = torch.where(type_per_edge == 1, 2, type_per_edge)
    graph = gb.from_csc(csc_indptr, indices, node_type_offset, type_per_edge, metadata)
    
    fanout_value = 10
    fanouts = torch.tensor([fanout_value]).repeat(num_etypes)
    seed_nodes = torch.arange(num_nodes)
    sub_graph = graph.sample_etype_neighbors(
            seed_nodes, fanouts, replace=True, return_eids=True
        )
    expected_num = num_nodes * fanout_value * (num_etypes - 1)
    sub_graph_node_num = sub_graph.indptr[-1].item()
    assert sub_graph_node_num == expected_num



if __name__ == "__main__":
    fanouts = torch.tensor([2, 1, -1])
    check_sample_basic(10, 3, False, 5, fanouts)
    test_sample_etype_without_replace(2, 5, 2)
    test_sample_with_probs(False, torch.tensor([2.5, 0, 8.4, 0, 0.4, 1.2]),)

