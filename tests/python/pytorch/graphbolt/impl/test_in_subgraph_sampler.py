import dgl.graphbolt as gb
import torch

from .. import gb_test_utils


def test_InSubgraphSampler_homo():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    """
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    graph = gb.from_fused_csc(indptr, indices)

    seed_nodes = torch.LongTensor([0, 5, 3])
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    batch_size = 1
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)

    it = iter(in_subgraph_sampler)

    def original_node_pairs(minibatch):
        sampled_subgraph = minibatch.sampled_subgraphs[0]
        _src = [
            sampled_subgraph.original_row_node_ids[id]
            for id in sampled_subgraph.node_pairs[0]
        ]
        _dst = [
            sampled_subgraph.original_column_node_ids[id]
            for id in sampled_subgraph.node_pairs[1]
        ]
        return _src, _dst

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([0]))
    assert original_node_pairs(mn) == ([0, 1, 4], [0, 0, 0])

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([5]))
    assert original_node_pairs(mn) == ([1, 4], [5, 5])

    mn = next(it)
    assert torch.equal(mn.seed_nodes, torch.LongTensor([3]))
    assert original_node_pairs(mn) == ([1, 2], [3, 3])


def test_InSubgraphSampler_hetero():
    """Original graph in COO:
    1   0   1   0   1   0
    1   0   0   1   0   1
    0   1   0   1   0   0
    0   1   0   0   1   0
    1   0   0   0   0   1
    0   0   1   0   1   0
    node_type_0: [0, 1, 2]
    node_type_1: [3, 4, 5]
    edge_type_0: node_type_0 -> node_type_0
    edge_type_1: node_type_0 -> node_type_1
    edge_type_2: node_type_1 -> node_type_0
    edge_type_3: node_type_1 -> node_type_1
    """
    ntypes = {
        "N0": 0,
        "N1": 1,
    }
    etypes = {
        "N0:R0:N0": 0,
        "N0:R1:N1": 1,
        "N1:R2:N0": 2,
        "N1:R3:N1": 3,
    }
    indptr = torch.LongTensor([0, 3, 5, 7, 9, 12, 14])
    indices = torch.LongTensor([0, 1, 4, 2, 3, 0, 5, 1, 2, 0, 3, 5, 1, 4])
    node_type_offset = torch.LongTensor([0, 3, 6])
    type_per_edge = torch.LongTensor([0, 0, 2, 0, 2, 0, 2, 1, 1, 1, 3, 3, 1, 3])
    graph = gb.from_fused_csc(
        csc_indptr=indptr,
        indices=indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )

    item_set = gb.ItemSetDict(
        {
            "N0": gb.ItemSet(torch.LongTensor([1, 0, 2]), names="seed_nodes"),
            "N1": gb.ItemSet(torch.LongTensor([0, 2, 1]), names="seed_nodes"),
        }
    )
    batch_size = 2
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)

    it = iter(in_subgraph_sampler)

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N0"], torch.LongTensor([1, 0]))
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([2, 1, 0]), torch.LongTensor([0, 1, 1])),
        "N0:R1:N1": (torch.LongTensor([]), torch.LongTensor([])),
        "N1:R2:N0": (torch.LongTensor([0, 1]), torch.LongTensor([0, 1])),
        "N1:R3:N1": (torch.LongTensor([]), torch.LongTensor([])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])

    mn = next(it)
    assert mn.seed_nodes == {
        "N0": torch.LongTensor([2]),
        "N1": torch.LongTensor([0]),
    }
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([1]), torch.LongTensor([0])),
        "N0:R1:N1": (torch.LongTensor([2, 0]), torch.LongTensor([0, 0])),
        "N1:R2:N0": (torch.LongTensor([1]), torch.LongTensor([0])),
        "N1:R3:N1": (torch.LongTensor([]), torch.LongTensor([])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])

    mn = next(it)
    assert torch.equal(mn.seed_nodes["N1"], torch.LongTensor([2, 1]))
    expected_node_pairs = {
        "N0:R0:N0": (torch.LongTensor([]), torch.LongTensor([])),
        "N0:R1:N1": (torch.LongTensor([0, 1]), torch.LongTensor([0, 1])),
        "N1:R2:N0": (torch.LongTensor([]), torch.LongTensor([])),
        "N1:R3:N1": (torch.LongTensor([1, 2, 0]), torch.LongTensor([0, 1, 1])),
    }
    for etype, pairs in mn.sampled_subgraphs[0].node_pairs.items():
        assert torch.equal(pairs[0], expected_node_pairs[etype][0])
        assert torch.equal(pairs[1], expected_node_pairs[etype][1])
