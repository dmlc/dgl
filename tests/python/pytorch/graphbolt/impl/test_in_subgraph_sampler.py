import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch


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

    item_set = gb.ItemSet([0, 1, 2])
    batch_size = 1
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)

    adjacency_list = [
        graph.indices[graph.csc_indptr[i] : graph.csc_indptr[i + 1]]
        for i in range(len(graph.csc_indptr) - 1)
    ]

    for _, data in enumerate(in_subgraph_sampler):
        sampled_subgraph = data.sampled_subgraphs[0]
        src = [
            sampled_subgraph.original_row_node_ids[id]
            for id in sampled_subgraph.node_pairs[0]
        ]
        dst = [
            sampled_subgraph.original_column_node_ids[id]
            for id in sampled_subgraph.node_pairs[1]
        ]
        assert len(src) == len(dst)
        for i in range(len(src)):
            assert dst[i] in data.seed_nodes
            assert dst[i] in adjacency_list[src[i]]
        assert torch.equal(
            sampled_subgraph.original_row_node_ids, data.input_nodes
        )
