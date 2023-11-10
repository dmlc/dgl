import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch

def test_InSubgraphSampler_homo():
    # Instantiate graph and required datapipes.
    num_seeds = 100
    item_set = gb.ItemSet(
        torch.arange(0, num_seeds), names="seed_nodes"
    )
    graph = gb_test_utils.rand_csc_graph(num_seeds, 0.1)

    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size, shuffle=True)
    in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)

    adjacency_list = [
        graph.indices[graph.csc_indptr[i]:graph.csc_indptr[i+1]]
        for i in range(len(graph.csc_indptr)-1)
    ]

    for _, data in enumerate(in_subgraph_sampler):
        assert len(data.seed_nodes) == batch_size
        sampled_subgraph = data.sampled_subgraphs[0]
        src = [sampled_subgraph.original_row_node_ids[id] for id in sampled_subgraph.node_pairs[0]]
        dst = [sampled_subgraph.original_column_node_ids[id] for id in sampled_subgraph.node_pairs[1]]
        assert len(src) == len(dst)
        for i in range(len(src)):
            assert dst[i] in adjacency_list[src[i]]
        assert torch.equal(sampled_subgraph.original_row_node_ids, data.input_nodes)
    