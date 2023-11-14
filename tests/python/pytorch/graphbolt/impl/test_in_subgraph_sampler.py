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
    graph = gb.from_fused_csc(indptr, indices); print(graph)
    graph1 = gb_test_utils.rand_csc_graph(6, density=0.2)

    seed_nodes = torch.LongTensor([3])
    item_set = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_set1 = gb.ItemSet(seed_nodes, names="seed_nodes")
    batch_size = 1
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)
    item_sampler1 = gb.ItemSampler(item_set1, batch_size=batch_size)

    in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)
    in_subgraph_sampler1 = gb.InSubgraphSampler(item_sampler1, graph1)

    adjacency_list = [
        graph.indices[graph.csc_indptr[i] : graph.csc_indptr[i + 1]]
        for i in range(len(graph.csc_indptr) - 1)
    ]

    for _, data in enumerate(in_subgraph_sampler): print(data)
    print(graph1)
    for _, data in enumerate(in_subgraph_sampler1): print(data)

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
            assert dst[i] in adjacency_list[src[i]], f"{src}\n{dst}\n{adjacency_list}\n{sampled_subgraph}"
        assert torch.equal(
            sampled_subgraph.original_row_node_ids, data.input_nodes
        )




# def test_InSubgraphSampler_homo():
#     num_seeds = 100
#     item_set = gb.ItemSet(torch.arange(0, num_seeds), names="seed_nodes")
#     graph = gb_test_utils.rand_csc_graph(num_seeds, density=0.1)
#     batch_size = 10
#     item_sampler = gb.ItemSampler(item_set, batch_size=batch_size, shuffle=True)
#     in_subgraph_sampler = gb.InSubgraphSampler(item_sampler, graph)

#     adjacency_list = [
#         graph.indices[graph.csc_indptr[i] : graph.csc_indptr[i + 1]]
#         for i in range(len(graph.csc_indptr) - 1)
#     ]

#     for _, data in enumerate(in_subgraph_sampler):
#         assert len(data.seed_nodes) == batch_size
#         sampled_subgraph = data.sampled_subgraphs[0]
#         src = [
#             sampled_subgraph.original_row_node_ids[id]
#             for id in sampled_subgraph.node_pairs[0]
#         ]
#         dst = [
#             sampled_subgraph.original_column_node_ids[id]
#             for id in sampled_subgraph.node_pairs[1]
#         ]
#         assert len(src) == len(dst)
#         for i in range(len(src)):
#             assert dst[i] in data.seed_nodes
#             assert dst[i] in adjacency_list[src[i]]
#         assert torch.equal(
#             sampled_subgraph.original_row_node_ids, data.input_nodes
#         )