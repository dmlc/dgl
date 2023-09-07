import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
from torchdata.datapipes.iter import Mapper


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_Uniform_NegativeSampler(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = gb_test_utils.rand_csc_graph(100, 0.05)
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, num_seeds * 2).reshape(-1, 2), names="node_pairs"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        negative_ratio,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        pos_src, pos_dst = data.node_pairs
        neg_src, neg_dst = data.negative_srcs, data.negative_dsts
        # Assertation
        assert len(pos_src) == batch_size
        assert len(pos_dst) == batch_size
        assert len(neg_src) == batch_size
        assert len(neg_dst) == batch_size
        assert neg_src.numel() == batch_size * negative_ratio
        assert neg_dst.numel() == batch_size * negative_ratio
        expected_src = pos_src.repeat(negative_ratio).view(-1, negative_ratio)
        assert torch.equal(expected_src, neg_src)


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    metadata = gb.GraphMetadata(ntypes, etypes)
    indptr = torch.LongTensor([0, 2, 4, 6, 8, 10])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    return gb.from_csc(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        metadata=metadata,
    )


def test_NegativeSampler_Hetero_Data():
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T,
                names="node_pairs",
            ),
            "n2:e2:n1": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T,
                names="node_pairs",
            ),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    negative_dp = gb.UniformNegativeSampler(item_sampler, 1, graph)
    for neg in negative_dp:
        print(neg)
    assert len(list(negative_dp)) == 5
