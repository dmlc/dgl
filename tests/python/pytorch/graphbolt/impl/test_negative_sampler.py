import re

import backend as F

import dgl.graphbolt as gb
import pytest
import torch

from .. import gb_test_utils


def test_NegativeSampler_invoke():
    # Instantiate graph and required datapipes.
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, 2 * num_seeds).reshape(-1, 2), names="seeds"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    negative_ratio = 2

    # Invoke NegativeSampler via class constructor.
    negative_sampler = gb.NegativeSampler(
        item_sampler,
        negative_ratio,
    )
    with pytest.raises(NotImplementedError):
        next(iter(negative_sampler))

    # Invoke NegativeSampler via functional form.
    negative_sampler = item_sampler.sample_negative(
        negative_ratio,
    )
    with pytest.raises(NotImplementedError):
        next(iter(negative_sampler))


def test_UniformNegativeSampler_invoke():
    # Instantiate graph and required datapipes.
    graph = gb_test_utils.rand_csc_graph(100, 0.05, bidirection_edge=True).to(
        F.ctx()
    )
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, 2 * num_seeds).reshape(-1, 2), names="seeds"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    negative_ratio = 2

    def _verify(negative_sampler):
        for data in negative_sampler:
            # Assertation
            seeds_len = batch_size + batch_size * negative_ratio
            assert data.seeds.size(0) == seeds_len
            assert data.labels.size(0) == seeds_len
            assert data.indexes.size(0) == seeds_len

    # Invoke UniformNegativeSampler via class constructor.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        graph,
        negative_ratio,
    )
    _verify(negative_sampler)

    # Invoke UniformNegativeSampler via functional form.
    negative_sampler = item_sampler.sample_uniform_negative(
        graph,
        negative_ratio,
    )
    _verify(negative_sampler)


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_Uniform_NegativeSampler(negative_ratio):
    # Construct FusedCSCSamplingGraph.
    graph = gb_test_utils.rand_csc_graph(100, 0.05, bidirection_edge=True).to(
        F.ctx()
    )
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, num_seeds * 2).reshape(-1, 2), names="seeds"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        graph,
        negative_ratio,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        seeds_len = batch_size + batch_size * negative_ratio
        # Assertation
        assert data.seeds.size(0) == seeds_len
        assert data.labels.size(0) == seeds_len
        assert data.indexes.size(0) == seeds_len
        # Check negative seeds value.
        pos_src = data.seeds[:batch_size, 0]
        neg_src = data.seeds[batch_size:, 0]
        assert torch.equal(pos_src.repeat_interleave(negative_ratio), neg_src)
        # Check labels.
        assert torch.equal(
            data.labels[:batch_size], torch.ones(batch_size).to(F.ctx())
        )
        assert torch.equal(
            data.labels[batch_size:],
            torch.zeros(batch_size * negative_ratio).to(F.ctx()),
        )
        # Check indexes.
        pos_indexes = torch.arange(0, batch_size).to(F.ctx())
        neg_indexes = pos_indexes.repeat_interleave(negative_ratio)
        expected_indexes = torch.cat((pos_indexes, neg_indexes))
        assert torch.equal(data.indexes, expected_indexes)


def test_Uniform_NegativeSampler_error_shape():
    # 1. seeds with shape N*3.
    # Construct FusedCSCSamplingGraph.
    graph = gb_test_utils.rand_csc_graph(100, 0.05, bidirection_edge=True).to(
        F.ctx()
    )
    num_seeds = 30
    item_set = gb.ItemSet(
        torch.arange(0, num_seeds * 3).reshape(-1, 3), names="seeds"
    )
    batch_size = 10
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    negative_ratio = 2
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        graph,
        negative_ratio,
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Only tensor with shape N*2 is "
            + "supported for negative sampling, but got torch.Size([10, 3])."
        ),
    ):
        next(iter(negative_sampler))

    # 2. seeds with shape N*2*1.
    # Construct FusedCSCSamplingGraph.
    item_set = gb.ItemSet(
        torch.arange(0, num_seeds * 2).reshape(-1, 2, 1), names="seeds"
    )
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        graph,
        negative_ratio,
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Only tensor with shape N*2 is "
            + "supported for negative sampling, but got torch.Size([10, 2, 1])."
        ),
    ):
        next(iter(negative_sampler))

    # 3. seeds with shape N.
    # Construct FusedCSCSamplingGraph.
    item_set = gb.ItemSet(torch.arange(0, num_seeds), names="seeds")
    item_sampler = gb.ItemSampler(item_set, batch_size=batch_size).copy_to(
        F.ctx()
    )
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        item_sampler,
        graph,
        negative_ratio,
    )
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Only tensor with shape N*2 is "
            + "supported for negative sampling, but got torch.Size([10])."
        ),
    ):
        next(iter(negative_sampler))


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1}
    etypes = {"n1:e1:n2": 0, "n2:e2:n1": 1}
    indptr = torch.LongTensor([0, 2, 4, 6, 8, 10])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    return gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
    )


def test_NegativeSampler_Hetero_Data():
    graph = get_hetero_graph().to(F.ctx())
    itemset = gb.HeteroItemSet(
        {
            "n1:e1:n2": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T,
                names="seeds",
            ),
            "n2:e2:n1": gb.ItemSet(
                torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T,
                names="seeds",
            ),
        }
    )
    batch_size = 2
    negative_ratio = 1
    item_sampler = gb.ItemSampler(itemset, batch_size=batch_size).copy_to(
        F.ctx()
    )
    negative_dp = gb.UniformNegativeSampler(item_sampler, graph, negative_ratio)
    assert len(list(negative_dp)) == 5
    # Perform negative sampling.
    expected_neg_src = [
        {"n1:e1:n2": torch.tensor([0, 0])},
        {"n1:e1:n2": torch.tensor([1, 1])},
        {"n2:e2:n1": torch.tensor([0, 0])},
        {"n2:e2:n1": torch.tensor([1, 1])},
        {"n2:e2:n1": torch.tensor([2, 2])},
    ]
    for i, data in enumerate(negative_dp):
        # Check negative seeds value.
        for etype, seeds_data in data.seeds.items():
            neg_src = seeds_data[batch_size:, 0]
            neg_dst = seeds_data[batch_size:, 1]
            assert torch.equal(expected_neg_src[i][etype].to(F.ctx()), neg_src)
            assert (neg_dst < 3).all(), neg_dst
