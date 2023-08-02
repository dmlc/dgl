import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Independent_Format(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = gb_test_utils.rand_csc_graph(100, 0.05)
    num_seeds = 30
    item_set = gb.ItemSet(
        (
            torch.arange(0, num_seeds),
            torch.arange(num_seeds, num_seeds * 2),
        )
    )
    batch_size = 10
    minibatch_sampler = gb.MinibatchSampler(item_set, batch_size=batch_size)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        minibatch_sampler,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.INDEPENDENT,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        src, dst, label = data
        # Assertation
        assert len(src) == batch_size * (negative_ratio + 1)
        assert len(dst) == batch_size * (negative_ratio + 1)
        assert len(label) == batch_size * (negative_ratio + 1)
        assert torch.all(torch.eq(label[:batch_size], 1))
        assert torch.all(torch.eq(label[batch_size:], 0))


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Conditioned_Format(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = gb_test_utils.rand_csc_graph(100, 0.05)
    num_seeds = 30
    item_set = gb.ItemSet(
        (
            torch.arange(0, num_seeds),
            torch.arange(num_seeds, num_seeds * 2),
        )
    )
    batch_size = 10
    minibatch_sampler = gb.MinibatchSampler(item_set, batch_size=batch_size)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        minibatch_sampler,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.CONDITIONED,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        pos_src, pos_dst, neg_src, neg_dst = data
        # Assertation
        assert len(pos_src) == batch_size
        assert len(pos_dst) == batch_size
        assert len(neg_src) == batch_size
        assert len(neg_dst) == batch_size
        assert neg_src.numel() == batch_size * negative_ratio
        assert neg_dst.numel() == batch_size * negative_ratio
        expected_src = pos_src.repeat(negative_ratio).view(-1, negative_ratio)
        assert torch.equal(expected_src, neg_src)
