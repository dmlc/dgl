import dgl.graphbolt as gb
import pytest
import torch


def rand_graph(num_nodes, num_edges):
    indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    indptr = torch.sort(indptr)[0]
    indptr[0] = 0
    indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return gb.from_csc(indptr, indices)


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Independent_Format(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = rand_graph(100, 500)

    # Construct NegativeSampler.
    negative_sampler = gb.PerSourceUniformSampler(
        graph,
        negative_ratio,
        gb.LinkedDataFormat.INDEPENDENT,
    )
    # Perform Negative sampling.
    num_seeds = 20
    pos_pairs = (
        torch.arange(0, num_seeds),
        torch.arange(num_seeds, num_seeds * 2),
    )
    src, dst, label = negative_sampler(pos_pairs)

    # Assertation
    assert len(src) == num_seeds * (negative_ratio + 1)
    assert len(dst) == num_seeds * (negative_ratio + 1)
    assert len(label) == num_seeds * (negative_ratio + 1)
    assert torch.all(torch.eq(label[:num_seeds], 1))
    assert torch.all(torch.eq(label[num_seeds:], 0))


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Conditioned_Format(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = rand_graph(100, 500)

    # Construct NegativeSampler.
    negative_sampler = gb.PerSourceUniformSampler(
        graph,
        negative_ratio,
        gb.LinkedDataFormat.CONDITIONED,
    )
    # Perform Negative sampling.
    num_seeds = 20
    pos_pairs = (
        torch.arange(0, num_seeds),
        torch.arange(num_seeds, num_seeds * 2),
    )
    pos_src, pos_dst, neg_src, neg_dst = negative_sampler(pos_pairs)

    # Assertation
    assert len(pos_src) == num_seeds
    assert len(pos_dst) == num_seeds
    assert len(neg_src) == num_seeds
    assert len(neg_dst) == num_seeds
    assert neg_src.numel() == num_seeds * negative_ratio
    assert neg_dst.numel() == num_seeds * negative_ratio
    expected_src = pos_src.repeat(negative_ratio).view(-1, negative_ratio)
    assert torch.equal(expected_src, neg_src)


if __name__ == "__main__":
    test_NegativeSampler_Independent_Format(1)
    test_NegativeSampler_Conditioned_Format(1)
