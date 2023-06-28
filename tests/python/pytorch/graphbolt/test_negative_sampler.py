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


def negative_sampler_func(
    node_pairs, graph, negative_ratio, linked_data_format
):
    src, dst = node_pairs
    label = torch.ones_like(src)
    shape, dtype = src.shape, src.dtype
    shape = (shape[0] * negative_ratio,)
    neg_src = src.repeat(negative_ratio)
    neg_dst = torch.randint(0, graph.num_nodes, shape, dtype=dtype)
    neg_label = torch.zeros_like(neg_src)
    src, dst = torch.cat([src, neg_src]), torch.cat([dst, neg_dst])
    label = torch.cat([label, neg_label])
    return (src, dst, label)


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler(negative_ratio):
    # Construct CSCSamplingGraph.
    graph = rand_graph(100, 500)

    # Construct mini-batch sampler
    num_seeds = 20
    node_pairs = (
        torch.arange(0, num_seeds),
        torch.arange(num_seeds, num_seeds * 2),
    )
    item_set = gb.ItemSet(node_pairs)
    batch_size = 4
    minibatch_sampler = gb.MinibatchSampler(
        item_set, batch_size=batch_size, shuffle=True, drop_last=False
    )
    negative_sampler = gb.NegativeSampler(
        minibatch_sampler,
        negative_sampler_func,
        graph,
        negative_ratio,
        gb.LinkedDataFormat.INDEPENDENT,
    )
    for i, (src, dst, labels) in enumerate(negative_sampler):
        is_last = (i + 1) * batch_size >= num_seeds
        if not is_last or num_seeds % batch_size == 0:
            expected_batch_size = batch_size
        else:
            expected_batch_size = num_seeds % batch_size
        assert len(src) == expected_batch_size * (negative_ratio + 1)
        assert len(dst) == expected_batch_size * (negative_ratio + 1)
        assert len(labels) == expected_batch_size * (negative_ratio + 1)
        assert torch.all(torch.eq(labels[:expected_batch_size], 1))
        assert torch.all(torch.eq(labels[expected_batch_size:], 0))


if __name__ == "__main__":
    test_NegativeSampler(1)
