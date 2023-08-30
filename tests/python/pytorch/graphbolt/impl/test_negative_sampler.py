import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
from torchdata.datapipes.iter import Mapper


def to_data_block(data):
    return gb.LinkPredictionBlock(node_pair=data)


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
    data_block_converter = Mapper(minibatch_sampler, to_data_block)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        data_block_converter,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.INDEPENDENT,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        src, dst = data.node_pair
        label = data.label
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
    data_block_converter = Mapper(minibatch_sampler, to_data_block)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        data_block_converter,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.CONDITIONED,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        pos_src, pos_dst = data.node_pair
        neg_src, neg_dst = data.negative_head, data.negative_tail
        # Assertation
        assert len(pos_src) == batch_size
        assert len(pos_dst) == batch_size
        assert len(neg_src) == batch_size
        assert len(neg_dst) == batch_size
        assert neg_src.numel() == batch_size * negative_ratio
        assert neg_dst.numel() == batch_size * negative_ratio
        expected_src = pos_src.repeat(negative_ratio).view(-1, negative_ratio)
        assert torch.equal(expected_src, neg_src)


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Head_Conditioned_Format(negative_ratio):
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
    data_block_converter = Mapper(minibatch_sampler, to_data_block)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        data_block_converter,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.HEAD_CONDITIONED,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        pos_src, pos_dst = data.node_pair
        neg_src = data.negative_head
        # Assertation
        assert len(pos_src) == batch_size
        assert len(pos_dst) == batch_size
        assert len(neg_src) == batch_size
        assert neg_src.numel() == batch_size * negative_ratio
        expected_src = pos_src.repeat(negative_ratio).view(-1, negative_ratio)
        assert torch.equal(expected_src, neg_src)


@pytest.mark.parametrize("negative_ratio", [1, 5, 10, 20])
def test_NegativeSampler_Tail_Conditioned_Format(negative_ratio):
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
    data_block_converter = Mapper(minibatch_sampler, to_data_block)
    # Construct NegativeSampler.
    negative_sampler = gb.UniformNegativeSampler(
        data_block_converter,
        negative_ratio,
        gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED,
        graph,
    )
    # Perform Negative sampling.
    for data in negative_sampler:
        pos_src, pos_dst = data.node_pair
        neg_dst = data.negative_tail
        # Assertation
        assert len(pos_src) == batch_size
        assert len(pos_dst) == batch_size
        assert len(neg_dst) == batch_size
        assert neg_dst.numel() == batch_size * negative_ratio


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1}
    etypes = {("n1", "e1", "n2"): 0, ("n2", "e2", "n1"): 1}
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


def to_link_block(data):
    block = gb.LinkPredictionBlock(node_pair=data)
    return block


@pytest.mark.parametrize(
    "format",
    [
        gb.LinkPredictionEdgeFormat.INDEPENDENT,
        gb.LinkPredictionEdgeFormat.CONDITIONED,
        gb.LinkPredictionEdgeFormat.HEAD_CONDITIONED,
        gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED,
    ],
)
def test_NegativeSampler_Hetero_Data(format):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                (
                    torch.LongTensor([0, 0, 1, 1]),
                    torch.LongTensor([0, 2, 0, 1]),
                )
            ),
            "n2:e2:n1": gb.ItemSet(
                (
                    torch.LongTensor([0, 0, 1, 1, 2, 2]),
                    torch.LongTensor([0, 1, 1, 0, 0, 1]),
                )
            ),
        }
    )

    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    negative_dp = gb.UniformNegativeSampler(
        data_block_converter, 1, format, graph
    )
    assert len(list(negative_dp)) == 5
