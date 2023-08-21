import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import Mapper


def to_node_block(data):
    block = gb.NodeClassificationBlock(seed_node=data)
    return block


def test_SubgraphSampler():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10))
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_node_block)
    sampler_dp = gb.NeighborSampler(data_block_converter, graph, fanouts)
    assert len(list(sampler_dp)) == 5


def to_link_block(data):
    block = gb.LinkPredictionBlock(node_pair=data)
    return block


def test_SubgraphSampler_link():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(
        (
            torch.arange(0, 10),
            torch.arange(10, 20),
        )
    )
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    neighbor_dp = gb.NeighborSampler(data_block_converter, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize(
    "format",
    [
        gb.LinkPredictionEdgeFormat.INDEPENDENT,
        gb.LinkPredictionEdgeFormat.CONDITIONED,
        gb.LinkPredictionEdgeFormat.HEAD_CONDITIONED,
        gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED,
    ],
)
def test_SubgraphSampler_link_with_Negative(format):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(
        (
            torch.arange(0, 10),
            torch.arange(10, 20),
        )
    )
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    negative_dp = gb.UniformNegativeSampler(
        data_block_converter, 1, format, graph
    )
    neighbor_dp = gb.NeighborSampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize(
    "format",
    [
        gb.LinkPredictionEdgeFormat.INDEPENDENT,
        gb.LinkPredictionEdgeFormat.CONDITIONED,
        gb.LinkPredictionEdgeFormat.HEAD_CONDITIONED,
        gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED,
    ],
)
def test_SubgraphSampler_link_hetero(format):
    ntypes = {"n1": 0, "n2": 1}
    etypes = {("n1", "e1", "n2"): 0, ("n2", "e2", "n1"): 1}
    metadata = gb.GraphMetadata(ntypes, etypes)
    indptr = torch.LongTensor([0, 2, 4, 6, 7, 9])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])

    # Construct CSCSamplingGraph.
    graph = gb.from_csc(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        metadata=metadata,
    )
    itemset = gb.ItemSet(
        (
            torch.arange(0, 10),
            torch.arange(10, 20),
        )
    )
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    negative_dp = gb.UniformNegativeSampler(
        data_block_converter, 1, format, graph
    )
    neighbor_dp = gb.NeighborSampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5