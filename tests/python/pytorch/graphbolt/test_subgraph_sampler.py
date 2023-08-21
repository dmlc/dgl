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

@pytest.mark.parametrize(
    "format", [gb.LinkPredictionEdgeFormat.INDEPENDENT, gb.LinkPredictionEdgeFormat.CONDITIONED, gb.LinkPredictionEdgeFormat.HEAD_CONDITIONED, gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED]
)
def test_SubgraphSampler_with_Negative(format):
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
    negative_dp = gb.UniformNegativeSampler(data_block_converter, 1, format, graph)
    neighbor_dp = gb.NeighborSampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5

test_SubgraphSampler_with_Negative(gb.LinkPredictionEdgeFormat.CONDITIONED)