import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
import torchdata.datapipes as dp
from torchdata.datapipes.iter import Mapper


def to_node_block(data):
    block = gb.NodeClassificationBlock(seed_node=data)
    return block


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10))
    minibatch_dp = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_node_block)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(data_block_converter, graph, fanouts)
    assert len(list(sampler_dp)) == 5


def to_link_block(data):
    block = gb.LinkPredictionBlock(node_pair=data)
    return block


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(
        (
            torch.arange(0, 10),
            torch.arange(10, 20),
        )
    )
    minibatch_dp = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(data_block_converter, graph, fanouts)
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
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_With_Negative(format, labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(
        (
            torch.arange(0, 10),
            torch.arange(10, 20),
        )
    )
    minibatch_dp = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    negative_dp = gb.UniformNegativeSampler(
        data_block_converter, 1, format, graph
    )
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


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


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_Hetero(labor):
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

    minibatch_dp = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(data_block_converter, graph, fanouts)
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
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_Hetero_With_Negative(format, labor):
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

    minibatch_dp = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, to_link_block)
    negative_dp = gb.UniformNegativeSampler(
        data_block_converter, 1, format, graph
    )
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5
