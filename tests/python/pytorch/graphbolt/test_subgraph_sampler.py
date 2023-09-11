import dgl.graphbolt as gb
import gb_test_utils
import pytest
import torch
from torchdata.datapipes.iter import Mapper


def test_SubgraphSampler_invoke():
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)

    # Invoke via class constructor.
    datapipe = gb.SubgraphSampler(item_sampler)
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))

    # Invokde via functional form.
    datapipe = item_sampler.sample_subgraph()
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))


@pytest.mark.parametrize("labor", [False, True])
def test_NeighborSampler_invoke(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    # Invoke via class constructor.
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(item_sampler, graph, fanouts)
    assert len(list(datapipe)) == 5

    # Invokde via functional form.
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 5


def to_link_batch(data):
    block = gb.MiniBatch(node_pairs=data)
    return block


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_With_Negative(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    negative_dp = gb.UniformNegativeSampler(item_sampler, 1, graph)
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
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(neighbor_dp)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_Hetero_With_Negative(labor):
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
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    negative_dp = gb.UniformNegativeSampler(item_sampler, 1, graph)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    neighbor_dp = Sampler(negative_dp, graph, fanouts)
    assert len(list(neighbor_dp)) == 5
