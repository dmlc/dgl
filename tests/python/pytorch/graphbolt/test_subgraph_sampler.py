from functools import partial

import dgl
import dgl.graphbolt as gb
import pytest
import torch
from torchdata.datapipes.iter import Mapper

from . import gb_test_utils


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
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True)
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
def test_NeighborSampler_fanouts(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True)
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2

    # `fanouts` is a list of tensors.
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5

    # `fanouts` is a list of integers.
    fanouts = [2 for _ in range(num_layer)]
    if labor:
        datapipe = item_sampler.sample_layer_neighbor(graph, fanouts)
    else:
        datapipe = item_sampler.sample_neighbor(graph, fanouts)
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True)
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
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    datapipe = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Link_With_Negative(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True)
    itemset = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2), names="node_pairs")
    datapipe = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


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


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Node_Hetero(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {"n2": gb.ItemSet(torch.arange(3), names="seed_nodes")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 2
    for minibatch in sampler_dp:
        assert len(minibatch.sampled_subgraphs) == num_layer


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

    datapipe = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


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

    datapipe = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_Random_Hetero_Graph(labor):
    num_nodes = 5
    num_edges = 9
    num_ntypes = 3
    num_etypes = 3
    (
        csc_indptr,
        indices,
        node_type_offset,
        type_per_edge,
        node_type_to_id,
        edge_type_to_id,
    ) = gb_test_utils.random_hetero_graph(
        num_nodes, num_edges, num_ntypes, num_etypes
    )
    edge_attributes = {
        "A1": torch.randn(num_edges),
        "A2": torch.randn(num_edges),
    }
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        edge_attributes=edge_attributes,
    )
    itemset = gb.ItemSetDict(
        {
            "n2": gb.ItemSet(torch.tensor([0]), names="seed_nodes"),
            "n1": gb.ItemSet(torch.tensor([0]), names="seed_nodes"),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    sampler_dp = Sampler(item_sampler, graph, fanouts, replace=True)

    for data in sampler_dp:
        for sampledsubgraph in data.sampled_subgraphs:
            for _, value in sampledsubgraph.sampled_csc.items():
                assert torch.equal(
                    torch.ge(value.indices, torch.zeros(len(value.indices))),
                    torch.ones(len(value.indices)),
                )
                assert torch.equal(
                    torch.ge(value.indptr, torch.zeros(len(value.indptr))),
                    torch.ones(len(value.indptr)),
                )
            for _, value in sampledsubgraph.original_column_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value))),
                    torch.ones(len(value)),
                )
            for _, value in sampledsubgraph.original_row_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value))),
                    torch.ones(len(value)),
                )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_without_dedpulication_Homo(labor):
    graph = dgl.graph(
        ([5, 0, 1, 5, 6, 7, 2, 2, 4], [0, 1, 2, 2, 2, 2, 3, 4, 4])
    )
    graph = gb.from_dglgraph(graph, True)
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes))
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(item_sampler, graph, fanouts, deduplicate=False)

    length = [17, 7]
    compacted_indices = [
        torch.arange(0, 10) + 7,
        torch.arange(0, 4) + 3,
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6, 8, 10]),
        torch.tensor([0, 1, 2, 4]),
    ]
    seeds = [torch.tensor([0, 3, 4, 5, 2, 2, 4]), torch.tensor([0, 3, 4])]
    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            assert len(sampled_subgraph.original_row_node_ids) == length[step]
            assert torch.equal(
                sampled_subgraph.sampled_csc.indices, compacted_indices[step]
            )
            assert torch.equal(
                sampled_subgraph.sampled_csc.indptr, indptr[step]
            )
            assert torch.equal(
                sampled_subgraph.original_column_node_ids, seeds[step]
            )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_without_dedpulication_Hetero(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {"n2": gb.ItemSet(torch.arange(2), names="seed_nodes")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(item_sampler, graph, fanouts, deduplicate=False)
    csc_formats = [
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([4, 5, 6, 7]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4, 6, 8]),
                indices=torch.tensor([2, 3, 4, 5, 6, 7, 8, 9]),
            ),
        },
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([0, 1, 2, 3]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0]),
                indices=torch.tensor([], dtype=torch.int64),
            ),
        },
    ]
    original_column_node_ids = [
        {
            "n1": torch.tensor([0, 1, 1, 0]),
            "n2": torch.tensor([0, 1]),
        },
        {
            "n1": torch.tensor([], dtype=torch.int64),
            "n2": torch.tensor([0, 1]),
        },
    ]
    original_row_node_ids = [
        {
            "n1": torch.tensor([0, 1, 1, 0, 0, 1, 1, 0]),
            "n2": torch.tensor([0, 1, 0, 2, 0, 1, 0, 1, 0, 2]),
        },
        {
            "n1": torch.tensor([0, 1, 1, 0]),
            "n2": torch.tensor([0, 1]),
        },
    ]

    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            for ntype in ["n1", "n2"]:
                assert torch.equal(
                    sampled_subgraph.original_row_node_ids[ntype],
                    original_row_node_ids[step][ntype],
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype],
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices,
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr,
                )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 6, 7, 2, 2, 4], [0, 1, 2, 2, 3, 4, 4]))
    graph = gb.from_dglgraph(graph, True)
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes))
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        replace=False,
        deduplicate=True,
        output_cscformat=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 5, 2, 6, 7]),
        torch.tensor([0, 3, 4, 5, 2]),
    ]
    compacted_indices = [
        torch.tensor([3, 4, 4, 2, 5, 6]),
        torch.tensor([3, 4, 4, 2]),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6]),
        torch.tensor([0, 1, 2, 4]),
    ]
    seeds = [torch.tensor([0, 3, 4, 5, 2]), torch.tensor([0, 3, 4])]
    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            assert torch.equal(
                sampled_subgraph.original_row_node_ids,
                original_row_node_ids[step],
            )
            assert torch.equal(
                sampled_subgraph.sampled_csc.indices, compacted_indices[step]
            )
            assert torch.equal(
                sampled_subgraph.sampled_csc.indptr, indptr[step]
            )
            assert torch.equal(
                sampled_subgraph.original_column_node_ids, seeds[step]
            )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Hetero(labor):
    graph = get_hetero_graph()
    itemset = gb.ItemSetDict(
        {"n2": gb.ItemSet(torch.arange(2), names="seed_nodes")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
        output_cscformat=True,
    )
    csc_formats = [
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([0, 1, 1, 0]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([0, 2, 0, 1]),
            ),
        },
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([0, 1, 1, 0]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0]),
                indices=torch.tensor([], dtype=torch.int64),
            ),
        },
    ]
    original_column_node_ids = [
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1]),
        },
        {
            "n1": torch.tensor([], dtype=torch.int64),
            "n2": torch.tensor([0, 1]),
        },
    ]
    original_row_node_ids = [
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1, 2]),
        },
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1]),
        },
    ]

    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            for ntype in ["n1", "n2"]:
                assert torch.equal(
                    sampled_subgraph.original_row_node_ids[ntype],
                    original_row_node_ids[step][ntype],
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype],
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices,
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr,
                )
