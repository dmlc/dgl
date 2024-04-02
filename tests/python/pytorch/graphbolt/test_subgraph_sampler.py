import unittest
import warnings

from enum import Enum
from functools import partial

import backend as F

import dgl
import dgl.graphbolt as gb
import pytest
import torch

from . import gb_test_utils


# Skip all tests on GPU when sampling with TemporalNeighborSampler.
def _check_sampler_type(sampler_type):
    if F._default_context_str != "cpu" and sampler_type == SamplerType.Temporal:
        pytest.skip(
            "TemporalNeighborSampler sampling tests are only supported on CPU."
        )


def _check_sampler_len(sampler, lenExp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        assert len(list(sampler)) == lenExp


class SamplerType(Enum):
    Normal = 0
    Layer = 1
    Temporal = 2


def _get_sampler(sampler_type):
    if sampler_type == SamplerType.Normal:
        return gb.NeighborSampler
    if sampler_type == SamplerType.Layer:
        return gb.LayerNeighborSampler
    return partial(
        gb.TemporalNeighborSampler,
        node_timestamp_attr_name="timestamp",
        edge_timestamp_attr_name="timestamp",
    )


def test_SubgraphSampler_invoke_nodes():
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())

    # Invoke via class constructor.
    datapipe = gb.SubgraphSampler(item_sampler)
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))

    # Invokde via functional form.
    datapipe = item_sampler.sample_subgraph()
    with pytest.raises(NotImplementedError):
        next(iter(datapipe))


@pytest.mark.parametrize("labor", [False, True])
def test_NeighborSampler_invoke_nodes(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
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
def test_NeighborSampler_fanouts_nodes(labor):
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    itemset = gb.ItemSet(torch.arange(10), names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
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


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Node_seed_nodes(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(10)
    names = "seed_nodes"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = ("seed_nodes", "timestamp")
    itemset = gb.ItemSet(items, names=names)
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    _check_sampler_len(sampler_dp, 5)


def to_link_batch(data):
    block = gb.MiniBatch(node_pairs=data)
    return block


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_node_pairs(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(20).reshape(-1, 2)
    names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = ("node_pairs", "timestamp")
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_With_Negative_node_pairs(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(20).reshape(-1, 2)
    names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = ("node_pairs", "timestamp")
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


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


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Node_seed_nodes_Hetero(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items = torch.arange(3)
    names = "seed_nodes"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(0, 10, (3,)))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict({"n2": gb.ItemSet(items, names=names)})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    _check_sampler_len(sampler_dp, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for minibatch in sampler_dp:
            assert len(minibatch.sampled_subgraphs) == num_layer


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_node_pairs(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "node_pairs"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e2:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_With_Negative_node_pairs(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "node_pairs"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e2:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_Unknown_Etype_node_pairs(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "node_pairs"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    # "e11" and "e22" are not valid edge types.
    itemset = gb.ItemSetDict(
        {
            "n1:e11:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e22:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_With_Negative_Unknown_Etype_node_pairs(
    sampler_type,
):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "node_pairs"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "node_pairs"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    # "e11" and "e22" are not valid edge types.
    itemset = gb.ItemSetDict(
        {
            "n1:e11:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e22:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    _check_sampler_len(datapipe, 5)


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
@pytest.mark.parametrize(
    "replace",
    [False, True],
)
def test_SubgraphSampler_Random_Hetero_Graph_seed_ndoes(sampler_type, replace):
    _check_sampler_type(sampler_type)
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
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
    node_attributes = {}
    edge_attributes = {
        "A1": torch.randn(num_edges),
        "A2": torch.randn(num_edges),
    }
    if sampler_type == SamplerType.Temporal:
        node_attributes["timestamp"] = torch.randint(0, 10, (num_nodes,))
        edge_attributes["timestamp"] = torch.randint(0, 10, (num_edges,))
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    ).to(F.ctx())
    first_items = torch.tensor([0])
    first_names = "seed_nodes"
    second_items = torch.tensor([0])
    second_names = "seed_nodes"
    if sampler_type == SamplerType.Temporal:
        first_items = (first_items, torch.randint(0, 10, (1,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (1,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n2": gb.ItemSet(first_items, names=first_names),
            "n1": gb.ItemSet(second_items, names=second_names),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)

    sampler_dp = sampler(item_sampler, graph, fanouts, replace=replace)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for data in sampler_dp:
            for sampledsubgraph in data.sampled_subgraphs:
                for _, value in sampledsubgraph.sampled_csc.items():
                    for idx in [value.indices, value.indptr]:
                        assert torch.equal(
                            torch.ge(idx, torch.zeros(len(idx)).to(F.ctx())),
                            torch.ones(len(idx)).to(F.ctx()),
                        )
                node_ids = [
                    sampledsubgraph.original_column_node_ids,
                    sampledsubgraph.original_row_node_ids,
                ]
                for ids in node_ids:
                    for _, value in ids.items():
                        assert torch.equal(
                            torch.ge(
                                value, torch.zeros(len(value)).to(F.ctx())
                            ),
                            torch.ones(len(value)).to(F.ctx()),
                        )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Homo_seed_nodes(sampler_type):
    _check_sampler_type(sampler_type)
    graph = dgl.graph(
        ([5, 0, 1, 5, 6, 7, 2, 2, 4], [0, 1, 2, 2, 2, 2, 3, 4, 4])
    )
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])
    items = seed_nodes
    names = "seed_nodes"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (3,)))
        names = (names, "timestamp")

    itemset = gb.ItemSet(items, names=names)
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)

    length = [17, 7]
    compacted_indices = [
        (torch.arange(0, 10) + 7).to(F.ctx()),
        (torch.arange(0, 4) + 3).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6, 8, 10]).to(F.ctx()),
        torch.tensor([0, 1, 2, 4]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 2, 2, 3, 4, 4, 5]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for data in datapipe:
            for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
                assert (
                    len(sampled_subgraph.original_row_node_ids) == length[step]
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc.indices,
                    compacted_indices[step],
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc.indptr, indptr[step]
                )
                assert torch.equal(
                    torch.sort(sampled_subgraph.original_column_node_ids)[0],
                    seeds[step],
                )


def _assert_hetero_values(
    datapipe, original_row_node_ids, original_column_node_ids, csc_formats
):
    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            for ntype in ["n1", "n2"]:
                assert torch.equal(
                    sampled_subgraph.original_row_node_ids[ntype],
                    original_row_node_ids[step][ntype].to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype].to(F.ctx()),
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices.to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr.to(F.ctx()),
                )


def _assert_homo_values(
    datapipe, original_row_node_ids, compacted_indices, indptr, seeds
):
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


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Hetero_seed_nodes(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items = torch.arange(2)
    names = "seed_nodes"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (2,)))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict({"n2": gb.ItemSet(items, names=names)})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        _assert_hetero_values(
            datapipe,
            original_row_node_ids,
            original_column_node_ids,
            csc_formats,
        )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Fails due to different result on the GPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_cpu_seed_nodes(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 6, 7, 2, 2, 4], [0, 1, 2, 2, 3, 4, 4]))
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 5, 2, 6, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([3, 4, 4, 2, 5, 6]).to(F.ctx()),
        torch.tensor([3, 4, 4, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6]).to(F.ctx()),
        torch.tensor([0, 1, 2, 4]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
    _assert_homo_values(
        datapipe, original_row_node_ids, compacted_indices, indptr, seeds
    )


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Fails due to different result on the CPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_gpu_seed_nodes(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 7, 7, 2, 4], [0, 1, 2, 2, 3, 4]))
    graph = gb.from_dglgraph(graph, is_homogeneous=True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seed_nodes")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([-1]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 2, 5, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([4, 3, 2, 5, 5]).to(F.ctx()),
        torch.tensor([4, 3, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 3, 5, 5]).to(F.ctx()),
        torch.tensor([0, 1, 2, 3]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
    _assert_homo_values(
        datapipe, original_row_node_ids, compacted_indices, indptr, seeds
    )


@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Hetero_seed_nodes(labor):
    graph = get_hetero_graph().to(F.ctx())
    itemset = gb.ItemSetDict(
        {"n2": gb.ItemSet(torch.arange(2), names="seed_nodes")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
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
    _assert_hetero_values(
        datapipe, original_row_node_ids, original_column_node_ids, csc_formats
    )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Hetero_multifanout_per_layer_seed_nodes(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items_n1 = torch.tensor([0])
    items_n2 = torch.tensor([1])
    names = "seed_nodes"
    item_length = 2
    if sampler_type == SamplerType.Temporal:
        item_length = 3
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        # All edges can be sampled.
        items_n1 = (items_n1, torch.tensor([10]))
        items_n2 = (items_n2, torch.tensor([10]))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1": gb.ItemSet(items=items_n1, names=names),
            "n2": gb.ItemSet(items=items_n2, names=names),
        }
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    # The number of edges to be sampled for each edge types of each node.
    fanouts = [torch.LongTensor([2, 1]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    indices_len = [
        {
            "n1:e1:n2": 4,
            "n2:e2:n1": item_length,
        },
        {
            "n1:e1:n2": 2,
            "n2:e2:n1": 1,
        },
    ]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for minibatch in sampler_dp:
            for step, sampled_subgraph in enumerate(
                minibatch.sampled_subgraphs
            ):
                assert (
                    len(sampled_subgraph.sampled_csc["n1:e1:n2"].indices)
                    == indices_len[step]["n1:e1:n2"]
                )
                assert (
                    len(sampled_subgraph.sampled_csc["n2:e2:n1"].indices)
                    == indices_len[step]["n2:e2:n1"]
                )


def test_SubgraphSampler_invoke():
    itemset = gb.ItemSet(torch.arange(10), names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())

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
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    itemset = gb.ItemSet(torch.arange(10), names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
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
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    itemset = gb.ItemSet(torch.arange(10), names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
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


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Node(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(10)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = (names, "timestamp")
    itemset = gb.ItemSet(items, names=names)
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 5


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(20).reshape(-1, 2)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = (names, "timestamp")
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5
    for data in datapipe:
        assert torch.equal(
            data.compacted_seeds, torch.tensor([[0, 1], [2, 3]]).to(F.ctx())
        )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_With_Negative(sampler_type):
    _check_sampler_type(sampler_type)
    graph = gb_test_utils.rand_csc_graph(20, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    items = torch.arange(20).reshape(-1, 2)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {"timestamp": torch.arange(20).to(F.ctx())}
        graph.edge_attributes = {
            "timestamp": torch.arange(len(graph.indices)).to(F.ctx())
        }
        items = (items, torch.arange(10))
        names = (names, "timestamp")
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Node_Hetero(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items = torch.arange(3)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(0, 10, (3,)))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict({"n2": gb.ItemSet(items, names=names)})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    assert len(list(sampler_dp)) == 2
    for step, minibatch in enumerate(sampler_dp):
        assert len(minibatch.sampled_subgraphs) == num_layer


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "seeds"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e2:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5
    for data in datapipe:
        for compacted_seeds in data.compacted_seeds.values():
            if sampler_type == SamplerType.Temporal:
                assert torch.equal(
                    compacted_seeds, torch.tensor([[0, 0], [1, 1]]).to(F.ctx())
                )
            else:
                assert torch.equal(
                    torch.sort(compacted_seeds.T, dim=1)[0].T,
                    torch.tensor([[0, 0], [0, 1]]).to(F.ctx()),
                )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_With_Negative(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "seeds"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1:e1:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e2:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_Unknown_Etype(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "seeds"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    # "e11" and "e22" are not valid edge types.
    itemset = gb.ItemSetDict(
        {
            "n1:e11:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e22:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Link_Hetero_With_Negative_Unknown_Etype(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    first_items = torch.LongTensor([[0, 0, 1, 1], [0, 2, 0, 1]]).T
    first_names = "seeds"
    second_items = torch.LongTensor([[0, 0, 1, 1, 2, 2], [0, 1, 1, 0, 0, 1]]).T
    second_names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        first_items = (first_items, torch.randint(0, 10, (4,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (6,)))
        second_names = (second_names, "timestamp")
    # "e11" and "e22" are not valid edge types.
    itemset = gb.ItemSetDict(
        {
            "n1:e11:n2": gb.ItemSet(
                first_items,
                names=first_names,
            ),
            "n2:e22:n1": gb.ItemSet(
                second_items,
                names=second_names,
            ),
        }
    )

    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    datapipe = gb.UniformNegativeSampler(datapipe, graph, 1)
    sampler = _get_sampler(sampler_type)
    datapipe = sampler(datapipe, graph, fanouts)
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    assert len(list(datapipe)) == 5


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
@pytest.mark.parametrize(
    "replace",
    [False, True],
)
def test_SubgraphSampler_Random_Hetero_Graph(sampler_type, replace):
    _check_sampler_type(sampler_type)
    if F._default_context_str == "gpu" and replace == True:
        pytest.skip("Sampling with replacement not yet supported on GPU.")
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
    node_attributes = {}
    edge_attributes = {
        "A1": torch.randn(num_edges),
        "A2": torch.randn(num_edges),
    }
    if sampler_type == SamplerType.Temporal:
        node_attributes["timestamp"] = torch.randint(0, 10, (num_nodes,))
        edge_attributes["timestamp"] = torch.randint(0, 10, (num_edges,))
    graph = gb.fused_csc_sampling_graph(
        csc_indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=node_type_to_id,
        edge_type_to_id=edge_type_to_id,
        node_attributes=node_attributes,
        edge_attributes=edge_attributes,
    ).to(F.ctx())
    first_items = torch.tensor([0])
    first_names = "seeds"
    second_items = torch.tensor([0])
    second_names = "seeds"
    if sampler_type == SamplerType.Temporal:
        first_items = (first_items, torch.randint(0, 10, (1,)))
        first_names = (first_names, "timestamp")
        second_items = (second_items, torch.randint(0, 10, (1,)))
        second_names = (second_names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n2": gb.ItemSet(first_items, names=first_names),
            "n1": gb.ItemSet(second_items, names=second_names),
        }
    )

    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)

    sampler_dp = sampler(item_sampler, graph, fanouts, replace=replace)

    for data in sampler_dp:
        for sampledsubgraph in data.sampled_subgraphs:
            for _, value in sampledsubgraph.sampled_csc.items():
                assert torch.equal(
                    torch.ge(
                        value.indices,
                        torch.zeros(len(value.indices)).to(F.ctx()),
                    ),
                    torch.ones(len(value.indices)).to(F.ctx()),
                )
                assert torch.equal(
                    torch.ge(
                        value.indptr, torch.zeros(len(value.indptr)).to(F.ctx())
                    ),
                    torch.ones(len(value.indptr)).to(F.ctx()),
                )
            for _, value in sampledsubgraph.original_column_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value)).to(F.ctx())),
                    torch.ones(len(value)).to(F.ctx()),
                )
            for _, value in sampledsubgraph.original_row_node_ids.items():
                assert torch.equal(
                    torch.ge(value, torch.zeros(len(value)).to(F.ctx())),
                    torch.ones(len(value)).to(F.ctx()),
                )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Homo_Node(sampler_type):
    _check_sampler_type(sampler_type)
    graph = dgl.graph(
        ([5, 0, 1, 5, 6, 7, 2, 2, 4], [0, 1, 2, 2, 2, 2, 3, 4, 4])
    )
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])
    items = seed_nodes
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (3,)))
        names = (names, "timestamp")

    itemset = gb.ItemSet(items, names=names)
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)

    length = [17, 7]
    compacted_indices = [
        (torch.arange(0, 10) + 7).to(F.ctx()),
        (torch.arange(0, 4) + 3).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6, 8, 10]).to(F.ctx()),
        torch.tensor([0, 1, 2, 4]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 2, 2, 3, 4, 4, 5]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
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
                torch.sort(sampled_subgraph.original_column_node_ids)[0],
                seeds[step],
            )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Hetero_Node(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items = torch.arange(2)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (2,)))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict({"n2": gb.ItemSet(items, names=names)})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)
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
                    original_row_node_ids[step][ntype].to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype].to(F.ctx()),
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices.to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr.to(F.ctx()),
                )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Fails due to different result on the GPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_Node_cpu(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 6, 7, 2, 2, 4], [0, 1, 2, 2, 3, 4, 4]))
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 5, 2, 6, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([3, 4, 4, 2, 5, 6]).to(F.ctx()),
        torch.tensor([3, 4, 4, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6]).to(F.ctx()),
        torch.tensor([0, 1, 2, 4]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
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


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Fails due to different result on the CPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_Node_gpu(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 7, 7, 2, 4], [0, 1, 2, 2, 3, 4]))
    graph = gb.from_dglgraph(graph, is_homogeneous=True).to(F.ctx())
    seed_nodes = torch.LongTensor([0, 3, 4])

    itemset = gb.ItemSet(seed_nodes, names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=len(seed_nodes)).copy_to(
        F.ctx()
    )
    num_layer = 2
    fanouts = [torch.LongTensor([-1]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 2, 5, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([4, 3, 2, 5, 5]).to(F.ctx()),
        torch.tensor([4, 3, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 3, 5, 5]).to(F.ctx()),
        torch.tensor([0, 1, 2, 3]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
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
def test_SubgraphSampler_unique_csc_format_Hetero_Node(labor):
    graph = get_hetero_graph().to(F.ctx())
    itemset = gb.ItemSetDict({"n2": gb.ItemSet(torch.arange(2), names="seeds")})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
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
                    original_row_node_ids[step][ntype].to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype].to(F.ctx()),
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices.to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr.to(F.ctx()),
                )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_Hetero_multifanout_per_layer(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items_n1 = torch.tensor([0])
    items_n2 = torch.tensor([1])
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.arange(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.arange(graph.indices.numel()).to(F.ctx())
        }
        # All edges can be sampled.
        items_n1 = (items_n1, torch.tensor([10]))
        items_n2 = (items_n2, torch.tensor([10]))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict(
        {
            "n1": gb.ItemSet(items=items_n1, names=names),
            "n2": gb.ItemSet(items=items_n2, names=names),
        }
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    # The number of edges to be sampled for each edge types of each node.
    fanouts = [torch.LongTensor([2, 1]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    sampler_dp = sampler(item_sampler, graph, fanouts)
    if sampler_type == SamplerType.Temporal:
        indices_len = [
            {
                "n1:e1:n2": 4,
                "n2:e2:n1": 3,
            },
            {
                "n1:e1:n2": 2,
                "n2:e2:n1": 1,
            },
        ]
    else:
        indices_len = [
            {
                "n1:e1:n2": 4,
                "n2:e2:n1": 2,
            },
            {
                "n1:e1:n2": 2,
                "n2:e2:n1": 1,
            },
        ]
    for minibatch in sampler_dp:
        for step, sampled_subgraph in enumerate(minibatch.sampled_subgraphs):
            assert (
                len(sampled_subgraph.sampled_csc["n1:e1:n2"].indices)
                == indices_len[step]["n1:e1:n2"]
            )
            assert (
                len(sampled_subgraph.sampled_csc["n2:e2:n1"].indices)
                == indices_len[step]["n2:e2:n1"]
            )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Homo_Link(sampler_type):
    _check_sampler_type(sampler_type)
    graph = dgl.graph(
        ([5, 0, 1, 5, 6, 7, 2, 2, 4], [0, 1, 2, 2, 2, 2, 3, 4, 4])
    )
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([[0, 1], [3, 5]])
    items = seed_nodes
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (2,)))
        names = (names, "timestamp")

    itemset = gb.ItemSet(items, names=names)
    item_sampler = gb.ItemSampler(itemset, batch_size=4).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)

    length = [13, 7]
    compacted_indices = [
        (torch.arange(0, 6) + 7).to(F.ctx()),
        (torch.arange(0, 3) + 4).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 3, 3, 3, 4, 6]).to(F.ctx()),
        torch.tensor([0, 1, 2, 3, 3]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 0, 1, 2, 3, 5, 5]).to(F.ctx()),
        torch.tensor([0, 1, 3, 5]).to(F.ctx()),
    ]
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
                torch.sort(sampled_subgraph.original_column_node_ids)[0],
                seeds[step],
            )


@pytest.mark.parametrize(
    "sampler_type",
    [SamplerType.Normal, SamplerType.Layer, SamplerType.Temporal],
)
def test_SubgraphSampler_without_deduplication_Hetero_Link(sampler_type):
    _check_sampler_type(sampler_type)
    graph = get_hetero_graph().to(F.ctx())
    items = torch.arange(2).view(1, 2)
    names = "seeds"
    if sampler_type == SamplerType.Temporal:
        graph.node_attributes = {
            "timestamp": torch.zeros(graph.csc_indptr.numel() - 1).to(F.ctx())
        }
        graph.edge_attributes = {
            "timestamp": torch.zeros(graph.indices.numel()).to(F.ctx())
        }
        items = (items, torch.randint(1, 10, (1,)))
        names = (names, "timestamp")
    itemset = gb.ItemSetDict({"n1:e1:n2": gb.ItemSet(items, names=names)})
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = _get_sampler(sampler_type)
    if sampler_type == SamplerType.Temporal:
        datapipe = sampler(item_sampler, graph, fanouts)
    else:
        datapipe = sampler(item_sampler, graph, fanouts, deduplicate=False)
    csc_formats = [
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4, 6]),
                indices=torch.tensor([3, 4, 5, 6, 7, 8]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4, 6]),
                indices=torch.tensor([3, 4, 5, 6, 7, 8]),
            ),
        },
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2]),
                indices=torch.tensor([1, 2]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2]),
                indices=torch.tensor([1, 2], dtype=torch.int64),
            ),
        },
    ]
    original_column_node_ids = [
        {
            "n1": torch.tensor([0, 1, 0]),
            "n2": torch.tensor([1, 0, 2]),
        },
        {
            "n1": torch.tensor([0]),
            "n2": torch.tensor([1]),
        },
    ]
    original_row_node_ids = [
        {
            "n1": torch.tensor([0, 1, 0, 1, 0, 0, 1, 0, 1]),
            "n2": torch.tensor([1, 0, 2, 0, 2, 0, 1, 0, 2]),
        },
        {
            "n1": torch.tensor([0, 1, 0]),
            "n2": torch.tensor([1, 0, 2]),
        },
    ]

    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            for ntype in ["n1", "n2"]:
                assert torch.equal(
                    sampled_subgraph.original_row_node_ids[ntype],
                    original_row_node_ids[step][ntype].to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.original_column_node_ids[ntype],
                    original_column_node_ids[step][ntype].to(F.ctx()),
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices.to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr.to(F.ctx()),
                )


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Fails due to different result on the GPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_Link_cpu(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 6, 7, 2, 2, 4], [0, 1, 2, 2, 3, 4, 4]))
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    seed_nodes = torch.LongTensor([[0, 3], [4, 4]])

    itemset = gb.ItemSet(seed_nodes, names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=4).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 5, 2, 6, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([3, 4, 4, 2, 5, 6]).to(F.ctx()),
        torch.tensor([3, 4, 4, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 4, 4, 6]).to(F.ctx()),
        torch.tensor([0, 1, 2, 4]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
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


@unittest.skipIf(
    F._default_context_str == "cpu",
    reason="Fails due to different result on the CPU.",
)
@pytest.mark.parametrize("labor", [False, True])
def test_SubgraphSampler_unique_csc_format_Homo_Link_gpu(labor):
    torch.manual_seed(1205)
    graph = dgl.graph(([5, 0, 7, 7, 2, 4], [0, 1, 2, 2, 3, 4]))
    graph = gb.from_dglgraph(graph, is_homogeneous=True).to(F.ctx())
    seed_nodes = torch.LongTensor([[0, 3], [4, 4]])

    itemset = gb.ItemSet(seed_nodes, names="seeds")
    item_sampler = gb.ItemSampler(itemset, batch_size=4).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([-1]) for _ in range(num_layer)]

    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )

    original_row_node_ids = [
        torch.tensor([0, 3, 4, 2, 5, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([4, 3, 2, 5, 5]).to(F.ctx()),
        torch.tensor([4, 3, 2]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 3, 5, 5]).to(F.ctx()),
        torch.tensor([0, 1, 2, 3]).to(F.ctx()),
    ]
    seeds = [
        torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
        torch.tensor([0, 3, 4]).to(F.ctx()),
    ]
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
def test_SubgraphSampler_unique_csc_format_Hetero_Link(labor):
    graph = get_hetero_graph().to(F.ctx())
    itemset = gb.ItemSetDict(
        {"n1:e1:n2": gb.ItemSet(torch.tensor([[0, 1]]), names="seeds")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.LayerNeighborSampler if labor else gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )
    csc_formats = [
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4, 6]),
                indices=torch.tensor([1, 0, 0, 1, 0, 1]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([1, 2, 1, 0]),
            ),
        },
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2]),
                indices=torch.tensor([1, 0]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2]),
                indices=torch.tensor([1, 2], dtype=torch.int64),
            ),
        },
    ]
    original_column_node_ids = [
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1, 2]),
        },
        {
            "n1": torch.tensor([0]),
            "n2": torch.tensor([1]),
        },
    ]
    original_row_node_ids = [
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1, 2]),
        },
        {
            "n1": torch.tensor([0, 1]),
            "n2": torch.tensor([0, 1, 2]),
        },
    ]

    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            for ntype in ["n1", "n2"]:
                assert torch.equal(
                    torch.sort(sampled_subgraph.original_row_node_ids[ntype])[
                        0
                    ],
                    original_row_node_ids[step][ntype].to(F.ctx()),
                )
                assert torch.equal(
                    torch.sort(
                        sampled_subgraph.original_column_node_ids[ntype]
                    )[0],
                    original_column_node_ids[step][ntype].to(F.ctx()),
                )
            for etype in ["n1:e1:n2", "n2:e2:n1"]:
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indices,
                    csc_formats[step][etype].indices.to(F.ctx()),
                )
                assert torch.equal(
                    sampled_subgraph.sampled_csc[etype].indptr,
                    csc_formats[step][etype].indptr.to(F.ctx()),
                )
