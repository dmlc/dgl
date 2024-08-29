import re
import unittest

from functools import partial

import backend as F

import dgl
import dgl.graphbolt as gb
import pytest
import torch


def test_add_reverse_edges_homo():
    edges = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]).T
    combined_edges = gb.add_reverse_edges(edges)
    assert torch.equal(
        combined_edges,
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]]).T,
    )
    # Tensor with uncorrect dimensions.
    edges = torch.tensor([0, 1, 2, 3])
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Only tensor with shape N*2 is supported now, but got torch.Size([4])."
        ),
    ):
        gb.add_reverse_edges(edges)


def test_add_reverse_edges_hetero():
    # reverse_etype doesn't exist in original etypes.
    edges = {"n1:e1:n2": torch.tensor([[0, 1, 2], [4, 5, 6]]).T}
    reverse_etype_mapping = {"n1:e1:n2": "n2:e2:n1"}
    combined_edges = gb.add_reverse_edges(edges, reverse_etype_mapping)
    assert torch.equal(
        combined_edges["n1:e1:n2"], torch.tensor([[0, 1, 2], [4, 5, 6]]).T
    )
    assert torch.equal(
        combined_edges["n2:e2:n1"], torch.tensor([[4, 5, 6], [0, 1, 2]]).T
    )
    # reverse_etype exists in original etypes.
    edges = {
        "n1:e1:n2": torch.tensor([[0, 1, 2], [4, 5, 6]]).T,
        "n2:e2:n1": torch.tensor([[7, 8, 9], [10, 11, 12]]).T,
    }
    reverse_etype_mapping = {"n1:e1:n2": "n2:e2:n1"}
    combined_edges = gb.add_reverse_edges(edges, reverse_etype_mapping)
    assert torch.equal(
        combined_edges["n1:e1:n2"], torch.tensor([[0, 1, 2], [4, 5, 6]]).T
    )
    assert torch.equal(
        combined_edges["n2:e2:n1"],
        torch.tensor([[7, 8, 9, 4, 5, 6], [10, 11, 12, 0, 1, 2]]).T,
    )
    # Tensor with uncorrect dimensions.
    edges = {
        "n1:e1:n2": torch.tensor([0, 1, 2]),
        "n2:e2:n1": torch.tensor([7, 8, 9]),
    }
    with pytest.raises(
        AssertionError,
        match=re.escape(
            "Only tensor with shape N*2 is supported now, but got torch.Size([3])."
        ),
    ):
        gb.add_reverse_edges(edges, reverse_etype_mapping)


@unittest.skipIf(
    F._default_context_str == "gpu",
    reason="Fails due to different result on the GPU.",
)
@pytest.mark.parametrize("use_datapipe", [False, True])
def test_exclude_seed_edges_homo_cpu(use_datapipe):
    graph = dgl.graph(([5, 0, 6, 7, 2, 2, 4], [0, 1, 2, 2, 3, 4, 4]))
    graph = gb.from_dglgraph(graph, True).to(F.ctx())
    items = torch.LongTensor([[0, 3], [4, 4]])
    names = "seeds"
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    sampler = gb.NeighborSampler
    datapipe = sampler(datapipe, graph, fanouts)
    if use_datapipe:
        datapipe = datapipe.exclude_seed_edges()
    else:
        datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    original_row_node_ids = [
        torch.tensor([0, 3, 4, 5, 2, 6, 7]).to(F.ctx()),
        torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
    ]
    compacted_indices = [
        torch.tensor([3, 4, 4, 5, 6]).to(F.ctx()),
        torch.tensor([3, 4, 4]).to(F.ctx()),
    ]
    indptr = [
        torch.tensor([0, 1, 2, 3, 3, 5]).to(F.ctx()),
        torch.tensor([0, 1, 2, 3]).to(F.ctx()),
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
@pytest.mark.parametrize("use_datapipe", [False, True])
@pytest.mark.parametrize("async_op", [False, True])
def test_exclude_seed_edges_gpu(use_datapipe, async_op):
    graph = dgl.graph(([5, 0, 7, 7, 2, 4], [0, 1, 2, 2, 3, 4]))
    graph = gb.from_dglgraph(graph, is_homogeneous=True).to(F.ctx())
    items = torch.LongTensor([[0, 3], [4, 4]])
    names = "seeds"
    itemset = gb.ItemSet(items, names=names)
    datapipe = gb.ItemSampler(itemset, batch_size=4).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([-1]) for _ in range(num_layer)]
    sampler = gb.NeighborSampler
    datapipe = sampler(
        datapipe,
        graph,
        fanouts,
        deduplicate=True,
    )
    if use_datapipe:
        datapipe = datapipe.exclude_seed_edges(asynchronous=async_op)
    else:
        datapipe = datapipe.transform(
            partial(gb.exclude_seed_edges, async_op=async_op)
        )
    if torch.cuda.get_device_capability()[0] < 7:
        original_row_node_ids = [
            torch.tensor([0, 3, 4, 2, 5, 7]).to(F.ctx()),
            torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
        ]
        compacted_indices = [
            torch.tensor([4, 3, 5, 5]).to(F.ctx()),
            torch.tensor([4, 3]).to(F.ctx()),
        ]
        indptr = [
            torch.tensor([0, 1, 2, 2, 5, 5]).to(F.ctx()),
            torch.tensor([0, 1, 2, 2]).to(F.ctx()),
        ]
        seeds = [
            torch.tensor([0, 3, 4, 2, 5]).to(F.ctx()),
            torch.tensor([0, 3, 4]).to(F.ctx()),
        ]
    else:
        original_row_node_ids = [
            torch.tensor([0, 3, 4, 5, 2, 7]).to(F.ctx()),
            torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
        ]
        compacted_indices = [
            torch.tensor([3, 4, 5, 5]).to(F.ctx()),
            torch.tensor([3, 4]).to(F.ctx()),
        ]
        indptr = [
            torch.tensor([0, 1, 2, 2, 2, 4]).to(F.ctx()),
            torch.tensor([0, 1, 2, 2]).to(F.ctx()),
        ]
        seeds = [
            torch.tensor([0, 3, 4, 5, 2]).to(F.ctx()),
            torch.tensor([0, 3, 4]).to(F.ctx()),
        ]
    for data in datapipe:
        for step, sampled_subgraph in enumerate(data.sampled_subgraphs):
            if async_op and not use_datapipe:
                sampled_subgraph = sampled_subgraph.wait()
            assert torch.equal(
                sampled_subgraph.original_row_node_ids,
                original_row_node_ids[step],
            )
            assert torch.equal(
                (sampled_subgraph.sampled_csc.indices), compacted_indices[step]
            )
            assert torch.equal(
                sampled_subgraph.sampled_csc.indptr, indptr[step]
            )
            assert torch.equal(
                sampled_subgraph.original_column_node_ids, seeds[step]
            )


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


def test_exclude_seed_edges_hetero():
    graph = get_hetero_graph().to(F.ctx())
    itemset = gb.HeteroItemSet(
        {"n1:e1:n2": gb.ItemSet(torch.tensor([[0, 1]]), names="seeds")}
    )
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    Sampler = gb.NeighborSampler
    datapipe = Sampler(
        item_sampler,
        graph,
        fanouts,
        deduplicate=True,
    )
    datapipe = datapipe.transform(partial(gb.exclude_seed_edges))
    csc_formats = [
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 1, 3, 5]),
                indices=torch.tensor([1, 0, 1, 0, 1]),
            ),
            "n2:e2:n1": gb.CSCFormatBase(
                indptr=torch.tensor([0, 2, 4]),
                indices=torch.tensor([1, 2, 1, 0]),
            ),
        },
        {
            "n1:e1:n2": gb.CSCFormatBase(
                indptr=torch.tensor([0, 1]),
                indices=torch.tensor([1]),
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
