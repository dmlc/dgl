import unittest
from functools import partial

import backend as F

import dgl
import dgl.graphbolt as gb
import pytest
import torch


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
    edge_attributes = {
        "weight": torch.FloatTensor(
            [2.5, 0, 8.4, 0, 0.4, 1.2, 2.5, 0, 8.4, 0.5]
        ),
        "mask": torch.BoolTensor([1, 0, 1, 0, 1, 1, 1, 0, 1, 1]),
    }
    node_type_offset = torch.LongTensor([0, 2, 5])
    return gb.fused_csc_sampling_graph(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        node_type_to_id=ntypes,
        edge_type_to_id=etypes,
        edge_attributes=edge_attributes,
    )


@unittest.skipIf(F._default_context_str != "gpu", reason="Enabled only on GPU.")
@pytest.mark.parametrize("hetero", [False, True])
@pytest.mark.parametrize("prob_name", [None, "weight", "mask"])
def test_NeighborSampler_GraphFetch(hetero, prob_name):
    items = torch.arange(3)
    names = "seed_nodes"
    itemset = gb.ItemSet(items, names=names)
    graph = get_hetero_graph().to(F.ctx())
    if hetero:
        itemset = gb.ItemSetDict({"n2": itemset})
    else:
        graph.type_per_edge = None
    item_sampler = gb.ItemSampler(itemset, batch_size=2).copy_to(F.ctx())
    fanout = torch.LongTensor([2])
    datapipe = item_sampler.map(gb.SubgraphSampler._preprocess)
    datapipe = datapipe.map(
        partial(gb.NeighborSampler._prepare, graph.node_type_to_id)
    )
    sample_per_layer = gb.SamplePerLayer(
        datapipe, graph.sample_neighbors, fanout, False, prob_name
    )
    compact_per_layer = sample_per_layer.compact_per_layer(True)
    gb.seed(123)
    expected_results = list(compact_per_layer)
    datapipe = gb.FetchInsubgraphData(datapipe, sample_per_layer)
    datapipe = datapipe.wait_future()
    datapipe = gb.SamplePerLayerFromFetchedSubgraph(datapipe, sample_per_layer)
    datapipe = datapipe.compact_per_layer(True)
    gb.seed(123)
    new_results = list(datapipe)
    assert len(expected_results) == len(new_results)
    for a, b in zip(expected_results, new_results):
        assert repr(a) == repr(b)
