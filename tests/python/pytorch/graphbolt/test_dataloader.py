import unittest

import backend as F

import dgl
import dgl.graphbolt
import pytest
import torch

import torchdata.dataloader2.graph as dp_utils

from . import gb_test_utils


def test_DataLoader():
    N = 40
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N), names="seed_nodes")
    graph = gb_test_utils.rand_csc_graph(200, 0.15, bidirection_edge=True)
    features = {}
    keys = [("node", None, "a"), ("node", None, "b")]
    features[keys[0]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    features[keys[1]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    feature_store = dgl.graphbolt.BasicFeatureStore(features)

    item_sampler = dgl.graphbolt.ItemSampler(itemset, batch_size=B)
    subgraph_sampler = dgl.graphbolt.NeighborSampler(
        item_sampler,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(2)],
    )
    feature_fetcher = dgl.graphbolt.FeatureFetcher(
        subgraph_sampler,
        feature_store,
        ["a", "b"],
    )
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.DataLoader(
        device_transferrer,
        num_workers=4,
    )
    assert len(list(dataloader)) == N // B


@unittest.skipIf(
    F._default_context_str != "gpu",
    reason="This test requires the GPU.",
)
@pytest.mark.parametrize(
    "sampler_name", ["NeighborSampler", "LayerNeighborSampler"]
)
@pytest.mark.parametrize("enable_feature_fetch", [True, False])
@pytest.mark.parametrize("overlap_feature_fetch", [True, False])
@pytest.mark.parametrize("overlap_graph_fetch", [True, False])
def test_gpu_sampling_DataLoader(
    sampler_name,
    enable_feature_fetch,
    overlap_feature_fetch,
    overlap_graph_fetch,
):
    N = 40
    B = 4
    num_layers = 2
    itemset = dgl.graphbolt.ItemSet(torch.arange(N), names="seed_nodes")
    graph = gb_test_utils.rand_csc_graph(200, 0.15, bidirection_edge=True).to(
        F.ctx()
    )
    features = {}
    keys = [("node", None, "a"), ("node", None, "b")]
    features[keys[0]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(200, 4, pin_memory=True)
    )
    features[keys[1]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(200, 4, pin_memory=True)
    )
    feature_store = dgl.graphbolt.BasicFeatureStore(features)

    datapipe = dgl.graphbolt.ItemSampler(itemset, batch_size=B)
    datapipe = datapipe.copy_to(F.ctx(), extra_attrs=["seed_nodes"])
    datapipe = getattr(dgl.graphbolt, sampler_name)(
        datapipe,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(num_layers)],
    )
    if enable_feature_fetch:
        datapipe = dgl.graphbolt.FeatureFetcher(
            datapipe,
            feature_store,
            ["a", "b"],
        )

    dataloader = dgl.graphbolt.DataLoader(
        datapipe,
        overlap_feature_fetch=overlap_feature_fetch,
        overlap_graph_fetch=overlap_graph_fetch,
    )
    bufferer_awaiter_cnt = int(enable_feature_fetch and overlap_feature_fetch)
    if overlap_graph_fetch:
        bufferer_awaiter_cnt += num_layers
    datapipe = dataloader.dataset
    datapipe_graph = dp_utils.traverse_dps(datapipe)
    awaiters = dp_utils.find_dps(
        datapipe_graph,
        dgl.graphbolt.Waiter,
    )
    assert len(awaiters) == bufferer_awaiter_cnt
    bufferers = dp_utils.find_dps(
        datapipe_graph,
        dgl.graphbolt.Bufferer,
    )
    assert len(bufferers) == bufferer_awaiter_cnt
    assert len(list(dataloader)) == N // B
