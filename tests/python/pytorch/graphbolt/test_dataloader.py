import os
import unittest
from sys import platform

import backend as F

import dgl
import dgl.graphbolt
import dgl.graphbolt as gb
import pytest
import torch
import torch.distributed as thd

from dgl.graphbolt.datapipes import find_dps, traverse_dps

from . import gb_test_utils


@pytest.mark.parametrize("overlap_feature_fetch", [False, True])
def test_DataLoader(overlap_feature_fetch):
    N = 40
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N), names="seeds")
    graph = gb_test_utils.rand_csc_graph(200, 0.15, bidirection_edge=True)
    features = {}
    keys = [("node", None, "a"), ("node", None, "b"), ("edge", None, "c")]
    features[keys[0]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    features[keys[1]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    M = graph.total_num_edges
    features[keys[2]] = dgl.graphbolt.TorchBasedFeature(torch.randn(M, 1))
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
        ["c"],
        overlap_fetch=overlap_feature_fetch,
    )
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.DataLoader(
        device_transferrer,
        num_workers=4,
    )
    for i, minibatch in enumerate(dataloader):
        assert "a" in minibatch.node_features
        assert "b" in minibatch.node_features
        for layer_id in range(minibatch.num_layers()):
            assert "c" in minibatch.edge_features[layer_id]
    assert i + 1 == N // B


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
@pytest.mark.parametrize("cooperative", [True, False])
@pytest.mark.parametrize("asynchronous", [True, False])
@pytest.mark.parametrize("num_gpu_cached_edges", [0, 1024])
@pytest.mark.parametrize("gpu_cache_threshold", [1, 3])
def test_gpu_sampling_DataLoader(
    sampler_name,
    enable_feature_fetch,
    overlap_feature_fetch,
    overlap_graph_fetch,
    cooperative,
    asynchronous,
    num_gpu_cached_edges,
    gpu_cache_threshold,
):
    if cooperative and not thd.is_initialized():
        # On Windows, the init method can only be file.
        init_method = (
            f"file:///{os.path.join(os.getcwd(), 'dis_tempfile')}"
            if platform == "win32"
            else "tcp://127.0.0.1:12345"
        )
        thd.init_process_group(
            init_method=init_method,
            world_size=1,
            rank=0,
        )
    N = 40
    B = 4
    num_layers = 2
    itemset = dgl.graphbolt.ItemSet(torch.arange(N), names="seeds")
    graph = gb_test_utils.rand_csc_graph(200, 0.15, bidirection_edge=True)
    graph = graph.pin_memory_() if overlap_graph_fetch else graph.to(F.ctx())
    features = {}
    keys = [
        ("node", None, "a"),
        ("node", None, "b"),
        ("node", None, "c"),
        ("edge", None, "d"),
    ]
    features[keys[0]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(200, 4, pin_memory=True)
    )
    features[keys[1]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(200, 4, pin_memory=True)
    )
    features[keys[2]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(200, 4, device=F.ctx())
    )
    features[keys[3]] = dgl.graphbolt.TorchBasedFeature(
        torch.randn(graph.total_num_edges, 1, device=F.ctx())
    )
    feature_store = dgl.graphbolt.BasicFeatureStore(features)

    dataloaders = []
    for i in range(2):
        datapipe = dgl.graphbolt.ItemSampler(itemset, batch_size=B)
        datapipe = datapipe.copy_to(F.ctx())
        kwargs = {
            "overlap_fetch": overlap_graph_fetch,
            "num_gpu_cached_edges": num_gpu_cached_edges,
            "gpu_cache_threshold": gpu_cache_threshold,
            "cooperative": cooperative,
            "asynchronous": asynchronous,
        }
        if i != 0:
            kwargs = {}
        datapipe = getattr(dgl.graphbolt, sampler_name)(
            datapipe,
            graph,
            fanouts=[torch.LongTensor([2]) for _ in range(num_layers)],
            **kwargs,
        )
        if enable_feature_fetch:
            datapipe = dgl.graphbolt.FeatureFetcher(
                datapipe,
                feature_store,
                ["a", "b", "c"],
                ["d"],
                overlap_fetch=overlap_feature_fetch and i == 0,
                cooperative=asynchronous and cooperative and i == 0,
            )
        dataloaders.append(dgl.graphbolt.DataLoader(datapipe))
    dataloader, dataloader2 = dataloaders

    bufferer_cnt = int(enable_feature_fetch and overlap_feature_fetch)
    if overlap_graph_fetch:
        bufferer_cnt += num_layers
        if num_gpu_cached_edges > 0:
            bufferer_cnt += 2 * num_layers
    if asynchronous:
        bufferer_cnt += 2 * num_layers + 1  # _preprocess stage has 1.
        if cooperative:
            bufferer_cnt += 3 * num_layers
            if enable_feature_fetch:
                bufferer_cnt += 1  # feature fetch has 1.
    if cooperative:
        # _preprocess stage.
        bufferer_cnt += 4
    datapipe_graph = traverse_dps(dataloader)
    bufferers = find_dps(
        datapipe_graph,
        dgl.graphbolt.Bufferer,
    )
    assert len(bufferers) == bufferer_cnt
    # Fixes the randomness of LayerNeighborSampler
    torch.manual_seed(1)
    minibatches = list(dataloader)
    assert len(minibatches) == N // B

    for i, _ in enumerate(dataloader):
        if i >= 1:
            break

    torch.manual_seed(1)

    for minibatch, minibatch2 in zip(minibatches, dataloader2):
        if enable_feature_fetch:
            assert "a" in minibatch.node_features
            assert "b" in minibatch.node_features
            assert "c" in minibatch.node_features
            if sampler_name == "LayerNeighborSampler":
                assert torch.equal(
                    minibatch.node_features["a"], minibatch2.node_features["a"]
                )
            for layer_id in range(minibatch.num_layers()):
                assert "d" in minibatch.edge_features[layer_id]
                edge_feature = minibatch.edge_features[layer_id]["d"]
                edge_feature_ref = minibatch2.edge_features[layer_id]["d"]
                if sampler_name == "LayerNeighborSampler":
                    assert torch.equal(edge_feature, edge_feature_ref)
    assert len(list(dataloader)) == N // B

    if asynchronous and cooperative:
        for minibatch in minibatches:
            x = torch.ones((minibatch.node_ids().size(0), 1), device=F.ctx())
            for subgraph in minibatch.sampled_subgraphs:
                x = gb.CooperativeConvFunction.apply(subgraph, x)
                x, edge_index, size = subgraph.to_pyg(x)
                x = x[0]
                one = torch.ones(
                    edge_index.shape[1], dtype=x.dtype, device=x.device
                )
                coo = torch.sparse_coo_tensor(
                    edge_index.flipud(), one, size=(size[1], size[0])
                )
                x = torch.sparse.mm(coo, x)
            assert x.shape[0] == minibatch.seeds.shape[0]
            assert x.shape[1] == 1

    if thd.is_initialized():
        thd.destroy_process_group()
