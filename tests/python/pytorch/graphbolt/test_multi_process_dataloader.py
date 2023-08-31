import os
import unittest

import backend as F

import dgl
import dgl.graphbolt
import gb_test_utils
import torch
from torchdata.datapipes.iter import Mapper


def test_DataLoader():
    N = 40
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N))
    graph = gb_test_utils.rand_csc_graph(200, 0.15)
    features = {}
    keys = [("node", None, "a"), ("node", None, "b")]
    features[keys[0]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    features[keys[1]] = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    feature_store = dgl.graphbolt.BasicFeatureStore(features)

    minibatch_sampler = dgl.graphbolt.MinibatchSampler(itemset, batch_size=B)
    block_converter = Mapper(minibatch_sampler, gb_test_utils.to_node_minibatch)
    subgraph_sampler = dgl.graphbolt.NeighborSampler(
        block_converter,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(2)],
    )
    feature_fetcher = dgl.graphbolt.FeatureFetcher(
        subgraph_sampler,
        feature_store,
        ["a", "b"],
    )
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.MultiProcessDataLoader(
        device_transferrer,
        num_workers=4,
    )
    assert len(list(dataloader)) == N // B
