import dgl.graphbolt as gb
import torch

from . import gb_test_utils


def test_dgl_minibatch_converter():
    N = 32
    B = 4
    itemset = gb.ItemSet(torch.arange(N), names="seed_nodes")
    graph = gb_test_utils.rand_csc_graph(200, 0.15, bidirection_edge=True)

    features = {}
    keys = [("node", None, "a"), ("node", None, "b")]
    features[keys[0]] = gb.TorchBasedFeature(torch.randn(200, 4))
    features[keys[1]] = gb.TorchBasedFeature(torch.randn(200, 4))
    feature_store = gb.BasicFeatureStore(features)

    item_sampler = gb.ItemSampler(itemset, batch_size=B)
    subgraph_sampler = gb.NeighborSampler(
        item_sampler,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(2)],
    )
    feature_fetcher = gb.FeatureFetcher(
        subgraph_sampler,
        feature_store,
        ["a"],
    )
    dgl_converter = gb.DGLMiniBatchConverter(feature_fetcher)
    dataloader = gb.DataLoader(dgl_converter)
    assert len(list(dataloader)) == N // B
    minibatch = next(iter(dataloader))
    assert isinstance(minibatch, gb.DGLMiniBatch)
