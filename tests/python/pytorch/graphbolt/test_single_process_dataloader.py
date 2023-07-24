import backend as F
import dgl
import dgl.graphbolt
import gb_test_utils
import torch


def test_DataLoader():
    N = 32
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N))
    graph = gb_test_utils.rand_csc_graph(200, 0.15)
    features = dgl.graphbolt.TorchBasedFeatureStore(torch.randn(200, 4))
    labels = dgl.graphbolt.TorchBasedFeatureStore(torch.randint(0, 10, (200,)))

    def sampler_func(data):
        adjs = []
        seeds = data

        for hop in range(2):
            sg = graph.sample_neighbors(seeds, torch.LongTensor([2]))
            seeds = sg.node_pairs[0]
            adjs.insert(0, sg)

        input_nodes = seeds
        output_nodes = data
        return input_nodes, output_nodes, adjs

    def fetch_func(data):
        input_nodes, output_nodes, adjs = data
        input_features = features.read(input_nodes)
        output_labels = labels.read(output_nodes)
        return input_features, output_labels, adjs

    minibatch_sampler = dgl.graphbolt.MinibatchSampler(itemset, batch_size=B)
    subgraph_sampler = dgl.graphbolt.SubgraphSampler(
        minibatch_sampler,
        sampler_func,
    )
    feature_fetcher = dgl.graphbolt.FeatureFetcher(subgraph_sampler, fetch_func)
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.SingleProcessDataLoader(device_transferrer)
    assert len(list(dataloader)) == N // B
