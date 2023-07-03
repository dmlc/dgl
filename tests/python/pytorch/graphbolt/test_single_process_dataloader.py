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
    feature_store = dgl.graphbolt.feature_store.TorchBasedFeatureStore(
        {
            "feature": torch.randn(200, 4),
            "label": torch.randint(0, 10, (200,)),
        }
    )

    def sampler_func(data):
        adjs = []
        seeds = data

        for hop in range(2):
            sg = graph.sample_neighbors(seeds, torch.LongTensor([2]))
            seeds = sg.indices
            adjs.insert(0, sg)

        input_nodes = seeds
        output_nodes = data
        return input_nodes, output_nodes, adjs

    def fetch_func(data):
        input_nodes, output_nodes, adjs = data
        input_features = feature_store.read("feature", input_nodes)
        output_labels = feature_store.read("label", output_nodes)
        return input_features, output_labels, adjs

    minibatch_dp = dgl.graphbolt.MinibatchSampler(itemset, batch_size=B)
    sampler_dp = dgl.graphbolt.SubgraphSampler(minibatch_dp, sampler_func)
    fetcher_dp = dgl.graphbolt.FeatureFetcher(sampler_dp, fetch_func)
    transfer_dp = dgl.graphbolt.CopyTo(fetcher_dp, F.ctx())

    dataloader = dgl.graphbolt.SingleProcessDataLoader(transfer_dp)
    assert len(list(dataloader)) == N // B
