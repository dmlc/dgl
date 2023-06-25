import backend as F
import dgl
import dgl.graphbolt
import torch


def test_DataLoader():
    N = 32
    B = 4
    itemset = torch.arange(N)
    graph = dgl.add_reverse_edges(dgl.rand_graph(200, 1000))
    neighbor_sampler = dgl.dataloading.NeighborSampler([2, 2])
    feature_store = dgl.graphbolt.feature_store.InMemoryFeatureStore(
        {
            "feature": torch.randn(200, 4),
            "label": torch.randint(0, 10, (200,)),
        }
    )

    def fetch(item):
        input_nodes, output_nodes, subgraphs = item
        return (
            feature_store.read("feature", input_nodes),
            feature_store.read("label", output_nodes),
            subgraphs,
        )

    # sharding_filter() is for distributing minibatches to worker processes.
    # It should be later on merged into MinibatchSampler to maximize efficiency.
    minibatch_sampler = dgl.graphbolt.MinibatchSampler(
        itemset, B
    ).sharding_filter()
    subgraph_sampler = dgl.graphbolt.MockSubgraphSampler(
        minibatch_sampler,
        graph,
        neighbor_sampler,
    )

    if F.ctx() != F.cpu():
        dataloader = dgl.graphbolt.MultiprocessingDataLoader(
            subgraph_sampler,
            fetch,
            num_workers=4,
            device=F.ctx(),
            stream=torch.cuda.Stream(device=F.ctx()),
        )
    else:
        dataloader = dgl.graphbolt.MockDataLoader(
            subgraph_sampler,
            fetch,
            num_workers=4,
            device=F.ctx(),
            stream=None,
        )

    samples = list(dataloader)
    assert len(samples) == N // B


if __name__ == "__main__":
    test_DataLoader()
