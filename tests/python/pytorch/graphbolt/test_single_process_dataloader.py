import backend as F
import dgl
import dgl.graphbolt
import gb_test_utils
import torch
from torchdata.datapipes.iter import Mapper


def to_node_block(data):
    block = dgl.graphbolt.NodeClassificationBlock(seed_node=data)
    return block


def to_tuple(data):
    output_nodes = data.sampled_subgraphs[-1].reverse_column_node_ids
    return data.input_nodes, output_nodes, data.sampled_subgraphs


def test_DataLoader():
    N = 32
    B = 4
    itemset = dgl.graphbolt.ItemSet(torch.arange(N))
    graph = gb_test_utils.rand_csc_graph(200, 0.15)
    features = dgl.graphbolt.TorchBasedFeature(torch.randn(200, 4))
    labels = dgl.graphbolt.TorchBasedFeature(torch.randint(0, 10, (200,)))

    def fetch_func(data):
        input_nodes, output_nodes, adjs = data
        input_features = features.read(input_nodes)
        output_labels = labels.read(output_nodes)
        return input_features, output_labels, adjs

    minibatch_sampler = dgl.graphbolt.MinibatchSampler(itemset, batch_size=B)
    block_converter = Mapper(minibatch_sampler, to_node_block)
    subgraph_sampler = dgl.graphbolt.NeighborSampler(
        block_converter,
        graph,
        fanouts=[torch.LongTensor([2]) for _ in range(2)],
    )
    tuple_converter = Mapper(subgraph_sampler, to_tuple)
    feature_fetcher = dgl.graphbolt.FeatureFetcher(tuple_converter, fetch_func)
    device_transferrer = dgl.graphbolt.CopyTo(feature_fetcher, F.ctx())

    dataloader = dgl.graphbolt.SingleProcessDataLoader(device_transferrer)
    assert len(list(dataloader)) == N // B
