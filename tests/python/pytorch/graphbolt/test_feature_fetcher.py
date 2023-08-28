import dgl.graphbolt as gb
import gb_test_utils
import torch
from torchdata.datapipes.iter import Mapper


def test_FeatureFetcher_homo():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    a = torch.randint(0, 10, (graph.num_nodes,))
    b = torch.randint(0, 10, (graph.num_edges,))

    features = {}
    keys = [("node", None, "a"), ("edge", None, "b")]
    features[keys[0]] = gb.TorchBasedFeature(a)
    features[keys[1]] = gb.TorchBasedFeature(b)
    feature_store = gb.BasicFeatureStore(features)

    itemset = gb.ItemSet(torch.arange(10))
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, gb_test_utils.to_node_block)
    sampler_dp = gb.NeighborSampler(data_block_converter, graph, fanouts)
    fetcher_dp = gb.FeatureFetcher(sampler_dp, feature_store, keys)

    assert len(list(fetcher_dp)) == 5


def test_FeatureFetcher_with_edges_homo():
    graph = gb_test_utils.rand_csc_graph(20, 0.15)
    a = torch.randint(0, 10, (graph.num_nodes,))
    b = torch.randint(0, 10, (graph.num_edges,))

    def add_node_and_edge_ids(seeds):
        subgraphs = []
        for _ in range(3):
            subgraphs.append(
                gb.SampledSubgraphImpl(
                    node_pairs=(torch.tensor([]), torch.tensor([])),
                    reverse_edge_ids=torch.randint(0, graph.num_edges, (10,)),
                )
            )
        data = gb.NodeClassificationBlock(
            input_nodes=seeds, sampled_subgraphs=subgraphs
        )
        return data

    features = {}
    keys = [("node", None, "a"), ("edge", None, "b")]
    features[keys[0]] = gb.TorchBasedFeature(a)
    features[keys[1]] = gb.TorchBasedFeature(b)
    feature_store = gb.BasicFeatureStore(features)

    itemset = gb.ItemSet(torch.arange(10))
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    converter_dp = Mapper(minibatch_dp, add_node_and_edge_ids)
    fetcher_dp = gb.FeatureFetcher(converter_dp, feature_store, keys)

    assert len(list(fetcher_dp)) == 5
    for data in fetcher_dp:
        assert data.node_feature[(None, "a")].size(0) == 2
        assert len(data.edge_feature) == 3
        for edge_feature in data.edge_feature:
            assert edge_feature[(None, "b")].size(0) == 10


def get_hetero_graph():
    # COO graph:
    # [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # [2, 4, 2, 3, 0, 1, 1, 0, 0, 1]
    # [1, 1, 1, 1, 0, 0, 0, 0, 0] - > edge type.
    # num_nodes = 5, num_n1 = 2, num_n2 = 3
    ntypes = {"n1": 0, "n2": 1}
    etypes = {("n1", "e1", "n2"): 0, ("n2", "e2", "n1"): 1}
    metadata = gb.GraphMetadata(ntypes, etypes)
    indptr = torch.LongTensor([0, 2, 4, 6, 8, 10])
    indices = torch.LongTensor([2, 4, 2, 3, 0, 1, 1, 0, 0, 1])
    type_per_edge = torch.LongTensor([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    node_type_offset = torch.LongTensor([0, 2, 5])
    return gb.from_csc(
        indptr,
        indices,
        node_type_offset=node_type_offset,
        type_per_edge=type_per_edge,
        metadata=metadata,
    )


def test_FeatureFetcher_hetero():
    graph = get_hetero_graph()
    a = torch.randint(0, 10, (2,))
    b = torch.randint(0, 10, (3,))

    features = {}
    keys = [("node", "n1", "a"), ("node", "n2", "a")]
    features[keys[0]] = gb.TorchBasedFeature(a)
    features[keys[1]] = gb.TorchBasedFeature(b)
    feature_store = gb.BasicFeatureStore(features)

    itemset = gb.ItemSetDict(
        {
            "n1": gb.ItemSet(torch.LongTensor([0, 1])),
            "n2": gb.ItemSet(torch.LongTensor([0, 1, 2])),
        }
    )
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    num_layer = 2
    fanouts = [torch.LongTensor([2]) for _ in range(num_layer)]
    data_block_converter = Mapper(minibatch_dp, gb_test_utils.to_node_block)
    sampler_dp = gb.NeighborSampler(data_block_converter, graph, fanouts)
    fetcher_dp = gb.FeatureFetcher(sampler_dp, feature_store, keys)

    assert len(list(fetcher_dp)) == 3


def test_FeatureFetcher_with_edges_hetero():
    a = torch.randint(0, 10, (20,))
    b = torch.randint(0, 10, (50,))

    def add_node_and_edge_ids(seeds):
        subgraphs = []
        reverse_edge_ids = {
            ("n1", "e1", "n2"): torch.randint(0, 50, (10,)),
            ("n2", "e2", "n1"): torch.randint(0, 50, (10,)),
        }
        for _ in range(3):
            subgraphs.append(
                gb.SampledSubgraphImpl(
                    node_pairs=(torch.tensor([]), torch.tensor([])),
                    reverse_edge_ids=reverse_edge_ids,
                )
            )
        data = gb.NodeClassificationBlock(
            input_nodes=seeds, sampled_subgraphs=subgraphs
        )
        return data

    features = {}
    keys = [("node", "n1", "a"), ("edge", "n1:e1:n2", "a")]
    features[keys[0]] = gb.TorchBasedFeature(a)
    features[keys[1]] = gb.TorchBasedFeature(b)
    feature_store = gb.BasicFeatureStore(features)

    itemset = gb.ItemSetDict(
        {
            "n1": gb.ItemSet(torch.randint(0, 20, (10,))),
        }
    )
    minibatch_dp = gb.MinibatchSampler(itemset, batch_size=2)
    converter_dp = Mapper(minibatch_dp, add_node_and_edge_ids)
    fetcher_dp = gb.FeatureFetcher(converter_dp, feature_store, keys)

    assert len(list(fetcher_dp)) == 5
    for data in fetcher_dp:
        assert data.node_feature[("n1", "a")].size(0) == 2
        assert len(data.edge_feature) == 3
        for edge_feature in data.edge_feature:
            assert edge_feature[("n1:e1:n2", "a")].size(0) == 10
