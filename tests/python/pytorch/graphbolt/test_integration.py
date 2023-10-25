import dgl
import dgl.graphbolt as gb
import dgl.sparse as dglsp
import torch


def test_integration_link_prediction():
    torch.manual_seed(926)

    indptr = torch.tensor([0, 0, 1, 3, 6, 8, 10])
    indices = torch.tensor([5, 3, 3, 3, 3, 4, 4, 0, 5, 4])

    matrix_a = dglsp.from_csc(indptr, indices)
    node_pairs = torch.t(torch.stack(matrix_a.coo()))
    node_feature_data = torch.tensor(
        [
            [0.9634, 0.2294],
            [0.6172, 0.7865],
            [0.2109, 0.1089],
            [0.8672, 0.2276],
            [0.5503, 0.8223],
            [0.5160, 0.2486],
        ]
    )
    edge_feature_data = torch.tensor(
        [
            [0.5123, 0.1709, 0.6150],
            [0.1476, 0.1902, 0.1314],
            [0.2582, 0.5203, 0.6228],
            [0.3708, 0.7631, 0.2683],
            [0.2126, 0.7878, 0.7225],
            [0.7885, 0.3414, 0.5485],
            [0.4088, 0.8200, 0.1851],
            [0.0056, 0.9469, 0.4432],
            [0.8972, 0.7511, 0.3617],
            [0.5773, 0.2199, 0.3366],
        ]
    )

    item_set = gb.ItemSet(node_pairs, names="node_pairs")
    graph = gb.from_csc(indptr, indices)

    node_feature = gb.TorchBasedFeature(node_feature_data)
    edge_feature = gb.TorchBasedFeature(edge_feature_data)
    features = {
        ("node", None, "feat"): node_feature,
        ("edge", None, "feat"): edge_feature,
    }
    feature_store = gb.BasicFeatureStore(features)
    datapipe = gb.ItemSampler(item_set, batch_size=4)
    datapipe = datapipe.sample_uniform_negative(graph, 1)
    fanouts = torch.LongTensor([1])
    datapipe = datapipe.sample_neighbor(graph, [fanouts, fanouts], replace=True)
    datapipe = datapipe.transform(gb.exclude_seed_edges)
    datapipe = datapipe.fetch_feature(
        feature_store, node_feature_keys=["feat"], edge_feature_keys=["feat"]
    )
    datapipe = datapipe.to_dgl()
    dataloader = gb.SingleProcessDataLoader(
        datapipe,
    )
    expected = [
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1, 1, 1]), tensor([2, 3, 3, 1])),
             output_nodes=None,
             node_features={'feat': tensor([[0.5160, 0.2486],
                                    [0.8672, 0.2276],
                                    [0.6172, 0.7865],
                                    [0.2109, 0.1089],
                                    [0.9634, 0.2294],
                                    [0.5503, 0.8223]])},
             negative_node_pairs=(tensor([0, 1, 1, 1]), tensor([0, 3, 4, 5])),
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=6,
                           num_dst_nodes=6,
                           num_edges=2),
                     Block(num_src_nodes=6,
                           num_dst_nodes=6,
                           num_edges=2)],
          )"""
        ),
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1, 1, 2]), tensor([0, 0, 1, 1])),
             output_nodes=None,
             node_features={'feat': tensor([[0.8672, 0.2276],
                                    [0.5503, 0.8223],
                                    [0.9634, 0.2294],
                                    [0.5160, 0.2486],
                                    [0.6172, 0.7865]])},
             negative_node_pairs=(tensor([0, 1, 1, 2]), tensor([1, 3, 4, 1])),
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=5,
                           num_dst_nodes=5,
                           num_edges=2),
                     Block(num_src_nodes=5,
                           num_dst_nodes=5,
                           num_edges=2)],
          )"""
        ),
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1]), tensor([0, 0])),
             output_nodes=None,
             node_features={'feat': tensor([[0.5160, 0.2486],
                                    [0.5503, 0.8223],
                                    [0.8672, 0.2276],
                                    [0.9634, 0.2294]])},
             negative_node_pairs=(tensor([0, 1]), tensor([1, 2])),
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=4,
                           num_dst_nodes=4,
                           num_edges=2),
                     Block(num_src_nodes=4,
                           num_dst_nodes=3,
                           num_edges=2)],
          )"""
        ),
    ]
    for step, data in enumerate(dataloader):
        assert expected[step] == str(data), print(data)


def test_integration_node_classification():
    torch.manual_seed(926)

    indptr = torch.tensor([0, 0, 1, 3, 6, 8, 10])
    indices = torch.tensor([5, 3, 3, 3, 3, 4, 4, 0, 5, 4])

    matrix_a = dglsp.from_csc(indptr, indices)
    node_pairs = torch.t(torch.stack(matrix_a.coo()))
    node_feature_data = torch.tensor(
        [
            [0.9634, 0.2294],
            [0.6172, 0.7865],
            [0.2109, 0.1089],
            [0.8672, 0.2276],
            [0.5503, 0.8223],
            [0.5160, 0.2486],
        ]
    )
    edge_feature_data = torch.tensor(
        [
            [0.5123, 0.1709, 0.6150],
            [0.1476, 0.1902, 0.1314],
            [0.2582, 0.5203, 0.6228],
            [0.3708, 0.7631, 0.2683],
            [0.2126, 0.7878, 0.7225],
            [0.7885, 0.3414, 0.5485],
            [0.4088, 0.8200, 0.1851],
            [0.0056, 0.9469, 0.4432],
            [0.8972, 0.7511, 0.3617],
            [0.5773, 0.2199, 0.3366],
        ]
    )

    item_set = gb.ItemSet(node_pairs, names="node_pairs")
    graph = gb.from_csc(indptr, indices)

    node_feature = gb.TorchBasedFeature(node_feature_data)
    edge_feature = gb.TorchBasedFeature(edge_feature_data)
    features = {
        ("node", None, "feat"): node_feature,
        ("edge", None, "feat"): edge_feature,
    }
    feature_store = gb.BasicFeatureStore(features)
    datapipe = gb.ItemSampler(item_set, batch_size=4)
    fanouts = torch.LongTensor([1])
    datapipe = datapipe.sample_neighbor(graph, [fanouts, fanouts], replace=True)
    datapipe = datapipe.fetch_feature(
        feature_store, node_feature_keys=["feat"], edge_feature_keys=["feat"]
    )
    datapipe = datapipe.to_dgl()
    dataloader = gb.SingleProcessDataLoader(
        datapipe,
    )
    expected = [
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1, 1, 1]), tensor([2, 3, 3, 1])),
             output_nodes=None,
             node_features={'feat': tensor([[0.5160, 0.2486],
                                    [0.8672, 0.2276],
                                    [0.6172, 0.7865],
                                    [0.2109, 0.1089],
                                    [0.5503, 0.8223],
                                    [0.9634, 0.2294]])},
             negative_node_pairs=None,
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=6,
                           num_dst_nodes=5,
                           num_edges=5),
                     Block(num_src_nodes=5,
                           num_dst_nodes=4,
                           num_edges=4)],
          )"""
        ),
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1, 1, 2]), tensor([0, 0, 1, 1])),
             output_nodes=None,
             node_features={'feat': tensor([[0.8672, 0.2276],
                                    [0.5503, 0.8223],
                                    [0.9634, 0.2294]])},
             negative_node_pairs=None,
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=3,
                           num_dst_nodes=3,
                           num_edges=2),
                     Block(num_src_nodes=3,
                           num_dst_nodes=3,
                           num_edges=2)],
          )"""
        ),
        str(
            """DGLMiniBatch(positive_node_pairs=(tensor([0, 1]), tensor([0, 0])),
             output_nodes=None,
             node_features={'feat': tensor([[0.5160, 0.2486],
                                    [0.5503, 0.8223],
                                    [0.9634, 0.2294]])},
             negative_node_pairs=None,
             labels=None,
             input_nodes=None,
             edge_features=[{},
                            {}],
             blocks=[Block(num_src_nodes=3,
                           num_dst_nodes=2,
                           num_edges=2),
                     Block(num_src_nodes=2,
                           num_dst_nodes=2,
                           num_edges=2)],
          )"""
        ),
    ]
    for step, data in enumerate(dataloader):
        assert expected[step] == str(data), print(data)
