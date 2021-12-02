import dgl.data as data
import unittest
import backend as F
import numpy as np
import gzip
import tempfile
import os
import pandas as pd
import yaml


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_minigc():
    ds = data.MiniGCDataset(16, 10, 20)
    g, l = list(zip(*ds))
    print(g, l)


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_gin():
    ds_n_graphs = {
        'MUTAG': 188,
        'IMDBBINARY': 1000,
        'IMDBMULTI': 1500,
        'PROTEINS': 1113,
        'PTC': 344,
    }
    for name, n_graphs in ds_n_graphs.items():
        ds = data.GINDataset(name, self_loop=False, degree_as_nlabel=False)
        assert len(ds) == n_graphs, (len(ds), name)


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_fraud():
    g = data.FraudDataset('amazon')[0]
    assert g.num_nodes() == 11944

    g = data.FraudAmazonDataset()[0]
    assert g.num_nodes() == 11944

    g = data.FraudYelpDataset()[0]
    assert g.num_nodes() == 45954


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_fakenews():
    ds = data.FakeNewsDataset('politifact', 'bert')
    assert len(ds) == 314

    ds = data.FakeNewsDataset('gossipcop', 'profile')
    assert len(ds) == 5464


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_tudataset_regression():    
    ds = data.TUDataset('ZINC_test', force_reload=True)
    assert len(ds) == 5000


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_data_hash():
    class HashTestDataset(data.DGLDataset):
        def __init__(self, hash_key=()):
            super(HashTestDataset, self).__init__('hashtest', hash_key=hash_key)
        def _load(self):
            pass

    a = HashTestDataset((True, 0, '1', (1,2,3)))
    b = HashTestDataset((True, 0, '1', (1,2,3)))
    c = HashTestDataset((True, 0, '1', (1,2,4)))
    assert a.hash == b.hash
    assert a.hash != c.hash


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_citation_graph():
    # cora
    g = data.CoraGraphDataset()[0]
    assert g.num_nodes() == 2708
    assert g.num_edges() == 10556
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # Citeseer
    g = data.CiteseerGraphDataset()[0]
    assert g.num_nodes() == 3327
    assert g.num_edges() == 9228
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # Pubmed
    g = data.PubmedGraphDataset()[0]
    assert g.num_nodes() == 19717
    assert g.num_edges() == 88651
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_gnn_benchmark():
    # AmazonCoBuyComputerDataset
    g = data.AmazonCoBuyComputerDataset()[0]
    assert g.num_nodes() == 13752
    assert g.num_edges() == 491722
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # AmazonCoBuyPhotoDataset
    g = data.AmazonCoBuyPhotoDataset()[0]
    assert g.num_nodes() == 7650
    assert g.num_edges() == 238163
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # CoauthorPhysicsDataset
    g = data.CoauthorPhysicsDataset()[0]
    assert g.num_nodes() == 34493
    assert g.num_edges() == 495924
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # CoauthorCSDataset
    g = data.CoauthorCSDataset()[0]
    assert g.num_nodes() == 18333
    assert g.num_edges() == 163788
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))

    # CoraFullDataset
    g = data.CoraFullDataset()[0]
    assert g.num_nodes() == 19793
    assert g.num_edges() == 126842
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_reddit():
    # RedditDataset
    g = data.RedditDataset()[0]
    assert g.num_nodes() == 232965
    assert g.num_edges() == 114615892
    dst = F.asnumpy(g.edges()[1])
    assert np.array_equal(dst, np.sort(dst))


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_extract_archive():
    # gzip
    with tempfile.TemporaryDirectory() as src_dir:
        gz_file = 'gz_archive'
        gz_path = os.path.join(src_dir, gz_file + '.gz')
        content = b"test extract archive gzip"
        with gzip.open(gz_path, 'wb') as f:
            f.write(content)
        with tempfile.TemporaryDirectory() as dst_dir:
            data.utils.extract_archive(gz_path, dst_dir, overwrite=True)
            assert os.path.exists(os.path.join(dst_dir, gz_file))


class GenerateFilesForCSVDatasetSingle:
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.meta_yaml = os.path.join(test_dir, "test_csv_meta.yml")
        self.csv_files = []

    def __enter__(self):
        # single graph with multiple edges.csv and nodes.csv
        meta_yaml_data = {'version': '0.0.1', 'dataset_type': 'user_defined',
                          'edges': [{'file_name': 'test_edges_0.csv',
                                     'etype': ['user', 'like', 'item'],
                                     },
                                    {'file_name': 'test_edges_1.csv',
                                     'etype': ['user', 'follow', 'user'],
                                     'labels': {'type': 'classification', 'num_classes': 3},
                                     }],
                          'nodes': [{'file_name': 'test_nodes_0.csv', 'ntype': 'user',
                                     },
                                    {'file_name': 'test_nodes_1.csv', 'ntype': 'item',
                                     'labels': {'type': 'regression'},
                                     }],
                          }

        meta_yaml = self.meta_yaml
        with open(meta_yaml, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        for i in range(2):
            edge_csv = os.path.join(
                self.test_dir, "test_edges_{}.csv".format(i))
            df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                               'dst': np.random.randint(num_nodes, size=num_edges),
                               'label': np.random.randint(2, size=num_edges),
                               'split_type': np.random.randint(3, size=num_edges),
                               'feat_0': np.random.rand(num_edges),
                               'feat_1': np.random.rand(num_edges)})
            df.to_csv(edge_csv)
            self.csv_files.append(edge_csv)
            node_csv = os.path.join(
                self.test_dir, "test_nodes_{}.csv".format(i))
            df = pd.DataFrame({'id': np.arange(num_nodes),
                               'label': np.random.randint(2, size=num_nodes),
                               'split_type': np.random.randint(3, size=num_nodes),
                               'feat_0': np.random.rand(num_nodes),
                               'feat_1': np.random.rand(num_nodes),
                               'feat_2': np.random.rand(num_nodes)})
            df.to_csv(node_csv)
            self.csv_files.append(node_csv)
        return self

    def __exit__(self, *args, **kwargs):
        os.remove(self.meta_yaml)
        [os.remove(csv_file) for csv_file in self.csv_files]


class GenerateFilesForCSVDatasetMultiple():
    def __init__(self, test_dir):
        self.meta_yaml = os.path.join(test_dir, "test_csv_meta.yml")
        self.edge_csv = os.path.join(test_dir, "test_edges.csv")
        self.node_csv = os.path.join(test_dir, "test_nodes.csv")
        self.graph_csv = os.path.join(test_dir, "test_graphs.csv")

    def __enter__(self):
        # multiple graphs with single edges.csv and nodes.csv
        meta_yaml_data = {'version': '0.0.1', 'dataset_type': 'user_defined', 'dataset_name': 'multiple_graphs',
                          'edges': [{'file_name': 'test_edges.csv', 'separator': ',',
                                     'graph_id_field': 'graph_id', 'src_id_field': 'src', 'dst_id_field': 'dst',
                                    'feat_prefix_field': 'feat_'}],
                          'nodes': [{'file_name': 'test_nodes.csv', 'separator': ',', 'graph_id_field': 'graph_id',
                                     'node_id_field': 'id', 'feat_prefix_field': 'feat_'}],
                          'graphs': {'file_name': 'test_graphs.csv', 'separator': ',',
                                     'graph_id_field': 'graph_id',
                                     'labels': {'type': 'classification', 'field': 'label', 'num_classes': 2},
                                     'split_type_field': 'split_type'},
                          }

        meta_yaml = self.meta_yaml
        with open(meta_yaml, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        num_graphs = 20
        df = pd.DataFrame({'graph_id': np.random.randint(num_graphs, size=num_edges*num_graphs),
                           'src': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           'dst': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           })
        df.to_csv(self.edge_csv)
        graph_id = []
        id = []
        for i in range(num_graphs):
            graph_id.extend(np.full(num_nodes, i))
            id.extend(np.arange(num_nodes))
        df = pd.DataFrame({'graph_id': graph_id,
                           'id': id,
                           'feat_0': np.random.rand(num_nodes*num_graphs),
                           'feat_1': np.random.rand(num_nodes*num_graphs),
                           'feat_2': np.random.rand(num_nodes*num_graphs)})
        df.to_csv(self.node_csv)
        df = pd.DataFrame({'graph_id': np.arange(num_graphs),
                           'label': np.random.randint(2, size=num_graphs),
                           'split_type': np.random.randint(3, size=num_graphs)})
        df.to_csv(self.graph_csv)
        return self

    def __exit__(self, *args, **kwargs):
        os.remove(self.meta_yaml)
        os.remove(self.edge_csv)
        os.remove(self.node_csv)
        os.remove(self.graph_csv)


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
def test_csv_dataset():
    with tempfile.TemporaryDirectory() as test_dir:
        for force_reload in [True, False]:
            # single graph with multiple edges.csv and nodes.csv
            with GenerateFilesForCSVDatasetSingle(test_dir) as f:
                csv_dataset = data.CSVDataset(
                    f.meta_yaml, force_reload=force_reload)
                assert len(csv_dataset) == 1
                graph = csv_dataset[0]
                assert ~graph.is_homogeneous
                assert csv_dataset.has_cache()

            # multiple graphs with single edges.csv and nodes.csv
            with GenerateFilesForCSVDatasetMultiple(test_dir) as f:
                csv_dataset = data.CSVDataset(
                    f.meta_yaml, force_reload=force_reload)
                assert len(csv_dataset) > 1
                graph, label = csv_dataset[0]
                assert graph.is_homogeneous
                assert 'train_mask' in csv_dataset.mask
                assert 'val_mask' in csv_dataset.mask
                assert 'test_mask' in csv_dataset.mask
                assert len(csv_dataset) == csv_dataset.mask['train_mask'].shape[0]
                assert csv_dataset.has_cache()


if __name__ == '__main__':
    test_minigc()
    test_gin()
    test_data_hash()
    test_tudataset_regression()
    test_fraud()
    test_fakenews()
    test_extract_archive()
    test_csv_dataset()
