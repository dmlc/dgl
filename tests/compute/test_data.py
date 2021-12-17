import dgl.data as data
import unittest
import backend as F
import numpy as np
import gzip
import tempfile
import os
import pandas as pd
import yaml
import pytest


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
            super(HashTestDataset, self).__init__(
                'hashtest', hash_key=hash_key)

        def _load(self):
            pass

    a = HashTestDataset((True, 0, '1', (1, 2, 3)))
    b = HashTestDataset((True, 0, '1', (1, 2, 3)))
    c = HashTestDataset((True, 0, '1', (1, 2, 4)))
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
        self.data_path = test_dir
        self.meta_yaml = os.path.join(test_dir, "meta.yaml")
        self.csv_files = []

    def __enter__(self):
        # single graph with multiple edges.csv and nodes.csv
        meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'single_graph',
                          'edge_data': [{'file_name': 'test_edges_0.csv',
                                         'etype': ['user', 'like', 'item'],
                                         'feats':{}
                                         },
                                        {'file_name': 'test_edges_1.csv',
                                         'etype': ['user', 'follow', 'user'],
                                         'feats':{},
                                         'labels': {'type': 'classification', 'num_classes': 3},
                                         }],
                          'node_data': [{'file_name': 'test_nodes_0.csv', 'ntype': 'user', 'feats': {}
                                         },
                                        {'file_name': 'test_nodes_1.csv', 'ntype': 'item', 'feats': {},
                                         'labels': {'type': 'regression'},
                                         }],
                          }

        meta_yaml = self.meta_yaml
        with open(meta_yaml, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        feat_dim = 3
        for i in range(2):
            edge_csv = os.path.join(
                self.test_dir, "test_edges_{}.csv".format(i))
            df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                               'dst': np.random.randint(num_nodes, size=num_edges),
                               'label': np.random.randint(2, size=num_edges),
                               'split_type': np.random.randint(3, size=num_edges),
                               'feat': np.array([",".join(item) for item in np.random.rand(num_edges, feat_dim).astype(str)])})
            df.to_csv(edge_csv, index=False)
            self.csv_files.append(edge_csv)
            node_csv = os.path.join(
                self.test_dir, "test_nodes_{}.csv".format(i))
            df = pd.DataFrame({'id': np.arange(num_nodes),
                               'label': np.random.randint(2, size=num_nodes),
                               'split_type': np.random.randint(3, size=num_nodes),
                               'feat': np.array([",".join(item) for item in np.random.rand(num_nodes, feat_dim).astype(str)])})
            df.to_csv(node_csv, index=False)
            self.csv_files.append(node_csv)
        return self

    def __exit__(self, *args, **kwargs):
        os.remove(self.meta_yaml)
        [os.remove(csv_file) for csv_file in self.csv_files]


class GenerateFilesForCSVDatasetMultiple():
    def __init__(self, test_dir):
        self.data_path = test_dir
        self.meta_yaml = os.path.join(test_dir, "meta.yaml")
        self.edge_csv = os.path.join(test_dir, "test_edges.csv")
        self.node_csv = os.path.join(test_dir, "test_nodes.csv")
        self.graph_csv = os.path.join(test_dir, "test_graphs.csv")

    def __enter__(self):
        # multiple graphs with single edges.csv and nodes.csv
        meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'multiple_graphs',
                          'edge_data': [{'file_name': 'test_edges.csv', 'separator': ',',
                                         'graph_id_field': 'graph_id', 'src_id_field': 'src', 'dst_id_field': 'dst',
                                         }],
                          'node_data': [{'file_name': 'test_nodes.csv', 'separator': ',', 'graph_id_field': 'graph_id',
                                         'node_id_field': 'id', 'feats': {'field': 'feat', 'separator': ','}}],
                          'graph_data': {'file_name': 'test_graphs.csv', 'separator': ',',
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
        feat_dim = 3
        df = pd.DataFrame({'graph_id': np.random.randint(num_graphs, size=num_edges*num_graphs),
                           'src': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           'dst': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           })
        df.to_csv(self.edge_csv, index=False)
        graph_id = []
        id = []
        for i in range(num_graphs):
            graph_id.extend(np.full(num_nodes, i))
            id.extend(np.arange(num_nodes))
        df = pd.DataFrame({'graph_id': graph_id,
                           'id': id,
                           'feat': np.array([",".join(item) for item in np.random.rand(num_nodes*num_graphs, feat_dim).astype(str)])})
        df.to_csv(self.node_csv, index=False)
        df = pd.DataFrame({'graph_id': np.arange(num_graphs),
                           'label': np.random.randint(2, size=num_graphs),
                           'split_type': np.random.randint(3, size=num_graphs)})
        df.to_csv(self.graph_csv, index=False)
        return self

    def __exit__(self, *args, **kwargs):
        os.remove(self.meta_yaml)
        os.remove(self.edge_csv)
        os.remove(self.node_csv)
        os.remove(self.graph_csv)


def _test_csvdt_simplex_homo(simplex_yaml):
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path = os.path.join(test_dir, "test_edges.csv")
        nodes_csv_path = os.path.join(test_dir, "test_nodes.csv")
        if simplex_yaml:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path),
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path),
                                             }],
                              }
        else:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path),
                                            'separator': ',', 'src_id_field': 'src', 'dst_id_field': 'dst',

                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path),
                                             'separator': ',', 'node_id_field': 'id',
                                             }],
                              }

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 300
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           })
        df.to_csv(edges_csv_path, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           })
        df.to_csv(nodes_csv_path, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == 1
        g = csv_dataset[0]
        assert g.is_homogeneous
        assert g.num_nodes() == num_nodes
        assert g.num_edges() == num_edges

        def _check(gdata):
            assert 'feat' not in gdata
            assert 'label' not in gdata
            assert 'train_mask' not in gdata
            assert 'val_mask' not in gdata
            assert 'test_mask' not in gdata
            assert 'num_classes' not in gdata
        _check(g.ndata)
        _check(g.edata)
        assert len(g.ndata) == 0
        assert len(g.edata) == 0


def _test_csvdt_complex_homo(simplex_yaml):
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path = os.path.join(test_dir, "test_edges.csv")
        nodes_csv_path = os.path.join(test_dir, "test_nodes.csv")
        if simplex_yaml:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path),
                                             'labels': {'type': 'regression'},
                                             'feats': {'separator': '|'}
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path),
                                             'labels': {'type': 'classification', 'num_classes': 3},
                                             'feats': {'separator': '|'}
                                             }],
                              }
        else:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path),
                                             'labels': {'type': 'regression', 'field': 'label'},
                                             'separator': ',',
                                             'src_id_field': 'src', 'dst_id_field': 'dst',
                                             'feats': {'field': 'feat', 'separator': '|'},
                                             'split_type_field': 'split_type',
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path),
                                             'labels': {'type': 'classification', 'num_classes': 3, 'field': 'label'},
                                             'separator': ',',
                                             'node_id_field': 'id',
                                             'feats': {'field': 'feat', 'separator': '|'}, 'split_type_field': 'split_type',
                                             }],
                              }

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        feat_dim = 3
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           'label': np.random.randint(2, size=num_edges),
                           'split_type': np.random.randint(3, size=num_edges),
                           'feat': np.array(["|".join(item) for item in np.random.rand(num_edges, feat_dim).astype(str)])
                           })
        df.to_csv(edges_csv_path, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           'label': np.random.randint(2, size=num_nodes),
                           'split_type': np.random.randint(3, size=num_nodes),
                           'feat': np.array(["|".join(item) for item in np.random.rand(num_nodes, feat_dim).astype(str)])
                           })
        df.to_csv(nodes_csv_path, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == 1
        g = csv_dataset[0]
        assert g.is_homogeneous
        assert g.num_nodes() == num_nodes
        assert g.num_edges() == num_edges

        def _check(gdata):
            assert 'feat' in gdata
            assert 'label' in gdata
            assert 'train_mask' in gdata
            assert 'val_mask' in gdata
            assert 'test_mask' in gdata
        _check(g.ndata)
        _check(g.edata)
        assert 'num_classes' in g.ndata
        assert 'num_classes' not in g.edata


def _test_csvdt_simplex_hetero(simplex_yaml):
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path_0 = os.path.join(test_dir, "test_edges_0.csv")
        edges_csv_path_1 = os.path.join(test_dir, "test_edges_1.csv")
        nodes_csv_path_0 = os.path.join(test_dir, "test_nodes_0.csv")
        nodes_csv_path_1 = os.path.join(test_dir, "test_nodes_1.csv")
        if simplex_yaml:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path_0),
                                             'etype': ['user', 'follow', 'user'],
                                             },
                                            {'file_name': os.path.basename(edges_csv_path_1),
                                             'etype': ['user', 'like', 'item'],
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path_0),
                                             'ntype': 'user',
                                             },
                                            {'file_name': os.path.basename(nodes_csv_path_1),
                                             'ntype': 'item',
                                             }],
                              }
        else:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path_0),
                                             'etype': ['user', 'follow', 'user'],
                                            'separator':',', 'src_id_field':'src', 'dst_id_field':'dst',

                                             },
                                            {'file_name': os.path.basename(edges_csv_path_1),
                                             'etype': ['user', 'like', 'item'],
                                            'separator':',', 'src_id_field':'src', 'dst_id_field':'dst',

                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path_0),
                                             'ntype': 'user',
                                             'separator': ',', 'node_id_field': 'id',
                                             },
                                            {'file_name': os.path.basename(nodes_csv_path_1),
                                             'ntype': 'item',
                                             'separator': ',', 'node_id_field': 'id',
                                             }],
                              }

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           })
        df.to_csv(edges_csv_path_0, index=False)
        df.to_csv(edges_csv_path_1, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           })
        df.to_csv(nodes_csv_path_0, index=False)
        df.to_csv(nodes_csv_path_1, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == 1
        g = csv_dataset[0]
        assert ~g.is_homogeneous

        def _check(gdata):
            assert 'feat' not in gdata
            assert 'label' not in gdata
            assert 'train_mask' not in gdata
            assert 'val_mask' not in gdata
            assert 'test_mask' not in gdata
            assert 'num_classes' not in gdata
        for ntype in g.ntypes:
            assert g.num_nodes(ntype) == num_nodes
            _check(g.nodes[ntype].data)
        for etype in g.etypes:
            assert g.num_edges(etype) == num_edges
            _check(g.edges[etype].data)


def _test_csvdt_complex_hetero(simplex_yaml):
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path_0 = os.path.join(test_dir, "test_edges_0.csv")
        edges_csv_path_1 = os.path.join(test_dir, "test_edges_1.csv")
        nodes_csv_path_0 = os.path.join(test_dir, "test_nodes_0.csv")
        nodes_csv_path_1 = os.path.join(test_dir, "test_nodes_1.csv")
        if simplex_yaml:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path_0),
                                             'etype': ['user', 'follow', 'user'],
                                             'labels': {'type': 'regression'},
                                             'feats': {}
                                             },
                                            {'file_name': os.path.basename(edges_csv_path_1),
                                             'etype': ['user', 'like', 'item'],
                                             'labels': {'type': 'regression'},
                                             'feats': {}
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path_0),
                                             'ntype': 'user',
                                             'labels': {'type': 'classification', 'num_classes': 3},
                                             'feats': {}
                                             },
                                            {'file_name': os.path.basename(nodes_csv_path_1),
                                             'ntype': 'item',
                                             'labels': {'type': 'classification', 'num_classes': 3},
                                             'feats': {}
                                             }],
                              }
        else:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path_0),
                                             'etype': ['user', 'follow', 'user'],
                                             'labels': {'type': 'regression', 'field': 'label'},
                                             'separator': ',',
                                             'src_id_field': 'src', 'dst_id_field': 'dst',
                                             'feats': {'field': 'feat', 'separator': ','},
                                             'split_type_field': 'split_type',
                                             },
                                            {'file_name': os.path.basename(edges_csv_path_1),
                                             'etype': ['user', 'like', 'item'],
                                             'labels': {'type': 'regression', 'field': 'label'},
                                             'separator': ',',
                                             'src_id_field': 'src', 'dst_id_field': 'dst',
                                             'feats': {'field': 'feat', 'separator': ','},
                                             'split_type_field': 'split_type',
                                             }],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path_0),
                                             'ntype': 'user',
                                             'labels': {'type': 'classification', 'num_classes': 3, 'field': 'label'},
                                             'separator': ',',
                                             'node_id_field': 'id',
                                             'feats': {'field': 'feat', 'separator': ','}, 'split_type_field': 'split_type',
                                             },
                                            {'file_name': os.path.basename(nodes_csv_path_1),
                                             'ntype': 'item',
                                             'labels': {'type': 'classification', 'num_classes': 3, 'field': 'label'},
                                             'separator': ',',
                                             'node_id_field': 'id',
                                             'feats': {'field': 'feat', 'separator': ','}, 'split_type_field': 'split_type',
                                             }],
                              }

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        feat_dim = 3
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           'label': np.random.randint(2, size=num_edges),
                           'split_type': np.random.randint(3, size=num_edges),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_edges, feat_dim).astype(str)])
                           })
        df.to_csv(edges_csv_path_0, index=False)
        df.to_csv(edges_csv_path_1, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           'label': np.random.randint(2, size=num_nodes),
                           'split_type': np.random.randint(3, size=num_nodes),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_nodes, feat_dim).astype(str)])
                           })
        df.to_csv(nodes_csv_path_0, index=False)
        df.to_csv(nodes_csv_path_1, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == 1
        g = csv_dataset[0]
        assert ~g.is_homogeneous

        def _check(gdata):
            assert 'feat' in gdata
            assert 'label' in gdata
            assert 'train_mask' in gdata
            assert 'val_mask' in gdata
            assert 'test_mask' in gdata
        for ntype in g.ntypes:
            assert g.num_nodes(ntype) == num_nodes
            _check(g.nodes[ntype].data)
            assert 'num_classes' in g.nodes[ntype].data
        for etype in g.etypes:
            assert g.num_edges(etype) == num_edges
            _check(g.edges[etype].data)
            assert 'num_classes' not in g.edges[etype].data


def _test_csvdt_multiple_graphs(simplex_yaml):
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path = os.path.join(test_dir, "test_edges.csv")
        nodes_csv_path = os.path.join(test_dir, "test_nodes.csv")
        graphs_csv_path = os.path.join(test_dir, "test_graphs.csv")
        num_classes = 3
        if simplex_yaml:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'multiple_graphs',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path), 'feats': {}
                                             },
                                            ],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path), 'feats': {}
                                             },
                                            ],
                              'graph_data': {'file_name': os.path.basename(graphs_csv_path), 'feats': {},
                                             'labels': {'type': 'classification', 'num_classes': num_classes},
                                             }}
        else:
            meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'multiple_graphs',
                              'edge_data': [{'file_name': os.path.basename(edges_csv_path), 'separator': ',',
                                             'graph_id_field': 'graph_id', 'src_id_field': 'src', 'dst_id_field': 'dst',
                                             'feats': {'field': 'feat', 'separator': ','},
                                             },
                                            ],
                              'node_data': [{'file_name': os.path.basename(nodes_csv_path), 'separator': ',', 'graph_id_field': 'graph_id',
                                             'node_id_field': 'id', 'feats': {'field': 'feat', 'separator': ','}
                                             },
                                            ],
                              'graph_data': {'file_name': os.path.basename(graphs_csv_path), 'separator': ',',
                                             'graph_id_field': 'graph_id', 'feats': {'field': 'feat', 'separator': ','},
                                             'labels': {'type': 'classification', 'field': 'label', 'num_classes': num_classes},
                                             'split_type_field': 'split_type'}}

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        num_graphs = 20
        feat_dim = 3
        graph_id = []
        for i in range(num_graphs):
            graph_id.extend(np.full(num_edges, i))
        df = pd.DataFrame({'graph_id': graph_id,
                           'src': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           'dst': np.random.randint(num_nodes, size=num_edges*num_graphs),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_edges*num_graphs, feat_dim).astype(str)])
                           })
        df.to_csv(edges_csv_path, index=False)
        graph_id = []
        id = []
        for i in range(num_graphs):
            graph_id.extend(np.full(num_nodes, i))
            id.extend(np.arange(num_nodes))
        df = pd.DataFrame({'graph_id': graph_id,
                           'id': id,
                           'feat': np.array([",".join(item) for item in np.random.rand(num_nodes*num_graphs, feat_dim).astype(str)])})
        df.to_csv(nodes_csv_path, index=False)
        df = pd.DataFrame({'graph_id': np.arange(num_graphs),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_graphs, feat_dim).astype(str)]),
                           'label': np.random.randint(num_classes, size=num_graphs),
                           'split_type': np.random.randint(3, size=num_graphs)})
        df.to_csv(graphs_csv_path, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == num_graphs
        for g, label, feat in csv_dataset:
            assert label is not None
            assert feat is not None
            assert g.is_homogeneous
            assert g.num_nodes() == num_nodes
            assert 'feat' in g.ndata
            assert g.num_edges() == num_edges
            assert 'feat' in g.edata
        assert csv_dataset.train_mask is not None
        assert csv_dataset.val_mask is not None
        assert csv_dataset.test_mask is not None
        assert len(csv_dataset) == csv_dataset.train_mask.shape[0]


def _test_csvdt_mg_empty():
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path = os.path.join(test_dir, "test_edges.csv")
        nodes_csv_path = os.path.join(test_dir, "test_nodes.csv")
        graphs_csv_path = os.path.join(test_dir, "test_graphs.csv")
        num_classes = 3
        meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'multiple_graphs',
                          'edge_data': [{'file_name': os.path.basename(edges_csv_path), 'feats': {}
                                         },
                                        ],
                          'node_data': [{'file_name': os.path.basename(nodes_csv_path), 'feats': {}
                                         },
                                        ],
                          'graph_data': {'file_name': os.path.basename(graphs_csv_path), 'feats': {},
                                         'labels': {'type': 'classification', 'num_classes': num_classes},
                                         }}

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        num_graphs = 20
        feat_dim = 3
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_edges, feat_dim).astype(str)])
                           })
        df.to_csv(edges_csv_path, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_nodes, feat_dim).astype(str)])})
        df.to_csv(nodes_csv_path, index=False)
        df = pd.DataFrame({'graph_id': np.arange(num_graphs),
                           'feat': np.array([",".join(item) for item in np.random.rand(num_graphs, feat_dim).astype(str)]),
                           'label': np.random.randint(num_classes, size=num_graphs),
                           'split_type': np.random.randint(3, size=num_graphs)})
        df.to_csv(graphs_csv_path, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == num_graphs
        for i, (g, label, feat) in enumerate(csv_dataset):
            assert g.is_homogeneous
            if i == 0:
                assert g.num_nodes() == num_nodes
                assert g.num_edges() == num_edges
                assert 'feat' in g.ndata
                assert 'feat' in g.edata
            else:
                assert g.num_nodes() == 0
                assert g.num_edges() == 0
        assert csv_dataset.train_mask is not None
        assert csv_dataset.val_mask is not None
        assert csv_dataset.test_mask is not None
        assert len(csv_dataset) == csv_dataset.train_mask.shape[0]


def _test_csvdt_homo_graph_feat():
    with tempfile.TemporaryDirectory() as test_dir:
        # generate YAML/CSVs
        meta_yaml_path = os.path.join(test_dir, "meta.yaml")
        edges_csv_path = os.path.join(test_dir, "test_edges.csv")
        nodes_csv_path = os.path.join(test_dir, "test_nodes.csv")
        graphs_csv_path = os.path.join(test_dir, "test_graphs.csv")
        meta_yaml_data = {'version': '1.0.0', 'dataset_name': 'default_name',
                          'edge_data': [{'file_name': os.path.basename(edges_csv_path),
                                         'labels': {'type': 'regression'},
                                         'feats': {'separator': '|'}
                                         }],
                          'node_data': [{'file_name': os.path.basename(nodes_csv_path),
                                         'labels': {'type': 'classification', 'num_classes': 3},
                                         'feats': {'separator': '|'}
                                         }],
                          'graph_data': {'file_name': os.path.basename(graphs_csv_path), 'feats': {'separator': '|'}}
                          }

        with open(meta_yaml_path, 'w') as f:
            yaml.dump(meta_yaml_data, f, sort_keys=False)
        num_nodes = 100
        num_edges = 500
        num_graphs = 1
        feat_dim = 3
        df = pd.DataFrame({'src': np.random.randint(num_nodes, size=num_edges),
                           'dst': np.random.randint(num_nodes, size=num_edges),
                           'label': np.random.randint(2, size=num_edges),
                           'split_type': np.random.randint(3, size=num_edges),
                           'feat': np.array(["|".join(item) for item in np.random.rand(num_edges, feat_dim).astype(str)])
                           })
        df.to_csv(edges_csv_path, index=False)
        df = pd.DataFrame({'id': np.arange(num_nodes),
                           'label': np.random.randint(2, size=num_nodes),
                           'split_type': np.random.randint(3, size=num_nodes),
                           'feat': np.array(["|".join(item) for item in np.random.rand(num_nodes, feat_dim).astype(str)])
                           })
        df.to_csv(nodes_csv_path, index=False)
        df = pd.DataFrame({'feat': np.array(
            ["|".join(item) for item in np.random.rand(num_graphs, feat_dim).astype(str)])})
        df.to_csv(graphs_csv_path, index=False)

        # load CSVDataset
        csv_dataset = data.CSVDataset(test_dir)
        assert len(csv_dataset) == 1
        g, feat = csv_dataset[0]
        assert g.is_homogeneous
        assert g.num_nodes() == num_nodes
        assert g.num_edges() == num_edges
        assert feat is not None

        def _check(gdata):
            assert 'feat' in gdata
            assert 'label' in gdata
            assert 'train_mask' in gdata
            assert 'val_mask' in gdata
            assert 'test_mask' in gdata
        _check(g.ndata)
        _check(g.edata)
        assert 'num_classes' in g.ndata
        assert 'num_classes' not in g.edata


@unittest.skipIf(F._default_context_str == 'gpu', reason="Datasets don't need to be tested on GPU.")
@pytest.mark.parametrize('simplex_yaml', [True, False])
def test_csv_dataset(simplex_yaml):
    # test various graphs with simplex or complex YAML/CSV
    _test_csvdt_simplex_homo(simplex_yaml)
    _test_csvdt_complex_homo(simplex_yaml)
    _test_csvdt_simplex_hetero(simplex_yaml)
    _test_csvdt_complex_hetero(simplex_yaml)
    _test_csvdt_multiple_graphs(simplex_yaml)

    # test more
    _test_csvdt_mg_empty()
    _test_csvdt_homo_graph_feat()

    # test dataset reload logic
    with tempfile.TemporaryDirectory() as test_dir:
        for force_reload in [True, False]:
            # single graph with multiple edges.csv and nodes.csv
            with GenerateFilesForCSVDatasetSingle(test_dir) as f:
                csv_dataset = data.CSVDataset(
                    f.data_path, force_reload=force_reload)
                assert len(csv_dataset) == 1
                graph = csv_dataset[0]
                assert ~graph.is_homogeneous
                assert csv_dataset.has_cache()

            # multiple graphs with single edges.csv and nodes.csv
            with GenerateFilesForCSVDatasetMultiple(test_dir) as f:
                csv_dataset = data.CSVDataset(
                    f.data_path, force_reload=force_reload)
                assert len(csv_dataset) > 1
                graph, label = csv_dataset[0]
                assert graph.is_homogeneous
                assert csv_dataset.train_mask is not None
                assert csv_dataset.val_mask is not None
                assert csv_dataset.test_mask is not None
                assert len(csv_dataset) == csv_dataset.train_mask.shape[0]
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
