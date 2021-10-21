import dgl.data as data
import unittest
import backend as F
import numpy as np
import gzip
import tempfile
import os


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


if __name__ == '__main__':
    test_minigc()
    test_gin()
    test_data_hash()
    test_tudataset_regression()
    test_fraud()
    test_fakenews()
    test_extract_archive()
