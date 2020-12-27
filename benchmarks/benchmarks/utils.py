import os
import shutil, zipfile
import requests
import numpy as np
import pandas
import dgl
import torch

def _download(url, path, filename):
    fn = os.path.join(path, filename)
    if os.path.exists(fn):
        return

    os.makedirs(path, exist_ok=True)
    f_remote = requests.get(url, stream=True)
    sz = f_remote.headers.get('content-length')
    assert f_remote.status_code == 200, 'fail to open {}'.format(url)
    with open(fn, 'wb') as writer:
        for chunk in f_remote.iter_content(chunk_size=1024*1024):
            writer.write(chunk)
    print('Download finished.')

def get_livejournal():
    _download('https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',
              '/tmp', 'soc-LiveJournal1.txt.gz')
    df = pandas.read_csv('/tmp/soc-LiveJournal1.txt.gz', sep='\t', skiprows=4, header=None,
                         names=['src', 'dst'], compression='gzip')
    src = np.array(df['src'])
    dst = np.array(df['dst'])
    print('construct the graph')
    return dgl.DGLGraph((src, dst), readonly=True)

def get_graph(name):
    if name == 'livejournal':
        return get_livejournal()
    else:
        print(name + " doesn't exist")
        return None

class OGBDataset(object):
    def __init__(self, g, num_labels, predict_category=None):
        self._g = g
        self._num_labels = num_labels
        self._predict_category = predict_category

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_classes(self):
        return self._num_labels

    @property
    def predict_category(self):
        return self._predict_category

    def __getitem__(self, idx):
        return self._g

def load_ogb_product(name):
    name = 'ogbn-products'
    from ogb.nodeproppred import DglNodePropPredDataset

    os.symlink('/tmp/dataset/', os.path.join(os.getcwd(), 'dataset'))

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['label'] = labels
    in_feats = graph.ndata['feat'].shape[1]
    num_labels = len(torch.unique(labels[torch.logical_not(torch.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    val_mask[val_nid] = True
    test_mask = torch.zeros((graph.number_of_nodes(),), dtype=torch.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    return OGBDataset(graph, num_labels)

def load_ogb_mag():
    name = 'ogbn-mag'
    from ogb.nodeproppred import DglNodePropPredDataset

    os.symlink('/tmp/dataset/', os.path.join(os.getcwd(), 'dataset'))

    print('load', name)
    dataset = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx["train"]['paper']
    val_idx = split_idx["valid"]['paper']
    test_idx = split_idx["test"]['paper']
    hg_orig, labels = dataset[0]
    subgs = {}
    for etype in hg_orig.canonical_etypes:
        u, v = hg_orig.all_edges(etype=etype)
        subgs[etype] = (u, v)
        subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
    hg = dgl.heterograph(subgs)
    hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
    hg.nodes['paper'].data['label'] = labels['paper'].squeeze()
    train_mask = torch.zeros((hg.number_of_nodes('paper'),), dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask = torch.zeros((hg.number_of_nodes('paper'),), dtype=torch.bool)
    val_mask[val_idx] = True
    test_mask = torch.zeros((hg.number_of_nodes('paper'),), dtype=torch.bool)
    test_mask[test_idx] = True
    hg.nodes['paper'].data['train_mask'] = train_mask
    hg.nodes['paper'].data['val_mask'] = val_mask
    hg.nodes['paper'].data['test_mask'] = test_mask

    num_classes = dataset.num_classes
    return OGBDataset(hg, num_classes, 'paper')

def process_data(name):
    if name == 'cora':
        return dgl.data.CoraGraphDataset()
    elif name == 'pubmed':
        return dgl.data.PubmedGraphDataset()
    elif name == 'reddit':
        return dgl.data.RedditDataset(self_loop=True)
    elif name == 'ogbn-products':
        return load_ogb_product()
    elif name == 'ogbn-mag':
        return load_ogb_mag()
    else:
        raise ValueError('Invalid dataset name:', name)

def get_bench_device():
    return os.environ.get('DGL_BENCH_DEVICE', 'cpu')

def setup_track_time(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)

def setup_track_acc(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)

TRACK_UNITS = {
    'time' : 's',
    'acc' : '%',
}

TRACK_SETUP = {
    'time' : setup_track_time,
    'acc' : setup_track_acc,
}

def parametrize(param_name, params):
    def _wrapper(func):
        if getattr(func, 'params', None) is None:
            func.params = []
        func.params.append(params)
        if getattr(func, 'param_names', None) is None:
            func.param_names = []
        func.param_names.append(param_name)
        return func
    return _wrapper

def benchmark(track_type, timeout=60):
    assert track_type in ['time', 'acc']
    def _wrapper(func):
        func.unit = TRACK_UNITS[track_type]
        func.setup = TRACK_SETUP[track_type]
        func.timeout = timeout
        return func
    return _wrapper
