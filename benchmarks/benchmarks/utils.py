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

class ogb_data(object):
    def __init__(self, g, num_labels):
        self._g = [g]
        self._num_labels = num_labels

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def num_classes(self):
        return self._num_labels

    def __getitem__(self, idx):
        return self._g

def load_ogb_product(name):
    from ogb.nodeproppred import DglNodePropPredDataset

    print('load', name)
    data = DglNodePropPredDataset(name=name)
    print('finish loading', name)
    splitted_idx = data.get_idx_split()
    graph, labels = data[0]
    labels = labels[:, 0]

    graph.ndata['features'] = graph.ndata['feat']
    graph.ndata['labels'] = labels
    in_feats = graph.ndata['features'].shape[1]
    num_labels = len(th.unique(labels[th.logical_not(th.isnan(labels))]))

    # Find the node IDs in the training, validation, and test set.
    train_nid, val_nid, test_nid = splitted_idx['train'], splitted_idx['valid'], splitted_idx['test']
    train_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    train_mask[train_nid] = True
    val_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    val_mask[val_nid] = True
    test_mask = th.zeros((graph.number_of_nodes(),), dtype=th.bool)
    test_mask[test_nid] = True
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    return ogb_data(graph, num_labels)

def process_data(name):
    if name == 'cora':
        return dgl.data.CoraGraphDataset()
    elif name == 'pubmed':
        return dgl.data.PubmedGraphDataset()
    elif name == 'reddit':
        return dgl.data.RedditDataset(self_loop=True)
    elif name == 'ogbn-products':
        return load_ogb_product('ogbn-products')
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
