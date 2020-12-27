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
        self._g = g
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

    return ogb_data(graph, num_labels)

class PinsageDataset:
    def __init__(self, g, user_ntype, item_ntype, textset):
        self._g = g
        self._user_ntype = user_ntype
        self._item_ntype = item_ntype
        self._textset = textset

    @property
    def user_ntype(self):
        return self._user_ntype

    @property
    def item_ntype(self):
        return self._item_ntype

    @property
    def textset(self):
        return self._textset

    def __getitem__(self, idx):
        return self._g

def load_nowplaying_rs():
    name = 'nowplaying_rs.pkl' # follow examples/pytorch/pinsage/README to create nowplaying_rs.pkl
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    os.symlink('/tmp/dataset/', dataset_dir)

    dataset_path = os.path.join(dataset_dir, name)
    # Load dataset
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    g = dataset['train-graph']
    val_matrix = dataset['val-matrix'].tocsr()
    test_matrix = dataset['test-matrix'].tocsr()
    item_texts = dataset['item-texts']
    user_ntype = dataset['user-type']
    item_ntype = dataset['item-type']
    user_to_item_etype = dataset['user-to-item-type']
    timestamp = dataset['timestamp-edge-column']

    device = torch.device(args.device)

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))

    # Prepare torchtext dataset and vocabulary
    fields = {}
    examples = []
    for key, texts in item_texts.items():
        fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
    for i in range(g.number_of_nodes(item_ntype)):
        example = torchtext.data.Example.fromlist(
            [item_texts[key][i] for key in item_texts.keys()],
            [(key, fields[key]) for key in item_texts.keys()])
        examples.append(example)
    textset = torchtext.data.Dataset(examples, fields)
    for key, field in fields.items():
        field.build_vocab(getattr(textset, key))

    return PinsageDataset(g, user_ntype, item_ntype, textset)

def process_data(name):
    if name == 'cora':
        return dgl.data.CoraGraphDataset()
    elif name == 'pubmed':
        return dgl.data.PubmedGraphDataset()
    elif name == 'reddit':
        return dgl.data.RedditDataset(self_loop=True)
    elif name == 'ogbn-products':
        return load_ogb_product('ogbn-products')
    elif name == 'nowplaying_rs':
        return load_nowplaying_rs()
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
