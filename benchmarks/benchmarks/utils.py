import os
import shutil, zipfile
import requests
import inspect
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

    return OGBDataset(graph, num_labels)

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
    elif name == 'aifb':
        return dgl.data.AIFBDataset()
    elif name == 'mutag':
        return dgl.data.MUTAGDataset()
    elif name == 'bgs':
        return dgl.data.BGSDataset()
    elif name == 'am':
        return dgl.data.AMDataset()
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
    """Decorator for benchmarking over a set of parameters.

    Parameters
    ----------
    param_name : str
        Parameter name. Must be one of the arguments of the decorated function.
    params : list[any]
        List of values to benchmark for the given parameter name. Recommend
        to use Python's native object type (e.g., int, str, list[int]) because
        ASV will display them on the plot.

    Examples
    --------

    Benchmark function `foo` when argument `x` is equal to 10 or 20.

    .. code::
        @benchmark('time')
        @parametrize('x', [10, 20]):
        def foo(x):
            pass

    Benchmark function with multiple parametrizations. It will run the function
    with all possible combinations. The example below generates 6 benchmarks.

    .. code::
        @benchmark('time')
        @parametrize('x', [10, 20]):
        @parametrize('y', [-1, -2, -3]):
        def foo(x, y):
            pass

    When using multiple parametrizations, it can have arbitrary order. The example
    below is the same as the above one.

    .. code::
        @benchmark('time')
        @parametrize('y', [-1, -2, -3]):
        @parametrize('x', [10, 20]):
        def foo(x, y):
            pass
    """
    def _wrapper(func):
        sig_params = inspect.signature(func).parameters.keys()
        num_params = len(sig_params)
        if getattr(func, 'params', None) is None:
            func.params = [None] * num_params
        if getattr(func, 'param_names', None) is None:
            func.param_names = [None] * num_params
        found_param = False
        for i, sig_param in enumerate(sig_params):
            if sig_param == param_name:
                func.params[i] = params
                func.param_names[i] = param_name
                found_param = True
                break
        if not found_param:
            raise ValueError('Invalid parameter name:', param_name)
        return func
    return _wrapper

def benchmark(track_type, timeout=60):
    """Decorator for indicating the benchmark type.

    Parameters
    ----------
    track_type : str
        Type. Must be either:

            - 'time' : For timing. Unit: second.
            - 'acc' : For accuracy. Unit: percentage, value between 0 and 100.
    timeout : int
        Timeout threshold in second.

    Examples
    --------

    .. code::
        @benchmark('time')
        def foo():
            pass
    """
    assert track_type in ['time', 'acc']
    def _wrapper(func):
        func.unit = TRACK_UNITS[track_type]
        func.setup = TRACK_SETUP[track_type]
        func.timeout = timeout
        return func
    return _wrapper
