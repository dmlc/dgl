import json
import os
import pickle
import shutil
import zipfile
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
    # Same as https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz
    _download('https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/livejournal/soc-LiveJournal1.txt.gz',
              '/tmp/dataset', 'soc-LiveJournal1.txt.gz')
    df = pandas.read_csv('/tmp/dataset/soc-LiveJournal1.txt.gz', sep='\t', skiprows=4, header=None,
                         names=['src', 'dst'], compression='gzip')
    src = df['src'].values
    dst = df['dst'].values
    print('construct the graph')
    return dgl.graph((src, dst))


def get_filmbaster():
    # Same as https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz
    _download('https://dgl-asv-data.s3-us-west-2.amazonaws.com/dataset/friendster/com-friendster.ungraph.txt.gz',
              '/tmp/dataset', 'com-friendster.ungraph.txt.gz')
    df = pandas.read_csv('/tmp/dataset/com-friendster.ungraph.txt.gz', sep='\t', skiprows=4, header=None,
                         names=['src', 'dst'], compression='gzip')
    src = df['src'].values
    dst = df['dst'].values
    print('construct the graph')
    return dgl.graph((src, dst))


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
    num_labels = len(torch.unique(
        labels[torch.logical_not(torch.isnan(labels))]))

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
    import torchtext
    # follow examples/pytorch/pinsage/README to create nowplaying_rs.pkl
    name = 'nowplaying_rs.pkl'
    dataset_dir = os.path.join(os.getcwd(), 'dataset')
    os.symlink('/tmp/dataset/', dataset_dir)

    dataset_path = os.path.join(dataset_dir, "nowplaying_rs", name)
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

    # Assign user and movie IDs and use them as features (to learn an individual trainable
    # embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(
        g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(
        g.number_of_nodes(item_ntype))

    # Prepare torchtext dataset and vocabulary
    fields = {}
    examples = []
    for key, texts in item_texts.items():
        fields[key] = torchtext.data.Field(
            include_lengths=True, lower=True, batch_first=True)
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
    device = os.environ.get('DGL_BENCH_DEVICE', 'cpu')
    if device.lower() == "gpu":
        return "cuda:0"
    else:
        return device


def setup_track_time(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)


def setup_track_acc(*args, **kwargs):
    # fix random seed
    np.random.seed(42)
    torch.random.manual_seed(42)


TRACK_UNITS = {
    'time': 's',
    'acc': '%',
}

TRACK_SETUP = {
    'time': setup_track_time,
    'acc': setup_track_acc,
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


class TestFilter:
    def __init__(self):
        self.conf = None
        if "DGL_REG_CONF" in os.environ:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(current_dir, "../../",
                                os.environ["DGL_REG_CONF"])
            with open(path, "r") as f:
                self.conf = json.load(f)
            if "INSTANCE_TYPE" in os.environ:
                instance_type = os.environ["INSTANCE_TYPE"]
            else:
                raise Exception(
                    "Must set both DGL_REG_CONF and INSTANCE_TYPE as env")
            self.enabled_tests = self.conf[instance_type]["tests"]
        else:
            import logging
            logging.warning("No regression test conf file specified")

    def check(self, func):
        funcfullname = inspect.getmodule(func).__name__ + "." + func.__name__
        if self.conf is None:
            return True
        else:
            for enabled_testname in self.enabled_tests:
                if enabled_testname in funcfullname:
                    return True
            return False


filter = TestFilter()


def benchmark(track_type, timeout=60):
    """Decorator for indicating the benchmark type.
       By default, the test will not run on multi-gpu testing

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
        if not filter.check(func):
            # skip if not enabled
            func.benchmark_name = "skip_" + func.__name__
        return func
    return _wrapper


def multi_gpu_test():
    def _wrapper(func):
        if hasattr(func, "benchmark_name"):
            if num_gpu() > 1 and func.benchmark_name.startswith("skip_multigpu_"):
                func.benchmark_name = func.benchmark_name[len(
                    "skip_multigpu_"):]
        return func
    return _wrapper


def run_when(run):
    def _wrapper(func):
        if not run:
            print("Skip {}".format(func.__name__))
            func.benchmark_name = "skip_" + func.__name__
        return func
    return _wrapper
