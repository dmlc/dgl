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

def process_data(name):
    if name == 'cora':
        return dgl.data.CoraGraphDataset()
    elif name == 'pubmed':
        return dgl.data.PubmedGraphDataset()
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

def benchmark(track_type):
    assert track_type in ['time', 'acc']
    def _wrapper(func):
        func.unit = TRACK_UNITS[track_type]
        func.setup = TRACK_SETUP[track_type]
        return func
    return _wrapper
