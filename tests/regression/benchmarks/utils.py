import os
import shutil, zipfile
import requests
import numpy as np
import pandas
import dgl
from tqdm import tqdm

def _download(url, path, filename):
    fn = os.path.join(path, filename)
    if os.path.exists(fn):
        return

    os.makedirs(path, exist_ok=True)
    f_remote = requests.get(url, stream=True)
    sz = f_remote.headers.get('content-length')
    assert f_remote.status_code == 200, 'fail to open {}'.format(url)
    with open(fn, 'wb') as writer:
        for chunk in tqdm(f_remote.iter_content(chunk_size=1024*1024)):
            writer.write(chunk)
    print('Download finished.')

def get_livejournal():
    _download('https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz',
              '/tmp', 'soc-LiveJournal1.txt.gz')
    df = pandas.read_csv('/tmp/soc-LiveJournal1.txt.gz', sep='\t', skiprows=4, header=None,
                         names=['src', 'dst'], compression='gzip')
    src = np.array(df['src'])
    dst = np.array(df['dst'])
    print('construct soc-LiveJournal')
    return dgl.DGLGraph((src, dst), readonly=True)

def get_friendster():
    _download('https://snap.stanford.edu/data/bigdata/communities/com-friendster.ungraph.txt.gz',
              '/tmp', 'com-friendster.ungraph.txt.gz')
    df = pandas.read_csv('/tmp/com-friendster.ungraph.txt.gz', sep='\t', skiprows=4, header=None,
                         names=['src', 'dst'], compression='gzip')
    src = np.array(df['src'])
    dst = np.array(df['dst'])
    print('construct friendster')
    return dgl.DGLGraph((src, dst), readonly=True)

def get_graph(name):
    if name == 'livejournal':
        return get_livejournal()
    elif name == 'friendster':
        return get_friendster()
    else:
        print(name + " doesn't exist")
        return None
