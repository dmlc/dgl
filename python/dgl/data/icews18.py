from scipy import io
import numpy as np
from dgl import DGLGraph
import os
import datetime

from .utils import get_download_dir, download, extract_archive


class ICEWS18(object):
    _url = {
        'train': 'https://github.com/INK-USC/RENet/raw/master/data/ICEWS18/train.txt',
        'valid': 'https://github.com/INK-USC/RENet/raw/master/data/ICEWS18/valid.txt',
        'test': 'https://github.com/INK-USC/RENet/raw/master/data/ICEWS18/test.txt',
    }
    splits = [0, 373018, 419013, 468558]  # Train/Val/Test splits.

    def __init__(self, name):
        assert name.lower() in self._url, "Name not valid"
        self.dir = get_download_dir()
        self.name = name
        self.graphs = []
        for dname in self._url:
            dpath = os.path.join(
                self.dir, 'ICEWS18', self._url[dname.lower()].split('/')[-1])
            download(self._url[dname.lower()], path=dpath)
        train_data = np.loadtxt(os.path.join(
            self.dir, 'ICEWS18', 'train.txt'), delimiter='\t').astype(np.int64)
        if self.name == 'train':
            self._load(train_data)
        elif self.name == 'valid':
            val_data = np.loadtxt(os.path.join(
                self.dir, 'ICEWS18', 'valid.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            self._load(np.concatenate([train_data, val_data], axis=0))
        elif self.name == 'test':
            val_data = np.loadtxt(os.path.join(
                self.dir, 'ICEWS18', 'valid.txt'), delimiter='\t').astype(np.int64)
            test_data = np.loadtxt(os.path.join(
                self.dir, 'ICEWS18', 'test.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            val_data[:, 3] = -1
            self._load(np.concatenate(
                [train_data, val_data, test_data], axis=0))

    def _load(self, data):
        num_nodes = 23033
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        time_index = np.floor(data[:, 3]/24).astype(np.int64)
        start_time = time_index[time_index != -1].min()
        end_time = time_index.max()
        print(time_index)
        print(self.name)
        for i in range(start_time, end_time+1):
            g = DGLGraph()
            g.add_nodes(num_nodes)
            row_mask = time_index <= i
            edges = data[row_mask][:, [0, 2]]
            rate = data[row_mask][:, 1]
            g.add_edges(edges[:, 0], edges[:, 1])
            g.edata['h'] = rate.reshape(-1, 1)
            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
