from scipy import io
import numpy as np
from dgl import DGLGraph
import os
import datetime

from .utils import get_download_dir, download, extract_archive


class GDELT(object):
    """
    The Global Database of Events, Language, and Tone (GDELT) dataset.
    This contains events happend all over the world (ie every protest held anywhere
     in Russia on a given day is collapsed to a single entry).

    This Dataset consists of
    events collected from 1/1/2018 to 1/31/2018 (15 minutes time granularity).
    
    Reference:
    - `Recurrent Event Network for Reasoning over Temporal
    Knowledge Graphs <https://arxiv.org/abs/1904.05530>`_
    - `The Global Database of Events, Language, and Tone (GDELT) <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28075>`_


    Parameters
    ------------
    mode: str
      Load train/valid/test data. Has to be one of ['train', 'valid', 'test']

    """
    _url = {
        'train': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/train.txt',
        'valid': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/valid.txt',
        'test': 'https://github.com/INK-USC/RENet/raw/master/data/GDELT/test.txt',
    }

    def __init__(self, mode):
        assert mode.lower() in self._url, "Mode not valid"
        self.dir = get_download_dir()
        self.mode = mode
        self.graphs = []
        for dname in self._url:
            dpath = os.path.join(
                self.dir, 'GDELT', self._url[dname.lower()].split('/')[-1])
            download(self._url[dname.lower()], path=dpath)
        train_data = np.loadtxt(os.path.join(
            self.dir, 'GDELT', 'train.txt'), delimiter='\t').astype(np.int64)
        if self.mode == 'train':
            self._load(train_data)
        elif self.mode == 'valid':
            val_data = np.loadtxt(os.path.join(
                self.dir, 'GDELT', 'valid.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            self._load(np.concatenate([train_data, val_data], axis=0))
        elif self.mode == 'test':
            val_data = np.loadtxt(os.path.join(
                self.dir, 'GDELT', 'valid.txt'), delimiter='\t').astype(np.int64)
            test_data = np.loadtxt(os.path.join(
                self.dir, 'GDELT', 'test.txt'), delimiter='\t').astype(np.int64)
            train_data[:, 3] = -1
            val_data[:, 3] = -1
            self._load(np.concatenate(
                [train_data, val_data, test_data], axis=0))

    def _load(self, data):
        num_nodes = 23033
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        time_index = np.floor(data[:, 3]/15).astype(np.int64)
        start_time = time_index[time_index != -1].min()
        end_time = time_index.max()
        for i in range(start_time, end_time+1):
            g = DGLGraph()
            g.add_nodes(num_nodes)
            row_mask = time_index <= i
            edges = data[row_mask][:, [0, 2]]
            rate = data[row_mask][:, 1]
            g.add_edges(edges[:, 0], edges[:, 1])
            g.edata['rel_type'] = rate.reshape(-1, 1)
            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
