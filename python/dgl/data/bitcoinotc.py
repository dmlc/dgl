from scipy import io
import numpy as np
from dgl import DGLGraph
import os
import datetime

from .utils import get_download_dir, download, extract_archive


class BitcoinOTC(object):
    """
    This is who-trusts-whom network of people who trade using Bitcoin 
    on a platform called Bitcoin OTC. 
    Since Bitcoin users are anonymous, there is a need to maintain a
     record of users' reputation to prevent transactions with fraudulent
     and risky users. Members of Bitcoin OTC rate other members in a
     scale of -10 (total distrust) to +10 (total trust) in steps of 1. 

    Reference:
    - `Bitcoin OTC trust weighted signed network <http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html>`_
    - `EvolveGCN: Evolving Graph
    Convolutional Networks for Dynamic Graphs
    <https://arxiv.org/abs/1902.10191>`_
    """
    _url = 'https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz'

    def __init__(self):
        self.dir = get_download_dir()
        self.zip_path = os.path.join(
            self.dir, 'bitcoin', "soc-sign-bitcoinotc.csv.gz")
        download(self._url, path=self.zip_path)
        extract_archive(self.zip_path, os.path.join(
            self.dir, 'bitcoin'))
        self.path = os.path.join(
            self.dir, 'bitcoin', "soc-sign-bitcoinotc.csv")
        self.graphs = []
        self._load(self.path)

    def _load(self, filename):
        data = np.loadtxt(filename, delimiter=',').astype(np.int64)
        data[:, 0:2] = data[:, 0:2] - data[:, 0:2].min()
        num_nodes = data[:, 0:2].max() - data[:, 0:2].min() + 1
        delta = datetime.timedelta(days=14).total_seconds()
        # The source code is not released, but the paper indicates there're
        # totally 137 samples. The cutoff below has exactly 137 samples.
        time_index = np.around(
            (data[:, 3] - data[:, 3].min())/delta).astype(np.int64)
        for i in range(time_index.max()):
            g = DGLGraph()
            g.add_nodes(num_nodes)
            row_mask = time_index <= i
            edges = data[row_mask][:, 0:2]
            rate = data[row_mask][:, 2]
            g.add_edges(edges[:, 0], edges[:, 1])
            g.edata['h'] = rate.reshape(-1, 1)
            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)

    @property
    def is_temporal(self):
        return True