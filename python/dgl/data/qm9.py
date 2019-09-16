from scipy import io
import numpy as np
from dgl import DGLGraph
import os

from .utils import get_download_dir, download, extract_archive

class QM9(object):

    _url = 'http://www.roemisch-drei.de/qm9.tar.gz'

    def __init__(self):
        self.dir = get_download_dir()
        self.path = os.path.join(self.dir, 'qm9', "qm9.tar.gz")
        download(_url, path=self.path)
        extract_archive(self.path, os.path.join(self.dir, 'qm9', 'qm9'))
        self.graphs = []
        self._load(self.path)

    def _load(self, filename):
        data = io.loadmat(self.path)
        labels = data['T']
        feats = data['X']
        num_graphs = labels.shape[0]
        for i in range(num_graphs):
            g = DGLGraph()
            edge_list = feats[i].nonzero()
            num_nodes = np.max(edge_list) + 1
            g.add_nodes(num_nodes)
            g.add_edges(edge_list[0], edge_list[1])
            g.edata['h'] = feats[i][edge_list[0], edge_list[1]].reshape(-1, 1)
            self.graphs.append(g)

    def __getitem__(self, idx):
        return self.graphs[idx]

    def __len__(self):
        return len(self.graphs)
