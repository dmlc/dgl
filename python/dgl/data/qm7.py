from scipy import io
import numpy as np
from dgl import DGLGraph

class QM7(object):

    _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
        'datasets/qm7b.mat'

    def __init__(self):
        self.dir = get_download_dir()
        self.path = os.path.join(self.dir, 'qm7', "qm7b.mat")
        download(_url, path=self.path)
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