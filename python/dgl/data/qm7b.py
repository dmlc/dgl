from scipy import io
import numpy as np
import os

from .dgl_dataset import DGLDataset
from .utils import download, save_graphs, load_graphs, check_sha1
from ..graph import DGLGraph
from .. import backend as F
from ..base import dgl_warning


class QM7bDataset(DGLDataset):
    r"""QM7b dataset for graph property prediction

    This dataset consists of 7,211 molecules with 14 regression targets.
    Nodes means atoms and edges means bonds. Edge data 'h' means
    the entry of Coulomb matrix.

    Reference: http://quantum-machine.org/datasets/

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ----------
    Number of graphs: 7,211
    Number of regression targets: 14
    Average number of nodes: 15
    Average number of edges: 245
    Edge feature size: 1

    Parameters
    ----------
    force_reload: bool

    Returns
    -------
    QM7bDataset object with two properties:
        graphs: a list of DGLGraph objects each with
            - edata['h']: edge feature, which is the entry of Coulomb matrix
        label: labels of the 14 regression targets, float tensor with size [7211, 14]

    Examples
    --------
    >>> data = QM7bDataset()
    >>> graphs = data.graphs  # get the list of graphs
    >>> labels = data.label   # get the labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/qm7b.mat'
        super(QM7bDataset, self).__init__(name='qm7b', url=url, raw_dir=raw_dir,
                                          force_reload=force_reload, verbose=verbose)

    def process(self, root_path):
        mat_path = root_path + '.mat'
        if not check_sha1(mat_path, '4102c744bb9d6fd7b40ac67a300e49cd87e28392'):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name))
        self.graphs, self.label = self._load_graph(mat_path)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        labels = F.tensor(data['T'], dtype=F.data_type_dict['float32'])
        feats = data['X']
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            g = DGLGraph()
            edge_list = feats[i].nonzero()
            num_nodes = np.max(edge_list) + 1
            g.add_nodes(num_nodes)
            g.add_edges(edge_list[0], edge_list[1])
            g.edata['h'] = feats[i][edge_list[0], edge_list[1]].reshape(-1, 1)
            graphs.append(g)
        return graphs, labels

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(str(graph_path), self.graphs, {'labels': self.label})

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
        self.graphs = graphs
        self.label = label_dict['labels']

    def download(self):
        r""" Automatically download data and extract it. """
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        download(self.url, path=file_path)

    def __getitem__(self, idx):
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        return len(self.graphs)


class QM7b(QM7bDataset):
    def __init__(self):
        dgl_warning('QM7b is deprecated, use QM7bDataset instead.', DeprecationWarning, stacklevel=2)
        super(QM7b, self).__init__()
