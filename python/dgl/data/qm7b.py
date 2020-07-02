"""QM7b dataset for graph property prediction (regression)."""
from scipy import io
import numpy as np
import os

from .dgl_dataset import DGLDataset
from .utils import download, save_graphs, load_graphs, \
    check_sha1, deprecate_property, deprecate_class
from ..graph import DGLGraph
from .. import backend as F


class QM7bDataset(DGLDataset):
    r"""QM7b dataset for graph property prediction (regression)

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
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    QM7bDataset object with three properties:
        graphs: a list of DGLGraph objects each with
            - edata['h']: edge feature, which is the entry of Coulomb matrix
        labels: labels of the 14 regression targets, float tensor with size [7211, 14]
        num_labels: number of labels for each graph, i.e. number of prediction tasks

    Examples
    --------
    >>> data = QM7bDataset()
    >>> graphs = data.graphs  # get the list of graphs
    >>> labels = data.labels  # get the labels
    >>> data.num_labels
    14
    >>>
    >>> # iterate over the dataset
    >>> for g, label in data:
    ...     edge_feat = g.edata['h']  # get edge feature
    ...     # your code here...
    ...
    >>>
    """

    _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
           'datasets/qm7b.mat'
    _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'

    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(QM7bDataset, self).__init__(name='qm7b',
                                          url=self._url,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def process(self, root_path):
        mat_path = root_path + '.mat'
        if not check_sha1(mat_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name))
        self._graphs, self._labels = self._load_graph(mat_path)

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
        save_graphs(str(graph_path), self.graphs, {'labels': self.labels})

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def load(self):
        graphs, label_dict = load_graphs(os.path.join(self.save_path, 'dgl_graph.bin'))
        self._graphs = graphs
        self._labels = label_dict['labels']

    def download(self):
        r""" Automatically download data and extract it. """
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        download(self.url, path=file_path)

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 14

    @property
    def label(self):
        deprecate_property('dataset.label', 'dataset.labels')
        return self._labels

    @property
    def labels(self):
        """A list of tensor that indicates labels for the graphs."""
        return self._labels

    @property
    def graphs(self):
        """A list of DGLGraph objects."""
        return self._graphs

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


class QM7b(QM7bDataset):
    def __init__(self):
        deprecate_class('QM7b', 'QM7bDataset')
        super(QM7b, self).__init__()
