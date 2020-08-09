"""QM7b dataset for graph property prediction (regression)."""
from scipy import io
import numpy as np
import os

from .dgl_dataset import DGLDataset
from .utils import download, save_graphs, load_graphs, \
    check_sha1, deprecate_property
from .. import backend as F
from ..convert import graph as dgl_graph


class QM7bDataset(DGLDataset):
    r"""QM7b dataset for graph property prediction (regression)

    This dataset consists of 7,211 molecules with 14 regression targets.
    Nodes means atoms and edges means bonds. Edge data 'h' means
    the entry of Coulomb matrix.

    Reference: `<http://quantum-machine.org/datasets/>`_

    Statistics:

    - Number of graphs: 7,211
    - Number of regression targets: 14
    - Average number of nodes: 15
    - Average number of edges: 245
    - Edge feature size: 1

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_labels : int
        Number of labels for each graph, i.e. number of prediction tasks

    Raises
    ------
    UserWarning
        If the raw data is changed in the remote server by the author.

    Examples
    --------
    >>> data = QM7bDataset()
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

    def process(self):
        mat_path = self.raw_path + '.mat'
        self.graphs, self.label = self._load_graph(mat_path)

    def _load_graph(self, filename):
        data = io.loadmat(filename)
        labels = F.tensor(data['T'], dtype=F.data_type_dict['float32'])
        feats = data['X']
        num_graphs = labels.shape[0]
        graphs = []
        for i in range(num_graphs):
            edge_list = feats[i].nonzero()
            g = dgl_graph(edge_list)
            g.edata['h'] = F.tensor(feats[i][edge_list[0], edge_list[1]].reshape(-1, 1),
                                    dtype=F.data_type_dict['float32'])
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
        file_path = os.path.join(self.raw_dir, self.name + '.mat')
        download(self.url, path=file_path)
        if not check_sha1(file_path, self._sha1_str):
            raise UserWarning('File {} is downloaded but the content hash does not match.'
                              'The repo may be outdated or download may be incomplete. '
                              'Otherwise you can create an issue for it.'.format(self.name))

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        return 14

    def __getitem__(self, idx):
        r""" Get graph and label by index

        Parameters
        ----------
        idx : int
            Item index

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
        """
        return self.graphs[idx], self.label[idx]

    def __len__(self):
        r"""Number of graphs in the dataset.

        Return
        -------
        int
        """
        return len(self.graphs)


QM7b = QM7bDataset
