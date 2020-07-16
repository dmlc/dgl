""" PPIDataset for inductive learning. """
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import _get_dgl_url, save_graphs, save_info, load_info, load_graphs, deprecate_property
from ..graph import DGLGraph
from .. import backend as F


class PPIDataset(DGLBuiltinDataset):
    """ Protein-Protein Interaction dataset for inductive node classification

    A toy Protein-Protein Interaction network dataset. The dataset contains
    24 graphs. The average number of nodes per graph is 2372. Each node has
    50 features and 121 labels. 20 graphs for training, 2 for validation
    and 2 for testing.

    Reference: http://snap.stanford.edu/graphsage/

    Statistics
    ----------
    Train examples: 20
    Valid examples: 2
    Test examples: 2

    Parameters
    ----------
    mode : str
        Must be one of ('train', 'valid', 'test'). Default: 'train'
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Returns
    -------
    PPIDataset object with two properties
        graphs: list of DGLGraph objects that contains graph structure, node features and node labels:
            - ndata['feat']: node features
            - ndata['label']: nodel labels
        graph: DGLGraph, the graph of the dataset
        num_labels: int, number of labels for each node
    Examples
    --------
    >>> data = PPIDataset(mode='valid')
    >>> num_labels = data.num_labels
    >>> for g, _ in data:
    ....    feat = g.ndata['feat']
    ....    label = g.ndata['label']
    ....    # your code here
    """
    def __init__(self, mode='train', raw_dir=None, force_reload=False, verbose=False):
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        _url = _get_dgl_url('dataset/ppi.zip')
        super(PPIDataset, self).__init__(name='ppi',
                                         url=_url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        graph_file = os.path.join(self.save_path, '{}_graph.json'.format(self.mode))
        label_file = os.path.join(self.save_path, '{}_labels.npy'.format(self.mode))
        feat_file = os.path.join(self.save_path, '{}_feats.npy'.format(self.mode))
        graph_id_file = os.path.join(self.save_path, '{}_graph_id.npy'.format(self.mode))

        g_data = json.load(open(graph_file))
        self._labels = np.load(label_file)
        self._feats = np.load(feat_file)
        self.graph = DGLGraph(nx.DiGraph(json_graph.node_link_graph(g_data)))
        graph_id = np.load(graph_id_file)

        # lo, hi means the range of graph ids for different portion of the dataset,
        # 20 graphs for training, 2 for validation and 2 for testing.
        lo, hi = 1, 21
        if self.mode == 'valid':
            lo, hi = 21, 23
        elif self.mode == 'test':
            lo, hi = 23, 25

        graph_masks = []
        self.graphs = []
        for g_id in range(lo, hi):
            g_mask = np.where(graph_id == g_id)[0]
            graph_masks.append(g_mask)
            g = self.graph.subgraph(g_mask)
            g.ndata['feat'] = F.tensor(self._feats[g_mask], dtype=F.data_type_dict['float32'])
            g.ndata['label'] = F.tensor(self._labels[g_mask], dtype=F.data_type_dict['int64'])
            self.graphs.append(g)

    def has_cache(self):
        graph_list_path = os.path.join(self.save_path, '{}_dgl_graph_list.bin'.format(self.mode))
        g_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        info_path = os.path.join(self.save_path, '{}_info.pkl'.format(self.mode))
        return os.path.exists(graph_list_path) and os.path.exists(g_path) and os.path.exists(info_path)

    def save(self):
        graph_list_path = os.path.join(self.save_path, '{}_dgl_graph_list.bin'.format(self.mode))
        g_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        info_path = os.path.join(self.save_path, '{}_info.pkl'.format(self.mode))
        save_graphs(graph_list_path, self.graphs)
        save_graphs(g_path, self.graph)
        save_info(info_path, {'labels': self._labels, 'feats': self._feats})

    def load(self):
        graph_list_path = os.path.join(self.save_path, '{}_dgl_graph_list.bin'.format(self.mode))
        g_path = os.path.join(self.save_path, '{}_dgl_graph.bin'.format(self.mode))
        info_path = os.path.join(self.save_path, '{}_info.pkl'.format(self.mode))
        self.graphs = load_graphs(graph_list_path)[0]
        g, _ = load_graphs(g_path)
        self.graph = g[0]
        info = load_info(info_path)
        self._labels = info['labels']
        self._feats = info['feats']

    @property
    def num_labels(self):
        return 121

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'dataset.graphs[i].ndata[\'label\']')
        return self._labels

    @property
    def features(self):
        deprecate_property('dataset.features', 'dataset.graphs[i].ndata[\'feat\']')
        return self._feats

    def __len__(self):
        """Return number of samples in this dataset."""
        return len(self.graphs)

    def __getitem__(self, item):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, ndarray)
            The graph, and its label.
        """
        return self.graphs[item], self.graphs[item].ndata['label']


class LegacyPPIDataset(PPIDataset):
    """Legacy version of PPI Dataset
    """

    def __getitem__(self, item):
        """Get the i^th sample.

        Paramters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, ndarray, ndarray)
            The graph, features and its label.
        """

        return self.graphs[item], self.graphs[item].ndata['feat'], self.graphs[item].ndata['label']
