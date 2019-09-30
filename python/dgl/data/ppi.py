"""PPI Dataset.
(zhang hao): Used for inductive learning.
"""
import json
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

from .utils import download, extract_archive, get_download_dir, _get_dgl_url
from ..graph import DGLGraph

_url = 'dataset/ppi.zip'


class PPIDataset(object):
    """A toy Protein-Protein Interaction network dataset.

    Adapted from https://github.com/williamleif/GraphSAGE/tree/master/example_data.

    The dataset contains 24 graphs. The average number of nodes per graph
    is 2372. Each node has 50 features and 121 labels.

    We use 20 graphs for training, 2 for validation and 2 for testing.
    """
    def __init__(self, mode):
        """Initialize the dataset.

        Paramters
        ---------
        mode : str
            ('train', 'valid', 'test').
        """
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self._load()
        self._preprocess()

    def _load(self):
        """Loads input data.

        train/test/valid_graph.json => the graph data used for training,
          test and validation as json format;
        train/test/valid_feats.npy => the feature vectors of nodes as
          numpy.ndarry object, it's shape is [n, v],
          n is the number of nodes, v is the feature's dimension;
        train/test/valid_labels.npy=> the labels of the input nodes, it
          is a numpy ndarry, it's like[[0, 0, 1, ... 0], 
          [0, 1, 1, 0 ...1]], shape of it is n*h, n is the number of nodes,
          h is the label's dimension;
        train/test/valid/_graph_id.npy => the element in it indicates which
          graph the nodes belong to, it is a one dimensional numpy.ndarray
          object and the length of it is equal the number of nodes,
          it's like [1, 1, 2, 1...20]. 
        """
        name = 'ppi'
        dir = get_download_dir()
        zip_file_path = '{}/{}.zip'.format(dir, name)
        download(_get_dgl_url(_url), path=zip_file_path)
        extract_archive(zip_file_path,
                        '{}/{}'.format(dir, name))
        print('Loading G...')
        if self.mode == 'train':
            with open('{}/ppi/train_graph.json'.format(dir)) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load('{}/ppi/train_labels.npy'.format(dir))
            self.features = np.load('{}/ppi/train_feats.npy'.format(dir))
            self.graph = DGLGraph(nx.DiGraph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load('{}/ppi/train_graph_id.npy'.format(dir))
        if self.mode == 'valid':
            with open('{}/ppi/valid_graph.json'.format(dir)) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load('{}/ppi/valid_labels.npy'.format(dir))
            self.features = np.load('{}/ppi/valid_feats.npy'.format(dir))
            self.graph = DGLGraph(nx.DiGraph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load('{}/ppi/valid_graph_id.npy'.format(dir))
        if self.mode == 'test':
            with open('{}/ppi/test_graph.json'.format(dir)) as jsonfile:
                g_data = json.load(jsonfile)
            self.labels = np.load('{}/ppi/test_labels.npy'.format(dir))
            self.features = np.load('{}/ppi/test_feats.npy'.format(dir))
            self.graph = DGLGraph(nx.DiGraph(json_graph.node_link_graph(g_data)))
            self.graph_id = np.load('{}/ppi/test_graph_id.npy'.format(dir))

    def _preprocess(self):
        if self.mode == 'train':
            self.train_mask_list = []
            self.train_graphs = []
            self.train_labels = []
            for train_graph_id in range(1, 21):
                train_graph_mask = np.where(self.graph_id == train_graph_id)[0]
                self.train_mask_list.append(train_graph_mask)
                self.train_graphs.append(self.graph.subgraph(train_graph_mask))
                self.train_labels.append(self.labels[train_graph_mask])
        if self.mode == 'valid':
            self.valid_mask_list = []
            self.valid_graphs = []
            self.valid_labels = []
            for valid_graph_id in range(21, 23):
                valid_graph_mask = np.where(self.graph_id == valid_graph_id)[0]
                self.valid_mask_list.append(valid_graph_mask)
                self.valid_graphs.append(self.graph.subgraph(valid_graph_mask))
                self.valid_labels.append(self.labels[valid_graph_mask])
        if self.mode == 'test':
            self.test_mask_list = []
            self.test_graphs = []
            self.test_labels = []
            for test_graph_id in range(23, 25):
                test_graph_mask = np.where(self.graph_id == test_graph_id)[0]
                self.test_mask_list.append(test_graph_mask)
                self.test_graphs.append(self.graph.subgraph(test_graph_mask))
                self.test_labels.append(self.labels[test_graph_mask])

    def __len__(self):
        """Return number of samples in this dataset."""
        if self.mode == 'train':
            return len(self.train_mask_list)
        if self.mode == 'valid':
            return len(self.valid_mask_list)
        if self.mode == 'test':
            return len(self.test_mask_list)

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
        if self.mode == 'train':
            g = self.train_graphs[item]
            g.ndata['feat'] = self.features[self.train_mask_list[item]]
            label =  self.train_labels[item]
        elif self.mode == 'valid':
            g = self.valid_graphs[item]
            g.ndata['feat'] = self.features[self.valid_mask_list[item]]
            label =  self.valid_labels[item]
        elif self.mode == 'test':
            g = self.test_graphs[item]
            g.ndata['feat'] = self.features[self.test_mask_list[item]]
            label =  self.test_labels[item]
        return g, label


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
        if self.mode == 'train':
            return self.train_graphs[item], self.features[self.train_mask_list[item]], self.train_labels[item]
        if self.mode == 'valid':
            return self.valid_graphs[item], self.features[self.valid_mask_list[item]], self.valid_labels[item]
        if self.mode == 'test':
            return self.test_graphs[item], self.features[self.test_mask_list[item]], self.test_labels[item]