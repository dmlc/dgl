"""PPI Dataset.
(zhang hao): Used for inductive learning.
"""

import numpy as np
from .utils import download, extract_archive, get_download_dir, _get_dgl_url
from sklearn.preprocessing import StandardScaler
from dgl import DGLGraph
import networkx as nx
from networkx.readwrite import json_graph
import json

_url = 'dataset/ppi.zip'

def load_ppi():
    """Loads input data
    ppi_G.json => the graph data used for training, test and validation as json format;
    ppi-feats.npy => the feature vectors of nodes as numpy.ndarry object, it's shape is [n, v],
    n is the number of nodes, v is the feature's dimension;
    ppi-class_map.json => the classes of the input nodes, the format of it is {"1":[0, 1, 0, 1]},
    "1" is the node index, [0, 1, 0, 1] is label;
    ppi-graph_id.npy => the element in it indicates which graph the nodes belong to,
    it is a one dimensional numpy.ndarray object and the length of it is equal the number of nodes,
    it's like [1, 1, 2, 1,,,23];

    """
    name = 'ppi'
    dir = get_download_dir()
    zip_file_path = '{}/{}.zip'.format(dir, name)
    download(_get_dgl_url(_url), path=zip_file_path)
    extract_archive(zip_file_path,
                    '{}/{}'.format(dir, name))
    print('Loading G...')
    with open('{}/ppi/ppi-G.json'.format(dir)) as jsonfile:
        g_data = json.load(jsonfile)

    with open('{}/ppi/ppi-class_map.json'.format(dir)) as jsonfile:
        class_map = json.load(jsonfile)
    features = np.load('{}/ppi/ppi-feats.npy'.format(dir))
    label_list = []
    for i in range(len(class_map)):
        label_list.append(np.expand_dims(np.array(class_map[str(i)]), axis=0))
    labels = np.concatenate(label_list)
    graph = DGLGraph(nx.DiGraph(json_graph.node_link_graph(g_data)))
    graph_id = np.load('{}/ppi/graph_id.npy'.format(dir))
    return (features, labels, graph, graph_id)

class PPIDataset(object):
    
    def __init__(self, mode, data):
        """Initialize the dataset.

        Paramters
        ---------
        mode : str
            ('train', 'valid', 'test').
        data : tuple()
            (features, labels, graph, graph_id)
          
        """ 
        self.mode = mode
        self.features, self.labels, self.graph, self.graph_id = data
        self._preprocess()
        self._normalize()
        
    def _preprocess(self):
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
                
    def _normalize(self):
        """
        Normalize the features
        """

        train_feats = self.features[np.concatenate(self.train_mask_list)]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        self.features = scaler.transform(self.features)
        
    def __len__(self):
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
        (dgl.DGLGraph, ndarray, ndarray)
            The graph, features and its label.
        """
        if self.mode == 'train':
            return self.train_graphs[item], self.features[self.train_mask_list[item]], self.train_labels[item]
        if self.mode == 'valid':
            return self.valid_graphs[item], self.features[self.valid_mask_list[item]], self.valid_labels[item]
        if self.mode == 'test':
            return self.test_graphs[item], self.features[self.test_mask_list[item]], self.test_labels[item]
