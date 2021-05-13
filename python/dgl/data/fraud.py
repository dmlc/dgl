"""Fraud Dataset
"""
import torch
import os
from scipy import io
from sklearn.model_selection import train_test_split
import numpy as np

from .utils import save_graphs, save_info, load_graphs, load_info, _get_dgl_url
from ..convert import from_scipy
from .dgl_dataset import DGLBuiltinDataset

class FraudDataset(DGLBuiltinDataset):
    """Fraud node prediction dataset.

    The dataset includes two homogeneous multi-relational graphs extracted from 
    Yelp and Amazon where nodes represent fraudulent reviews or fraudulent reviewers.
    It was first proposed in a CIKM'20 paper <https://arxiv.org/pdf/2008.08692.pdf> and 
    has been used by a recent WWW'21 paper <https://ponderly.github.io/pub/PCGNN_WWW2021.pdf> 
    as a benchmark. Another paper also takes the dataset as an example to study the 
    non-homophilous graphs. This dataset is built upon industrial data and has rich 
    relational information and unique properties like class-imbalance and feature 
    inconsistency, which makes the dataset be a good instance to investigate how GNNs 
    perform on real-world noisy graphs.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    """
    file_urls = {
        'yelp' : 'dataset/FraudYelp.zip',
        'amazon' : 'dataset/FraudAmazon.zip'
    }
    relations = {
        'yelp' : ['net_rsr', 'net_rtr', 'net_rur'],
        'amazon' : ['homo', 'net_upu', 'net_usu', 'net_uvu']
    }
    file_names = {
        'yelp' : 'YelpChi.mat',
        'amazon' : 'Amazon.mat'
    }
    
    def __init__(self, name, raw_dir=None, random_seed=2, train_size=0.7, val_size=0.1):
        url = _get_dgl_url(self.file_urls[name])
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size
        super(FraudDataset, self).__init__(name=name,
                                           url=url,
                                           raw_dir=raw_dir)

    def process(self):
        # process raw data to graphs, labels, splitting masks
        file_path = os.path.join(self.raw_path, self.file_names[self.name])
        
        data = io.loadmat(file_path)
        node_features = torch.from_numpy(data['features'].todense())
        node_labels = torch.from_numpy(data['label'])
        
        N = node_labels.shape[0]
        
        graphs = []
        for relation in self.relations[self.name]:
            g = from_scipy(data[relation])
            graphs.append(g)
        self.graphs = graphs
        
        self.feature = node_features
        self.label = node_labels
        self._random_split(self.feature, self.label, self.seed, self.train_size, self.val_size)

    def __getitem__(self, idx):
        # get one example by index
        return self.graphs[idx]

    def __len__(self):
        # number of data examples
        return len(self.graphs)

    def save(self):
        # save processed data to directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + '_dgl_graph.pkl')
        save_graphs(str(graph_path), self.graphs)
        save_info(info_path, {'label': self.label, 'feature': self.feature})

    def load(self):
        # load processed data from directory `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + '_dgl_graph.pkl')

        graphs, _ = load_graphs(str(graph_path))
        info = load_info(str(info_path))
        self.graphs = graphs
        self.label = info['label']
        self.feature = info['feature']
        self._random_split(self.feature, self.label, self.seed, self.train_size, self.val_size)

    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path = os.path.join(self.save_path, self.name + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.name + '_dgl_graph.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

    def _random_split(self, x, node_labels, seed=2, train_size=0.7, val_size=0.1):
        N = x.shape[0]
        node_labels = node_labels.transpose(0, 1)
        if self.name == 'yelp':
            index = list(range(N))
            train_idx, test_idx, _, y = train_test_split(index, node_labels, stratify=node_labels, train_size=train_size,
                                                                    random_state=seed, shuffle=True)
        elif self.name == 'amazon':
            # 0-3304 are unlabeled nodes
            index = list(range(3305, N))
            train_idx, test_idx, _, y = train_test_split(index, node_labels[3305:], stratify=node_labels[3305:],
                                                                    test_size=train_size, random_state=seed, shuffle=True)
        val_idx, test_idx, _, _ = train_test_split(test_idx, y, stratify=y, train_size=val_size/(1-train_size),
                                                    random_state=seed, shuffle=True)
        train_mask = torch.zeros(N, dtype=torch.bool)
        val_mask = torch.zeros(N, dtype=torch.bool)
        test_mask = torch.zeros(N, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
    
class FraudYelpDataset(FraudDataset):
    def __init__(self, raw_dir=None, random_seed=2, train_size=0.7, val_size=0.1):
        super(FraudYelpDataset, self).__init__(name='yelp',
                                           raw_dir=raw_dir,
                                           random_seed=random_seed,
                                           train_size=train_size,
                                           val_size=val_size)
                                           
class FraudAmazonDataset(FraudDataset):
    def __init__(self, raw_dir=None, random_seed=2, train_size=0.7, val_size=0.1):
        super(FraudAmazonDataset, self).__init__(name='amazon',
                                           raw_dir=raw_dir,
                                           random_seed=random_seed,
                                           train_size=train_size,
                                           val_size=val_size)
