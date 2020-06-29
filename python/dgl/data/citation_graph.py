"""Cora, citeseer, pubmed dataset.

(lingfan): following dataset loading and preprocessing code from tkipf/gcn
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""
from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys

from .dgl_dataset import DGLBuiltinDataset
from .utils import download, extract_archive, get_download_dir
from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor
from ..utils import retry_method_with_fix
from .. import backend as F
from ..graph import DGLGraph
from ..graph import batch as graph_batch
from ..convert import to_networkx

backend = os.environ.get('DGLBACKEND', 'pytorch')

_urls = {
    'cora_v2' : 'dataset/cora_v2.zip',
    'citeseer' : 'dataset/citeseer.zip',
    'pubmed' : 'dataset/pubmed.zip',
    'cora_binary' : 'dataset/cora_binary.zip',
}

def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)

class CitationGraphDataset(DGLBuiltinDataset):
    r"""The citation graph dataset, including cora, citeseer and pubmeb.
    Nodes mean authors and edges mean citation relationships.

    Parameters
    -----------
    name: str
      name can be 'cora', 'citeseer' or 'pubmed'.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        assert name.lower() in ['cora', 'citeseer', 'pubmed']

        # Previously we use the pre-processing in pygcn (https://github.com/tkipf/pygcn)
        # for Cora, which is slightly different from the one used in the GCN paper
        if name.lower() == 'cora':
            name = 'cora_v2'

        url = _get_dgl_url(_urls[name])
        super(CitationGraphDataset, self).__init__(name,
                                                   url=url,
                                                   raw_dir=raw_dir,
                                                   force_reload=force_reload,
                                                   verbose=verbose)

    def process(self, root_path):
        """Loads input data from gcn/data directory

        ind.name.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
        ind.name.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
        ind.name.allx => the feature vectors of both labeled and unlabeled training instances
            (a superset of ind.name.x) as scipy.sparse.csr.csr_matrix object;
        ind.name.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
        ind.name.ty => the one-hot labels of the test instances as numpy.ndarray object;
        ind.name.ally => the labels for instances in ind.name.allx as numpy.ndarray object;
        ind.name.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
            object;
        ind.name.test.index => the indices of test instances in graph, for the inductive setting as list object.

        All objects above must be saved using python pickle module.

        :param name: Dataset name
        :return: All data input files loaded (as well the training/test data).
        """
        root = root_path
        objnames = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(objnames)):
            with open("{}/ind.{}.{}".format(root, self.name, objnames[i]), 'rb') as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file("{}/ind.{}.test.index".format(root, self.name))
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]
        graph = nx.DiGraph(nx.from_dict_of_lists(graph))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        self._graph = graph
        g = DGLGraph(graph)

        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)
        g.ndata['label'] = F.tensor(labels)
        g.ndata['feat'] = F.tensor(_preprocess_features(features), dtype=F.data_type_dict['float32'])
        self._num_labels = onehot_labels.shape[1]
        self._g = g

        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self.g.number_of_nodes()))
            print('  NumEdges: {}'.format(self.g.number_of_edges()))
            print('  NumFeats: {}'.format(self.g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_labels))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['test_mask']).shape[0]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.g

    def __len__(self):
        return 1

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self.g)
        save_info(str(info_path), {'num_labels': self.num_labels})
        if self.verbose:
            print('Done saving data into cached files.')

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        if self.verbose:
            print('Done loading data into cached files.')
        self._g = graphs[0]
        self._graph = to_networkx(self._g)
        self._g.readonly(False)
        self._num_labels = info['num_labels']
        self._g.ndata['train_mask'] = generate_mask_tensor(self._g.ndata['train_mask'].numpy())
        self._g.ndata['val_mask'] = generate_mask_tensor(self._g.ndata['val_mask'].numpy())
        self._g.ndata['test_mask'] = generate_mask_tensor(self._g.ndata['test_mask'].numpy())

        if self.verbose:
            print('  NumNodes: {}'.format(self.g.number_of_nodes()))
            print('  NumEdges: {}'.format(self.g.number_of_edges()))
            print('  NumFeats: {}'.format(self.g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_labels))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self.g.ndata['test_mask']).shape[0]))

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def g(self):
        return self._g

    @property
    def num_labels(self):
        return self._num_labels

    """ Citation graph is used in many examples
        We preserve these properties for compatability.
    """
    @property
    def graph(self):
        return self._graph

    @property
    def train_mask(self):
        return self.g.ndata['train_mask']

    @property
    def val_mask(self):
        return self.g.ndata['val_mask']

    @property
    def test_mask(self):
        return self.g.ndata['test_mask']

    @property
    def labels(self):
        return self.g.ndata['label']

    @property
    def features(self):
        return self.g.ndata['feat']

class CoraGraphDataset(CitationGraphDataset):
    r""" Cora citation network dataset.
    
    Nodes mean paper and edges mean citation 
    relationships. Each node has a predefined 
    feature with 1433 dimensions. The dataset is 
    designed for the node classification task. 
    The task is to predict the category of 
    certain paper.

    Statistics
    ===
    Nodes: 2708
    Edges: 10556
    Number of Classes: 7
    Label Split: Train: 140 ,Valid: 500, Test: 1000

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    CoraDataset object with two properties:
        graph: A Homogeneous graph containing the 
            graph structure, node features and labels.
            - ndata['train_mask']： mask for training node set
            - ndata['val_mask']: mask for validation node set
            - ndata['test_mask']: mask for test node set
            - ndata['feat']: node feature
        num_of_class: number of paper categories for 
            the classification task.
    
    Examples
    ===
    
    >>> dataset = CoraDataset()
    >>> g = dataset.graph
    >>> num_class = g.num_of_class
    >>>
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>> 
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    >>>
    >>> # get labels
    >>> label = g.ndata['label']
    >>>
    >>> # Train, Validation and Test
    
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        name = 'cora'

        super(CoraGraphDataset, self).__init__(name, raw_dir, force_reload, verbose)

class CiteseerGraphDataset(CitationGraphDataset):
    r""" Citeseer citation network dataset.
    
    Nodes mean scientific publications and edges 
    mean citation relationships. Each node has a 
    predefined feature with 3703 dimensions. The 
    dataset is designed for the node classification 
    task. The task is to predict the category of 
    certain publication.

    Statistics
    ===
    Nodes: 3327
    Edges: 9228
    Number of Classes: 6
    Label Split: Train: 120 ,Valid: 500, Test: 1000

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    CiteseerDataset object with two properties:
        graph: A Homogeneous graph containing the 
            graph structure, node features and labels.
            - ndata['train_mask']： mask for training node set
            - ndata['val_mask']: mask for validation node set
            - ndata['test_mask']: mask for test node set
            - ndata['feat']: node feature
        num_of_class: number of publication categories 
            for the classification task.
    
    Examples
    ===
    
    >>> dataset = CiteseerDataset()
    >>> g = dataset.graph
    >>> num_class = g.num_of_class
    >>>
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>> 
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    >>>
    >>> # get labels
    >>> label = g.ndata['label']
    >>>
    >>> # Train, Validation and Test
    
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        name = 'citeseer'

        super(CiteseerGraphDataset, self).__init__(name, raw_dir, force_reload, verbose)

class PubmedGraphDataset(CitationGraphDataset):
    r""" Pubmed citation network dataset.
    
    Nodes mean scientific publications and edges 
    mean citation relationships. Each node has a 
    predefined feature with 500 dimensions. The 
    dataset is designed for the node classification 
    task. The task is to predict the category of 
    certain publication.

    Statistics
    ===
    Nodes: 19717
    Edges: 88651
    Number of Classes: 3
    Label Split: Train: 60 ,Valid: 500, Test: 1000

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    PubmedDataset object with two properties:
        graph: A Homogeneous graph containing the 
            graph structure, node features and labels.
            - ndata['train_mask']： mask for training node set
            - ndata['val_mask']: mask for validation node set
            - ndata['test_mask']: mask for test node set
            - ndata['feat']: node feature
        num_of_class: number of publication categories 
            for the classification task.
    
    Examples
    ===
    
    >>> dataset = PubmedDataset()
    >>> g = dataset.graph
    >>> num_class = g.num_of_class
    >>>
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>> 
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    >>>
    >>> # get labels
    >>> label = g.ndata['label']
    >>>
    >>> # Train, Validation and Test
    
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        name = 'pubmed'

        super(PubmedGraphDataset, self).__init__(name, raw_dir, force_reload, verbose)


def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())

def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask

"""Get CoraGraphDataset

Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    CoraDataset object
"""
def load_cora(raw_dir=None, force_reload=False, verbose=True):
    data = CoraGraphDataset(raw_dir, force_reload, verbose)
    return data

"""Get CiteseerDataset

Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    CiteseerDataset object
"""
def load_citeseer(raw_dir=None, force_reload=False, verbose=True):
    data = CiteseerGraphDataset(raw_dir, force_reload, verbose)
    return data

"""Get PubmedDataset

Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    
    Returns
    ===
    PubmedDataset object
"""
def load_pubmed(raw_dir=None, force_reload=False, verbose=True):
    data = PubmedGraphDataset(raw_dir, force_reload, verbose)
    return data

class CoraBinary(DGLBuiltinDataset):
    """A mini-dataset for binary classification task using Cora.

    After loaded, it has following members:

    graphs : list of :class:`~dgl.DGLGraph`
    pmpds : list of :class:`scipy.sparse.coo_matrix`
    labels : list of :class:`numpy.ndarray`

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    ===
    CoraBinary dataset
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=True):
        name = 'cora_binary'
        url = _get_dgl_url(_urls[name])
        super(CoraBinary, self).__init__(name,
                                         url=url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self, root_path)
        root = root_path
        # load graphs
        self.graphs = []
        with open("{}/graphs.txt".format(root), 'r') as f:
            elist = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(elist) != 0:
                        self.graphs.append(DGLGraph(elist))
                    elist = []
                else:
                    u, v = line.strip().split(' ')
                    elist.append((int(u), int(v)))
            if len(elist) != 0:
                self.graphs.append(DGLGraph(elist))
        with open("{}/pmpds.pkl".format(root), 'rb') as f:
            self.pmpds = _pickle_load(f)
        self.labels = []
        with open("{}/labels.txt".format(root), 'r') as f:
            cur = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(cur) != 0:
                        self.labels.append(np.asarray(cur))
                    cur = []
                else:
                    cur.append(int(line.strip()))
            if len(cur) != 0:
                self.labels.append(np.asarray(cur))
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        if os.path.exists(graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        save_graphs(str(graph_path), self.graphs, self.labels)
        if self.verbose:
            print('Done saving data into cached files.')

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        self.graphs, self.labels = load_graphs(str(graph_path))

        # load pmpds under self.raw_path
        with open("{}/pmpds.pkl".format(self.raw_path), 'rb') as f:
            self.pmpds = _pickle_load(f)
        if self.verbose:
            print('Done loading data into cached files.')
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        return (self.graphs[i], self.pmpds[i], self.labels[i])

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @staticmethod
    def collate_fn(batch):
        graphs, pmpds, labels = zip(*batch)
        batched_graphs = graph_batch(graphs)
        batched_pmpds = sp.block_diag(pmpds)
        batched_labels = np.concatenate(labels, axis=0)
        return batched_graphs, batched_pmpds, batched_labels

def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.asarray(list(map(classes_dict.get, labels)),
                               dtype=np.int32)
    return labels_onehot
