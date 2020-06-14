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
from .utils import download, extract_archive, get_download_dir, _get_dgl_url
from .utils import save_graphs, load_graphs, save_info, load_info
from ..utils import retry_method_with_fix
from ..graph import DGLGraph
from ..graph import batch as graph_batch

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
    """
    def __init__(self, name):
        assert name.lower() in ['cora', 'citeseer', 'pubmed']

        # Previously we use the pre-processing in pygcn (https://github.com/tkipf/pygcn)
        # for Cora, which is slightly different from the one used in the GCN paper
        if name.lower() == 'cora':
            name = 'cora_v2'

        url = _get_dgl_url(_urls[name])
        super(CitationGraphDataset, self).__init__(name, url)

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

        self._features = _preprocess_features(features)
        self._labels = labels
        self._num_labels = onehot_labels.shape[1]
        self._train_mask = train_mask
        self._val_mask = val_mask
        self._test_mask = test_mask
        self._g = DGLGraph(graph)

        print('Finished data loading and preprocessing.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.node_features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        g = self.graph
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        return g

    def __len__(self):
        return 1

    def has_cache(self):
        if os.path.exists(os.path.join(self.raw_path, 'graph.bin')) and \
            os.path.exists(os.path.join(self.raw_path, 'dgl_info.pickle')):
            return True

        return False

    def load(self):
        g = load_graphs(os.path.join(self.raw_path, 'graph.bin'))[0]
        info = load_info(os.path.join(self.raw_path, 'dgl_info.pickle'))
        self._features = g.pop('feat')
        self._labels = g.pop('label')
        self._num_labels = info['num_labels']
        self._train_mask = g.pop('train_mask')
        self._val_mask = g.pop('val_mask')
        self._test_mask = g.pop('test_mask')
        self._g = g

        print('Finished loading data from cached files.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.node_features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def save(self):
        g = self.graph
        g.ndata['train_mask'] = self.train_mask
        g.ndata['val_mask'] = self.val_mask
        g.ndata['test_mask'] = self.test_mask
        g.ndata['label'] = self.labels
        g.ndata['feat'] = self.features
        graph_path = os.path.join(self.raw_path, 'dgl_graph.bin')
        info_path = os.path.join(self.raw_path, 'dgl_info.pickle')
        info = {'num_labels' : self.num_labels}

        # save data in files
        save_graphs(graph_path, self.graph)
        save_info(info_path, info)
        print('Done saving data into cached files.')

    @property
    def graph(self):
        return self._g

    @property
    def train_mask(self):
        return self._train_mask

    @property
    def val_mask(self):
        return self._val_mask

    @property
    def test_mask(self):
        return self._test_mask

    @property
    def labels(self):
        return self._labels

    @property
    def num_labels(self):
        return self._num_labels

    @property
    def node_features(self):
        return self._features

    @property
    def features(self):
        """For backward compatability
        """
        return self.node_features

class CoraGraphDataset(CitationGraphDataset):
    def __init__(self):
        name = 'cora'

        super(CoraGraphDataset, self).__init__(name)

class CiteseerGraphDataset(CitationGraphDataset):
    def __init__(self):
        name = 'citeseer'

        super(CiteseerGraphDataset, self).__init__(name)

class PubmedGraphDataset(CitationGraphDataset):
    def __init__(self):
        name = 'pubmed'

        super(PubmedGraphDataset, self).__init__(name)

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

def load_cora():
    data = CoraGraphDataset()
    return data

def load_citeseer():
    data = CiteseerGraphDataset()
    return data

def load_pubmed():
    data = PubmedGraphDataset()
    return data

class CoraBinary(DGLBuiltinDataset):
    """A mini-dataset for binary classification task using Cora.

    After loaded, it has following members:

    graphs : list of :class:`~dgl.DGLGraph`
    pmpds : list of :class:`scipy.sparse.coo_matrix`
    labels : list of :class:`numpy.ndarray`
    """
    def __init__(self):
        name = 'cora_binary'

        url = _get_dgl_url(_urls[name])
        super(CoraBinary, self).__init__(name, url)

    def process(self, root_path):
        root = root_path
        # load graphs
        self._graphs = []
        with open("{}/graphs.txt".format(root), 'r') as f:
            elist = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(elist) != 0:
                        self._graphs.append(DGLGraph(elist))
                    elist = []
                else:
                    u, v = line.strip().split(' ')
                    elist.append((int(u), int(v)))
            if len(elist) != 0:
                self._graphs.append(DGLGraph(elist))
        with open("{}/pmpds.pkl".format(root), 'rb') as f:
            self._pmpds = _pickle_load(f)
        self._labels = []
        with open("{}/labels.txt".format(root), 'r') as f:
            cur = []
            for line in f.readlines():
                if line.startswith('graph'):
                    if len(cur) != 0:
                        self._labels.append(np.asarray(cur))
                    cur = []
                else:
                    cur.append(int(line.strip()))
            if len(cur) != 0:
                self._labels.append(np.asarray(cur))
        # sanity check
        assert len(self.graph) == len(self.pmpds)
        assert len(self.graph) == len(self.labels)

    def save(self):
        graph_path = os.path.join(self.raw_path, 'dgl_graph.bin')
        info_path = os.path.join(self.raw_path, 'dgl_info.pickle')
        info = {'pmpds': self.pmpds,
                'labels': self.labels}

        #save data in files
        save_graphs(graph_path, self.graph)
        save_info(info_path, info)
        print('Done saving data into cached files.')

    def load(self):
        graphs = load_graphs(os.path.join(self.raw_path, 'graph.bin'))
        info = load_info(os.path.join(self.raw_path, 'dgl_info.pickle'))

        self._graphs = graphs
        self._pmpds = info['pmpds']
        self._labels = info['labels']
        print('Finished loading data from cached files.')

    @property
    def graph(self):
        return self._graphs

    @property
    def labels(self):
        return self._labels

    @property
    def pmpds(self):
        return self._pmpds

    def __len__(self):
        return len(self.graph)

    def __getitem__(self, i):
        return (self.graph[i], self.pmpds[i], self.labels[i])

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
