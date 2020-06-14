""" Synthetic Graph
"""

from __future__ import absolute_import

import numpy as np
import networkx as nx
import scipy.sparse as sp

class GCNSyntheticDataset(object):
    def __init__(self,
                 graph_generator,
                 num_feats=500,
                 num_classes=10,
                 train_ratio=1.,
                 val_ratio=0.,
                 test_ratio=0.,
                 seed=None):
        rng = np.random.RandomState(seed)
        # generate graph
        self.graph = graph_generator(seed)
        num_nodes = self.graph.number_of_nodes()

        # generate features
        #self.features = rng.randn(num_nodes, num_feats).astype(np.float32)
        self.features = np.zeros((num_nodes, num_feats), dtype=np.float32)

        # generate labels
        self.labels = rng.randint(num_classes, size=num_nodes)
        onehot_labels = np.zeros((num_nodes, num_classes), dtype=np.float32)
        onehot_labels[np.arange(num_nodes), self.labels] = 1.
        self.onehot_labels = onehot_labels
        self.num_labels = num_classes

        # generate masks
        ntrain = int(num_nodes * train_ratio)
        nval = int(num_nodes * val_ratio)
        ntest = int(num_nodes * test_ratio)
        mask_array = np.zeros((num_nodes,), dtype=np.int32)
        mask_array[0:ntrain] = 1
        mask_array[ntrain:ntrain+nval] = 2
        mask_array[ntrain+nval:ntrain+nval+ntest] = 3
        rng.shuffle(mask_array)
        self.train_mask = (mask_array == 1).astype(np.int32)
        self.val_mask = (mask_array == 2).astype(np.int32)
        self.test_mask = (mask_array == 3).astype(np.int32)

        print('Finished synthetic dataset generation.')
        print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
        print('  NumEdges: {}'.format(self.graph.number_of_edges()))
        print('  NumFeats: {}'.format(self.features.shape[1]))
        print('  NumClasses: {}'.format(self.num_labels))
        print('  NumTrainingSamples: {}'.format(len(np.nonzero(self.train_mask)[0])))
        print('  NumValidationSamples: {}'.format(len(np.nonzero(self.val_mask)[0])))
        print('  NumTestSamples: {}'.format(len(np.nonzero(self.test_mask)[0])))

    def __getitem__(self, idx):
        return self

    def __len__(self):
        return 1

def get_gnp_generator(args):
    n = args.syn_gnp_n
    p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p
    def _gen(seed):
        return nx.fast_gnp_random_graph(n, p, seed, True)
    return _gen

class ScipyGraph(object):
    """A simple graph object that uses scipy matrix."""
    def __init__(self, mat):
        self._mat = mat

    def get_graph(self):
        return self._mat

    def number_of_nodes(self):
        return self._mat.shape[0]

    def number_of_edges(self):
        return self._mat.getnnz()

def get_scipy_generator(args):
    n = args.syn_gnp_n
    p = (2 * np.log(n) / n) if args.syn_gnp_p == 0. else args.syn_gnp_p
    def _gen(seed):
        return ScipyGraph(sp.random(n, n, p, format='coo'))
    return _gen

def load_synthetic(args):
    ty = args.syn_type
    if ty == 'gnp':
        gen = get_gnp_generator(args)
    elif ty == 'scipy':
        gen = get_scipy_generator(args)
    else:
        raise ValueError('Unknown graph generator type: {}'.format(ty))
    return GCNSyntheticDataset(
            gen,
            args.syn_nfeats,
            args.syn_nclasses,
            args.syn_train_ratio,
            args.syn_val_ratio,
            args.syn_test_ratio,
            args.syn_seed)

def register_args(parser):
    # Args for synthetic graphs.
    parser.add_argument('--syn-type', type=str, default='gnp',
            help='Type of the synthetic graph generator')
    parser.add_argument('--syn-nfeats', type=int, default=500,
            help='Number of node features')
    parser.add_argument('--syn-nclasses', type=int, default=10,
            help='Number of output classes')
    parser.add_argument('--syn-train-ratio', type=float, default=.1,
            help='Ratio of training nodes')
    parser.add_argument('--syn-val-ratio', type=float, default=.2,
            help='Ratio of validation nodes')
    parser.add_argument('--syn-test-ratio', type=float, default=.5,
            help='Ratio of testing nodes')
    # Args for GNP generator
    parser.add_argument('--syn-gnp-n', type=int, default=1000,
            help='n in gnp random graph')
    parser.add_argument('--syn-gnp-p', type=float, default=0.0,
            help='p in gnp random graph')
    parser.add_argument('--syn-seed', type=int, default=42,
            help='random seed')
