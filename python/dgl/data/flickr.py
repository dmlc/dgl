"""Flickr Dataset"""
import os
import json
import numpy as np
import scipy.sparse as sp
from .. import backend as F
from ..convert import graph
from ..transforms import reorder_graph
from .dgl_dataset import DGLBuiltinDataset
from .utils import generate_mask_tensor, load_graphs, save_graphs, _get_dgl_url


class FlickrDataset(DGLBuiltinDataset):
    r"""Flickr dataset for node classification from `GraphSAINT: Graph Sampling Based Inductive
    Learning Method <https://arxiv.org/abs/1907.04931>`_

    The task of this dataset is categorizing types of images based on the descriptions and common
    properties of online images.

    Flickr dataset statistics:

    - Nodes: 89,250
    - Edges: 899,756
    - Number of classes: 7
    - Node feature size: 500

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset.
        Default: False
    verbose : bool
        Whether to print out progress information.
        Default: False
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    num_classes : int
        Number of node classes

    Examples
    --------
    >>> dataset = FlickrDataset()
    >>> dataset.num_classes
    7
    >>> g = dataset[0]
    >>> # get node feature
    >>> feat = g.ndata['feat']
    >>> # get node labels
    >>> labels = g.ndata['label']
    >>> # get data split
    >>> train_mask = g.ndata['train_mask']
    >>> val_mask = g.ndata['val_mask']
    >>> test_mask = g.ndata['test_mask']
    """

    def __init__(self, raw_dir=None, force_reload=False, verbose=False, transform=None):
        _url = _get_dgl_url('dataset/flickr.zip')
        super(FlickrDataset, self).__init__(name='flickr',
                                            raw_dir=raw_dir,
                                            url=_url,
                                            force_reload=force_reload,
                                            verbose=verbose,
                                            transform=transform)

    def process(self):
        """process raw data to graph, labels and masks"""
        f = np.load(os.path.join(self.raw_path, 'adj_full.npz'))
        adj = sp.csr_matrix((f['data'], f['indices'], f['indptr']), f['shape'])
        adj = adj.tocoo()
        edges = np.vstack((adj.row, adj.col)).transpose()

        features = np.load(os.path.join(self.raw_path, 'feats.npy'))
        features = F.tensor(features, dtype=F.float32)

        y = [-1] * features.shape[0]
        with open(os.path.join(self.raw_path, 'class_map.json')) as f:
            class_map = json.load(f)
            for key, item in class_map.items():
                y[int(key)] = item
        labels = F.tensor(np.array(y), dtype=F.int64)

        with open(os.path.join(self.raw_path, 'role.json')) as f:
            role = json.load(f)

        train_mask = np.zeros(features.shape[0], dtype=bool)
        train_mask[role['tr']] = True

        val_mask = np.zeros(features.shape[0], dtype=bool)
        val_mask[role['va']] = True

        test_mask = np.zeros(features.shape[0], dtype=bool)
        test_mask[role['te']] = True

        g = graph((edges[:, 0], edges[:, 1]))

        g.ndata['feat'] = features
        g.ndata['label'] = labels
        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)

        g = reorder_graph(g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)

        self._graph = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        return os.path.exists(graph_path)

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self._graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        g, _ = load_graphs(graph_path)
        self._graph = g[0]

    @property
    def num_classes(self):
        return 7

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return 1

    def __getitem__(self, idx):
        r""" Get graph object

        Parameters
        ----------
        idx : int
            Item index, FlickrDataset has only one graph object

        Returns
        -------
        :class:`dgl.DGLGraph`

            The graph contains:

            - ``ndata['label']``: node label
            - ``ndata['feat']``: node feature
            - ``ndata['train_mask']``： mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']:`` mask for test node set

        """
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._graph
        else:
            return self._transform(self._graph)