import scipy.sparse as sp
import numpy as np
import os

from .dgl_dataset import DGLDataset
from .utils import download, save_graphs, load_graphs, get_download_dir, makedirs
from ..graph import DGLGraph
from ..base import dgl_warning

__all__ = ["AmazonCoBuyDataset", "CoauthorDataset", "CoraFullDataset", "AmazonCoBuy", "Coauthor", "CoraFull"]


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


class GNNBenchmarkDataset(DGLDataset):
    r"""Base Class for GNN Benchmark dataset from https://github.com/shchur/gnn-benchmark#datasets"""
    _url = {}

    def __init__(self, **kwargs):
        name = kwargs.get('name', '').lower()
        assert name in self._url, "Dataset name not valid"
        url = self._url[name]
        raw_dir = kwargs.get('raw_dir', None)
        if raw_dir is None:
            raw_dir = os.path.join(get_download_dir(), 'gnn_benchmark')
        else:
            raw_dir = os.path.join(raw_dir, 'gnn_benchmark')
        kwargs.update(name=name, url=url, raw_dir=raw_dir)
        super(GNNBenchmarkDataset, self).__init__(**kwargs)

    def process(self, root_path):
        g = self._load_npz(root_path + '.npz')
        self.graph = g
        self.data = [g]

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        if os.path.exists(graph_path):
            return True
        return False

    def save(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        save_graphs(graph_path, self.graph)

    def load(self):
        graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
        graphs, _ = load_graphs(graph_path)
        self.graph = graphs[0]
        self.data = [graphs[0]]

    def download(self):
        r"""Automatically download npz data."""
        file_path = os.path.join(self.raw_dir, self.name + '.npz')
        download(self.url, path=file_path)

    def _load_npz(self, file_name):
        with np.load(file_name, allow_pickle=True) as loader:
            loader = dict(loader)
            num_nodes = loader['adj_shape'][0]
            adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                    shape=loader['adj_shape']).tocoo()

            if 'attr_data' in loader:
                # Attributes are stored as a sparse CSR matrix
                attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                            shape=loader['attr_shape']).todense()
            elif 'attr_matrix' in loader:
                # Attributes are stored as a (dense) np.ndarray
                attr_matrix = loader['attr_matrix']
            else:
                attr_matrix = None

            if 'labels_data' in loader:
                # Labels are stored as a CSR matrix
                labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                    shape=loader['labels_shape']).todense()
            elif 'labels' in loader:
                # Labels are stored as a numpy array
                labels = loader['labels']
            else:
                labels = None
        g = DGLGraph()
        g.add_nodes(num_nodes)
        g.add_edges(adj_matrix.row, adj_matrix.col)
        g.add_edges(adj_matrix.col, adj_matrix.row)
        g.ndata['feat'] = attr_matrix
        g.ndata['label'] = labels
        return g

    @property
    def num_classes(self):
        """Number of classes."""
        raise NotImplementedError

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        return 1


class CoraFullDataset(GNNBenchmarkDataset):
    r"""CORA-Full dataset for node classification task

    Extended Cora dataset from `Deep Gaussian Embedding of Graphs:
    Unsupervised Inductive Learning via Ranking`.
    Nodes represent paper and edges represent citations.
    Reference: https://github.com/shchur/gnn-benchmark#datasets

    Statistics
    ===
    Nodes: 19,793
    Edges: 130,622
    Number of Classes: 70
    Node feature size: 8,710

    Returns
    ===
    CoraFullDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
        num_classes: number of node classes

    Examples
    ===
    >>> data = CoraFullDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = {"cora_full": 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora_full.npz'}

    def __init__(self, **kwargs):
        super(CoraFullDataset, self).__init__(name="cora_full", **kwargs)

    @property
    def num_classes(self):
        """Number of classes."""
        return 70


class CoauthorDataset(GNNBenchmarkDataset):
    r"""Coauthor dataset for node classification

    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge. Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Parameters
    ---------------
    name: str
      Name of the dataset, has to be 'cs' or 'physics'

    Statistics for 'cs' part of the dataset
    ===
    Nodes: 18,333
    Edges: 327,576
    Number of classes: 15
    Node feature size: 6,805

    Statistics for 'physics' part of the dataset
    ===
    Nodes: 34,493
    Edges: 991,848
    Number of classes: 5
    Node feature size: 8,415

    Returns
    ===
    CoauthorDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
        num_classes: number of node classes

    Examples
    ===
    >>> data = CoauthorDataset('cs')
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = {
        'cs': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz",
        'physics': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_phy.npz"
    }

    def __init__(self, name, **kwargs):
        kwargs.update(name=name)
        super(CoauthorDataset, self).__init__(**kwargs)

    @property
    def num_classes(self):
        """Number of classes."""
        if self.name == 'cs':
            return 15
        elif self.name == 'physics':
            return 5
        return 0


class AmazonCoBuyDataset(GNNBenchmarkDataset):
    r"""AmazonCoBuy dataset for node classification task.

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    Parameters
    ---------------
    name: str
      Name of the dataset, has to be 'computers' or 'photo'

    Statistics for 'computers' part of the dataset
    ===
    Nodes: 13,752
    Edges: 574,418
    Number of classes: 5
    Node feature size: 767

    Statistics for 'photo' part of the dataset
    ===
    Nodes: 7,650
    Edges: 287,326
    Number of classes: 5
    Node feature size: 745

    Returns
    ===
    AmazonCoBuyDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
        num_classes: number of node classes

    Examples
    ===
    >>> data = AmazonCoBuyDataset('computers')
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = {
        'computers': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz",
        'photo': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz"
    }

    def __init__(self, name, **kwargs):
        kwargs.update(name=name)
        super(AmazonCoBuyDataset, self).__init__(**kwargs)

    @property
    def num_classes(self):
        """Number of classes."""
        return 5


class CoraFull(CoraFullDataset):
    def __init__(self, **kwargs):
        dgl_warning('CoraFull is deprecated, use CoraFullDataset instead.',
                    DeprecationWarning, stacklevel=2)
        super(CoraFull, self).__init__(**kwargs)


class AmazonCoBuy(AmazonCoBuyDataset):
    def __init__(self, name, **kwargs):
        dgl_warning('AmazonCoBuy is deprecated, use AmazonCoBuyDataset instead.',
                    DeprecationWarning, stacklevel=2)
        kwargs.update(name=name)
        super(AmazonCoBuy, self).__init__(**kwargs)


class Coauthor(CoauthorDataset):
    def __init__(self, name, **kwargs):
        dgl_warning('Coauthor is deprecated, use CoauthorDataset instead.',
                    DeprecationWarning, stacklevel=2)
        kwargs.update(name=name)
        super(Coauthor, self).__init__(**kwargs)
