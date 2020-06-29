import scipy.sparse as sp
import numpy as np
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import save_graphs, load_graphs, _get_dgl_url
from ..graph import DGLGraph
from ..base import dgl_warning

__all__ = ["AmazonCoBuyComputerDataset", "AmazonCoBuyPhotoDataset", "CoauthorPhysicsDataset", "CoauthorCSDataset",
           "CoraFullDataset", "AmazonCoBuy", "Coauthor", "CoraFull"]


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


class GNNBenchmarkDataset(DGLBuiltinDataset):
    r"""Base Class for GNN Benchmark dataset from https://github.com/shchur/gnn-benchmark#datasets"""
    _url = None

    def __init__(self, name, force_reload=False):
        _url = _get_dgl_url('dataset/' + name + '.zip')
        super(GNNBenchmarkDataset, self).__init__(name=name, url=_url, force_reload=force_reload)

    def process(self, root_path):
        npz_path = os.path.join(root_path, self.name + '.npz')
        g = self._load_npz(npz_path)
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
    r"""CORA-Full dataset for node classification task.

    Extended Cora dataset from `Deep Gaussian Embedding of Graphs:
    Unsupervised Inductive Learning via Ranking`.
    Nodes represent paper and edges represent citations.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ===
    Nodes: 19,793
    Edges: 130,622
    Number of Classes: 70
    Node feature size: 8,710

    Parameters
    ----------
    force_reload: bool
        Whether to reload the dataset. Default: False

    Returns
    ===
    CoraFullDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    ===
    >>> data = CoraFullDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = 'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora_full.npz'

    def __init__(self, force_reload=False):
        super(CoraFullDataset, self).__init__(name="cora_full", force_reload=force_reload)

    @property
    def num_classes(self):
        """Number of classes."""
        return 70


class CoauthorCSDataset(GNNBenchmarkDataset):
    r""" 'Computer Science (CS)' part of the Coauthor dataset for node classification task.

    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge. Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ===
    Nodes: 18,333
    Edges: 327,576
    Number of classes: 15
    Node feature size: 6,805

    Parameters
    ----------
    force_reload: bool
        Whether to reload the dataset. Default: False

    Returns
    ===
    CoauthorCSDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    ===
    >>> data = CoauthorCSDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz"

    def __init__(self, force_reload=False):
        super(CoauthorCSDataset, self).__init__(name='coauthor_cs', force_reload=force_reload)

    @property
    def num_classes(self):
        """Number of classes."""
        return 15


class CoauthorPhysicsDataset(GNNBenchmarkDataset):
    r""" 'Physics' part of the Coauthor dataset for node classification task.

    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge. Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ===
    Nodes: 34,493
    Edges: 991,848
    Number of classes: 5
    Node feature size: 8,415

    Parameters
    ----------
    force_reload: bool
        Whether to reload the dataset. Default: False

    Returns
    ===
    CoauthorPhysicsDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    ===
    >>> data = CoauthorPhysicsDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_phy.npz"

    def __init__(self, force_reload=False):
        super(CoauthorPhysicsDataset, self).__init__(name='coauthor_physics', force_reload=force_reload)

    @property
    def num_classes(self):
        """Number of classes."""
        return 5


class AmazonCoBuyComputerDataset(GNNBenchmarkDataset):
    r""" 'Computer' part of the AmazonCoBuy dataset for node classification task.

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ===
    Nodes: 13,752
    Edges: 574,418
    Number of classes: 5
    Node feature size: 767

    Parameters
    ----------
    force_reload: bool
        Whether to reload the dataset. Default: False

    Returns
    ===
    AmazonCoBuyComputerDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    ===
    >>> data = AmazonCoBuyComputerDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz"

    def __init__(self, force_reload=False):
        super(AmazonCoBuyComputerDataset, self).__init__(name='amazon_co_buy_computer', force_reload=force_reload)

    @property
    def num_classes(self):
        """Number of classes."""
        return 5


class AmazonCoBuyPhotoDataset(GNNBenchmarkDataset):
    r"""AmazonCoBuy dataset for node classification task.

    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley et al., 2015],
    where nodes represent goods, edges indicate that two goods are frequently bought together, node
    features are bag-of-words encoded product reviews, and class labels are given by the product category.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    Statistics
    ===
    Nodes: 7,650
    Edges: 287,326
    Number of classes: 5
    Node feature size: 745

    Parameters
    ----------
    force_reload: bool
        Whether to reload the dataset. Default: False

    Returns
    ===
    AmazonCoBuyDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    ===
    >>> data = AmazonCoBuyPhotoDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    _url = "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz"

    def __init__(self, force_reload=False):
        super(AmazonCoBuyPhotoDataset, self).__init__(name='amazon_co_buy_photo', force_reload=force_reload)

    @property
    def num_classes(self):
        """Number of classes."""
        return 5


class CoraFull(CoraFullDataset):
    def __init__(self, **kwargs):
        dgl_warning('CoraFull is deprecated, use CoraFullDataset instead.',
                    DeprecationWarning, stacklevel=2)
        super(CoraFull, self).__init__(**kwargs)


def AmazonCoBuy(name):
    dgl_warning('AmazonCoBuy is deprecated, use AmazonCoBuyComputerDataset or AmazonCoBuyPhotoDataset instead.',
                DeprecationWarning, stacklevel=2)
    if name == 'computers':
        return AmazonCoBuyComputerDataset()
    elif name == 'photo':
        return AmazonCoBuyPhotoDataset()
    else:
        raise ValueError('Dataset name should be "computers" or "photo".')


def Coauthor(name):
    dgl_warning('Coauthor is deprecated, use CoauthorCSDataset or CoauthorPhysicsDataset instead.',
                DeprecationWarning, stacklevel=2)
    if name == 'cs':
        return CoauthorCSDataset()
    elif name == 'physics':
        return CoauthorPhysicsDataset()
    else:
        raise ValueError('Dataset name should be "cs" or "physics".')
