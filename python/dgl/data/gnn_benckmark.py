"""GNN Benchmark datasets for node classification."""
import scipy.sparse as sp
import numpy as np
import os

from .dgl_dataset import DGLBuiltinDataset
from .utils import save_graphs, load_graphs, _get_dgl_url, deprecate_property, deprecate_class
from ..graph import DGLGraph

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
    r"""Base Class for GNN Benchmark dataset

    Reference: https://github.com/shchur/gnn-benchmark#datasets
    """
    def __init__(self, name, raw_dir=None, force_reload=False, verbose=False):
        _url = _get_dgl_url('dataset/' + name + '.zip')
        super(GNNBenchmarkDataset, self).__init__(name=name,
                                                  url=_url,
                                                  raw_dir=raw_dir,
                                                  force_reload=force_reload,
                                                  verbose=verbose)

    def process(self, root_path):
        npz_path = os.path.join(root_path, self.name + '.npz')
        g = self._load_npz(npz_path)
        self._graph = g
        self._data = [g]
        self._print_info()

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
        self._graph = graphs[0]
        self._data = [graphs[0]]
        self._print_info()

    def _print_info(self):
        if self.verbose:
            print('  NumNodes: {}'.format(self.graph.number_of_nodes()))
            print('  NumEdges: {}'.format(self.graph.number_of_edges()))
            print('  NumFeats: {}'.format(self.graph.ndata['feat'].shape[-1]))
            print('  NumbClasses: {}'.format(self.num_classes))

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

    @property
    def graph(self):
        return self._graph

    @property
    def data(self):
        deprecate_property('.data', '.graph')
        return self._data

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
    ----------
    Nodes: 19,793
    Edges: 130,622
    Number of Classes: 70
    Node feature size: 8,710

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    CoraFullDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    --------
    >>> data = CoraFullDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(CoraFullDataset, self).__init__(name="cora_full",
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

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
    ----------
    Nodes: 18,333
    Edges: 327,576
    Number of classes: 15
    Node feature size: 6,805

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    CoauthorCSDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    --------
    >>> data = CoauthorCSDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(CoauthorCSDataset, self).__init__(name='coauthor_cs',
                                                raw_dir=raw_dir,
                                                force_reload=force_reload,
                                                verbose=verbose)

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
    ----------
    Nodes: 34,493
    Edges: 991,848
    Number of classes: 5
    Node feature size: 8,415

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    CoauthorPhysicsDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    --------
    >>> data = CoauthorPhysicsDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(CoauthorPhysicsDataset, self).__init__(name='coauthor_physics',
                                                     raw_dir=raw_dir,
                                                     force_reload=force_reload,
                                                     verbose=verbose)

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
    ----------
    Nodes: 13,752
    Edges: 574,418
    Number of classes: 5
    Node feature size: 767

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    AmazonCoBuyComputerDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    --------
    >>> data = AmazonCoBuyComputerDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(AmazonCoBuyComputerDataset, self).__init__(name='amazon_co_buy_computer',
                                                         raw_dir=raw_dir,
                                                         force_reload=force_reload,
                                                         verbose=verbose)

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
    ----------
    Nodes: 7,650
    Edges: 287,326
    Number of classes: 5
    Node feature size: 745

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    -------
    AmazonCoBuyDataset object with two properties:
        graph: A homogeneous graph contains the graph structure, node features and node labels
            - ndata['feat']: tensor of the node features
            - ndata['label']: tensor of the node label
        num_classes: number of node classes

    Examples
    --------
    >>> data = AmazonCoBuyPhotoDataset()
    >>> g = data.graph
    >>> num_class = data.num_classes
    >>> feat = g.ndata['feat']  # get node feature
    >>> label = g.ndata['label']  # get node labels
    """
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        super(AmazonCoBuyPhotoDataset, self).__init__(name='amazon_co_buy_photo',
                                                      raw_dir=raw_dir,
                                                      force_reload=force_reload,
                                                      verbose=verbose)

    @property
    def num_classes(self):
        """Number of classes."""
        return 5


class CoraFull(CoraFullDataset):
    def __init__(self, **kwargs):
        deprecate_class('CoraFull', 'CoraFullDataset')
        super(CoraFull, self).__init__(**kwargs)


def AmazonCoBuy(name):
    if name == 'computers':
        deprecate_class('AmazonCoBuy', 'AmazonCoBuyComputerDataset')
        return AmazonCoBuyComputerDataset()
    elif name == 'photo':
        deprecate_class('AmazonCoBuy', 'AmazonCoBuyPhotoDataset')
        return AmazonCoBuyPhotoDataset()
    else:
        raise ValueError('Dataset name should be "computers" or "photo".')


def Coauthor(name):
    if name == 'cs':
        deprecate_class('Coauthor', 'CoauthorCSDataset')
        return CoauthorCSDataset()
    elif name == 'physics':
        deprecate_class('Coauthor', 'CoauthorPhysicsDataset')
        return CoauthorPhysicsDataset()
    else:
        raise ValueError('Dataset name should be "cs" or "physics".')
