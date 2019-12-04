import scipy.sparse as sp
import numpy as np
from dgl import graph_index, DGLGraph, transform
import os
from .utils import download, extract_archive, get_download_dir, _get_dgl_url

__all__=["AmazonCoBuy", "Coauthor", 'CoraFull']

def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


class GNNBenchmarkDataset(object):
    """Base Class for GNN Benchmark dataset from https://github.com/shchur/gnn-benchmark#datasets"""
    _url = {}

    def __init__(self, name):
        assert name.lower() in self._url, "Name not valid"
        self.dir = get_download_dir()
        self.path = os.path.join(
            self.dir, 'gnn_benckmark', self._url[name.lower()].split('/')[-1])
        download(self._url[name.lower()], path=self.path)
        g = self.load_npz(self.path)
        self.data = [g]

    @staticmethod
    def load_npz(file_name):
        with np.load(file_name) as loader:
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

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self.data[0]

    def __len__(self):
        return len(self.data)


class CoraFull(GNNBenchmarkDataset):
    r"""
    Extended Cora dataset from `Deep Gaussian Embedding of Graphs: 
    Unsupervised Inductive Learning via Ranking`. Nodes represent paper and edges represent citations.

    Reference: https://github.com/shchur/gnn-benchmark#datasets
    """
    _url = {"cora_full":'https://github.com/shchur/gnn-benchmark/raw/master/data/npz/cora_full.npz'}

    def __init__(self):
        super().__init__("cora_full")


class Coauthor(GNNBenchmarkDataset):
    r"""
    Coauthor CS and Coauthor Physics are co-authorship graphs based on the Microsoft Academic Graph
    from the KDD Cup 2016 challenge 3
    . Here, nodes are authors, that are connected by an edge if they
    co-authored a paper; node features represent paper keywords for each author’s papers, and class
    labels indicate most active fields of study for each author.

    Parameters
    ---------------
    name: str
      Name of the dataset, has to be 'cs' or 'physics'

    """
    _url = {
        'cs': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_cs.npz",
        'physics': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/ms_academic_phy.npz"
    }


class AmazonCoBuy(GNNBenchmarkDataset):
    r"""
    Amazon Computers and Amazon Photo are segments of the Amazon co-purchase graph [McAuley
    et al., 2015], where nodes represent goods, edges indicate that two goods are frequently bought
    together, node features are bag-of-words encoded product reviews, and class labels are given by the
    product category.

    Reference: https://github.com/shchur/gnn-benchmark#datasets

    Parameters
    ---------------
    name: str
      Name of the dataset, has to be 'computer' or 'photo'

    """
    _url = {
        'computers': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_computers.npz",
        'photo': "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/amazon_electronics_photo.npz"
    }
