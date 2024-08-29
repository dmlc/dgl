"""Cora, citeseer, pubmed dataset.

(lingfan): following dataset loading and preprocessing code from tkipf/gcn
https://github.com/tkipf/gcn/blob/master/gcn/utils.py
"""

from __future__ import absolute_import

import os, sys
import pickle as pkl
import warnings

import networkx as nx

import numpy as np
import scipy.sparse as sp

from .. import backend as F, convert
from ..batch import batch as batch_graphs
from ..convert import from_networkx, graph as dgl_graph, to_networkx
from ..transforms import reorder_graph
from .dgl_dataset import DGLBuiltinDataset

from .utils import (
    _get_dgl_url,
    deprecate_function,
    deprecate_property,
    generate_mask_tensor,
    load_graphs,
    load_info,
    makedirs,
    save_graphs,
    save_info,
)

backend = os.environ.get("DGLBACKEND", "pytorch")


def _pickle_load(pkl_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        if sys.version_info > (3, 0):
            return pkl.load(pkl_file, encoding="latin1")
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
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    reorder : bool
        Whether to reorder the graph using :func:`~dgl.reorder_graph`. Default: False.
    """

    _urls = {
        "cora_v2": "dataset/cora_v2.zip",
        "citeseer": "dataset/citeseer.zip",
        "pubmed": "dataset/pubmed.zip",
    }

    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        reverse_edge=True,
        transform=None,
        reorder=False,
    ):
        assert name.lower() in ["cora", "citeseer", "pubmed"]

        # Previously we use the pre-processing in pygcn (https://github.com/tkipf/pygcn)
        # for Cora, which is slightly different from the one used in the GCN paper
        if name.lower() == "cora":
            name = "cora_v2"

        url = _get_dgl_url(self._urls[name])
        self._reverse_edge = reverse_edge
        self._reorder = reorder

        super(CitationGraphDataset, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        """Loads input data from data directory and reorder graph for better locality

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
        """
        root = self.raw_path
        objnames = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
        objects = []
        for i in range(len(objnames)):
            with open(
                "{}/ind.{}.{}".format(root, self.name, objnames[i]), "rb"
            ) as f:
                objects.append(_pickle_load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = _parse_index_file(
            "{}/ind.{}.test.index".format(root, self.name)
        )
        test_idx_range = np.sort(test_idx_reorder)

        if self.name == "citeseer":
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(
                min(test_idx_reorder), max(test_idx_reorder) + 1
            )
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        if self.reverse_edge:
            graph = nx.DiGraph(nx.from_dict_of_lists(graph))
            g = from_networkx(graph)
        else:
            graph = nx.Graph(nx.from_dict_of_lists(graph))
            edges = list(graph.edges())
            u, v = map(list, zip(*edges))
            g = dgl_graph((u, v))

        onehot_labels = np.vstack((ally, ty))
        onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
        labels = np.argmax(onehot_labels, 1)

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = generate_mask_tensor(
            _sample_mask(idx_train, labels.shape[0])
        )
        val_mask = generate_mask_tensor(_sample_mask(idx_val, labels.shape[0]))
        test_mask = generate_mask_tensor(
            _sample_mask(idx_test, labels.shape[0])
        )

        g.ndata["train_mask"] = train_mask
        g.ndata["val_mask"] = val_mask
        g.ndata["test_mask"] = test_mask
        g.ndata["label"] = F.tensor(labels)
        g.ndata["feat"] = F.tensor(
            _preprocess_features(features), dtype=F.data_type_dict["float32"]
        )
        self._num_classes = onehot_labels.shape[1]
        self._labels = labels
        if self._reorder:
            self._g = reorder_graph(
                g,
                node_permute_algo="rcmk",
                edge_permute_algo="dst",
                store_ids=False,
            )
        else:
            self._g = g

        if self.verbose:
            print("Finished data loading and preprocessing.")
            print("  NumNodes: {}".format(self._g.num_nodes()))
            print("  NumEdges: {}".format(self._g.num_edges()))
            print("  NumFeats: {}".format(self._g.ndata["feat"].shape[1]))
            print("  NumClasses: {}".format(self.num_classes))
            print(
                "  NumTrainingSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["train_mask"]).shape[0]
                )
            )
            print(
                "  NumValidationSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["val_mask"]).shape[0]
                )
            )
            print(
                "  NumTestSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["test_mask"]).shape[0]
                )
            )

    @property
    def graph_path(self):
        return os.path.join(self.save_path, self.save_name + ".bin")

    @property
    def info_path(self):
        return os.path.join(self.save_path, self.save_name + ".pkl")

    def has_cache(self):
        if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        save_graphs(str(self.graph_path), self._g)
        save_info(str(self.info_path), {"num_classes": self.num_classes})

    def load(self):
        graphs, _ = load_graphs(str(self.graph_path))

        info = load_info(str(self.info_path))
        graph = graphs[0]
        self._g = graph
        # for compatability
        graph = graph.clone()
        graph.ndata.pop("train_mask")
        graph.ndata.pop("val_mask")
        graph.ndata.pop("test_mask")
        graph.ndata.pop("feat")
        graph.ndata.pop("label")
        graph = to_networkx(graph)

        self._num_classes = info["num_classes"]
        self._g.ndata["train_mask"] = generate_mask_tensor(
            F.asnumpy(self._g.ndata["train_mask"])
        )
        self._g.ndata["val_mask"] = generate_mask_tensor(
            F.asnumpy(self._g.ndata["val_mask"])
        )
        self._g.ndata["test_mask"] = generate_mask_tensor(
            F.asnumpy(self._g.ndata["test_mask"])
        )
        # hack for mxnet compatability

        if self.verbose:
            print("  NumNodes: {}".format(self._g.num_nodes()))
            print("  NumEdges: {}".format(self._g.num_edges()))
            print("  NumFeats: {}".format(self._g.ndata["feat"].shape[1]))
            print("  NumClasses: {}".format(self.num_classes))
            print(
                "  NumTrainingSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["train_mask"]).shape[0]
                )
            )
            print(
                "  NumValidationSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["val_mask"]).shape[0]
                )
            )
            print(
                "  NumTestSamples: {}".format(
                    F.nonzero_1d(self._g.ndata["test_mask"]).shape[0]
                )
            )

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    @property
    def save_name(self):
        return self.name + "_dgl_graph"

    @property
    def num_labels(self):
        deprecate_property("dataset.num_labels", "dataset.num_classes")
        return self.num_classes

    @property
    def num_classes(self):
        return self._num_classes

    """ Citation graph is used in many examples
        We preserve these properties for compatability.
    """

    @property
    def reverse_edge(self):
        return self._reverse_edge


def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    features = _normalize(features)
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


class CoraGraphDataset(CitationGraphDataset):
    r"""Cora citation network dataset.

    Nodes mean paper and edges mean citation
    relationships. Each node has a predefined
    feature with 1433 dimensions. The dataset is
    designed for the node classification task.
    The task is to predict the category of
    certain paper.

    Statistics:

    - Nodes: 2708
    - Edges: 10556
    - Number of Classes: 7
    - Label split:

        - Train: 140
        - Valid: 500
        - Test: 1000

    Parameters
    ----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    reorder : bool
        Whether to reorder the graph using :func:`~dgl.reorder_graph`. Default: False.

    Attributes
    ----------
    num_classes: int
        Number of label classes

    Notes
    -----
    The node feature is row-normalized.

    Examples
    --------
    >>> dataset = CoraGraphDataset()
    >>> g = dataset[0]
    >>> num_class = dataset.num_classes
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

    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        reverse_edge=True,
        transform=None,
        reorder=False,
    ):
        name = "cora"

        super(CoraGraphDataset, self).__init__(
            name,
            raw_dir,
            force_reload,
            verbose,
            reverse_edge,
            transform,
            reorder,
        )

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, CoraGraphDataset has only one graph object

        Return
        ------
        :class:`dgl.DGLGraph`

            graph structure, node features and labels.

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']``: mask for test node set
            - ``ndata['feat']``: node feature
            - ``ndata['label']``: ground truth labels
        """
        return super(CoraGraphDataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(CoraGraphDataset, self).__len__()


class CiteseerGraphDataset(CitationGraphDataset):
    r"""Citeseer citation network dataset.

    Nodes mean scientific publications and edges
    mean citation relationships. Each node has a
    predefined feature with 3703 dimensions. The
    dataset is designed for the node classification
    task. The task is to predict the category of
    certain publication.

    Statistics:

    - Nodes: 3327
    - Edges: 9228
    - Number of Classes: 6
    - Label Split:

        - Train: 120
        - Valid: 500
        - Test: 1000

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    reorder : bool
        Whether to reorder the graph using :func:`~dgl.reorder_graph`. Default: False.

    Attributes
    ----------
    num_classes: int
        Number of label classes

    Notes
    -----
    The node feature is row-normalized.

    In citeseer dataset, there are some isolated nodes in the graph.
    These isolated nodes are added as zero-vecs into the right position.

    Examples
    --------
    >>> dataset = CiteseerGraphDataset()
    >>> g = dataset[0]
    >>> num_class = dataset.num_classes
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

    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        reverse_edge=True,
        transform=None,
        reorder=False,
    ):
        name = "citeseer"

        super(CiteseerGraphDataset, self).__init__(
            name,
            raw_dir,
            force_reload,
            verbose,
            reverse_edge,
            transform,
            reorder,
        )

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, CiteseerGraphDataset has only one graph object

        Return
        ------
        :class:`dgl.DGLGraph`

            graph structure, node features and labels.

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']``: mask for test node set
            - ``ndata['feat']``: node feature
            - ``ndata['label']``: ground truth labels
        """
        return super(CiteseerGraphDataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(CiteseerGraphDataset, self).__len__()


class PubmedGraphDataset(CitationGraphDataset):
    r"""Pubmed citation network dataset.

    Nodes mean scientific publications and edges
    mean citation relationships. Each node has a
    predefined feature with 500 dimensions. The
    dataset is designed for the node classification
    task. The task is to predict the category of
    certain publication.

    Statistics:

    - Nodes: 19717
    - Edges: 88651
    - Number of Classes: 3
    - Label Split:

        - Train: 60
        - Valid: 500
        - Test: 1000

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    reorder : bool
        Whether to reorder the graph using :func:`~dgl.reorder_graph`. Default: False.

    Attributes
    ----------
    num_classes: int
        Number of label classes

    Notes
    -----
    The node feature is row-normalized.

    Examples
    --------
    >>> dataset = PubmedGraphDataset()
    >>> g = dataset[0]
    >>> num_class = dataset.num_of_class
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

    """

    def __init__(
        self,
        raw_dir=None,
        force_reload=False,
        verbose=True,
        reverse_edge=True,
        transform=None,
        reorder=False,
    ):
        name = "pubmed"

        super(PubmedGraphDataset, self).__init__(
            name,
            raw_dir,
            force_reload,
            verbose,
            reverse_edge,
            transform,
            reorder,
        )

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, PubmedGraphDataset has only one graph object

        Return
        ------
        :class:`dgl.DGLGraph`

            graph structure, node features and labels.

            - ``ndata['train_mask']``: mask for training node set
            - ``ndata['val_mask']``: mask for validation node set
            - ``ndata['test_mask']``: mask for test node set
            - ``ndata['feat']``: node feature
            - ``ndata['label']``: ground truth labels
        """
        return super(PubmedGraphDataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(PubmedGraphDataset, self).__len__()


def load_cora(
    raw_dir=None,
    force_reload=False,
    verbose=True,
    reverse_edge=True,
    transform=None,
):
    """Get CoraGraphDataset

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Return
    -------
    CoraGraphDataset
    """
    data = CoraGraphDataset(
        raw_dir, force_reload, verbose, reverse_edge, transform
    )
    return data


def load_citeseer(
    raw_dir=None,
    force_reload=False,
    verbose=True,
    reverse_edge=True,
    transform=None,
):
    """Get CiteseerGraphDataset

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Return
    -------
    CiteseerGraphDataset
    """
    data = CiteseerGraphDataset(
        raw_dir, force_reload, verbose, reverse_edge, transform
    )
    return data


def load_pubmed(
    raw_dir=None,
    force_reload=False,
    verbose=True,
    reverse_edge=True,
    transform=None,
):
    """Get PubmedGraphDataset

    Parameters
    -----------
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information. Default: True.
    reverse_edge : bool
        Whether to add reverse edges in graph. Default: True.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Return
    -------
    PubmedGraphDataset
    """
    data = PubmedGraphDataset(
        raw_dir, force_reload, verbose, reverse_edge, transform
    )
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
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.
    """

    def __init__(
        self, raw_dir=None, force_reload=False, verbose=True, transform=None
    ):
        name = "cora_binary"
        url = _get_dgl_url("dataset/cora_binary.zip")
        super(CoraBinary, self).__init__(
            name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        root = self.raw_path
        # load graphs
        self.graphs = []
        with open("{}/graphs.txt".format(root), "r") as f:
            elist = []
            for line in f.readlines():
                if line.startswith("graph"):
                    if len(elist) != 0:
                        self.graphs.append(dgl_graph(tuple(zip(*elist))))
                    elist = []
                else:
                    u, v = line.strip().split(" ")
                    elist.append((int(u), int(v)))
            if len(elist) != 0:
                self.graphs.append(dgl_graph(tuple(zip(*elist))))
        with open("{}/pmpds.pkl".format(root), "rb") as f:
            self.pmpds = _pickle_load(f)
        self.labels = []
        with open("{}/labels.txt".format(root), "r") as f:
            cur = []
            for line in f.readlines():
                if line.startswith("graph"):
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

    @property
    def graph_path(self):
        return os.path.join(self.save_path, self.save_name + ".bin")

    def has_cache(self):
        if os.path.exists(self.graph_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        labels = {}
        for i, label in enumerate(self.labels):
            labels["{}".format(i)] = F.tensor(label)
        save_graphs(str(self.graph_path), self.graphs, labels)
        if self.verbose:
            print("Done saving data into cached files.")

    def load(self):
        self.graphs, labels = load_graphs(str(self.graph_path))

        self.labels = []
        for i in range(len(labels)):
            self.labels.append(F.asnumpy(labels["{}".format(i)]))
        # load pmpds under self.raw_path
        with open("{}/pmpds.pkl".format(self.raw_path), "rb") as f:
            self.pmpds = _pickle_load(f)
        if self.verbose:
            print("Done loading data into cached files.")
        # sanity check
        assert len(self.graphs) == len(self.pmpds)
        assert len(self.graphs) == len(self.labels)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, i):
        r"""Gets the idx-th sample.

        Parameters
        -----------
        idx : int
            The sample index.

        Returns
        -------
        (dgl.DGLGraph, scipy.sparse.coo_matrix, int)
            The graph, scipy sparse coo_matrix and its label.
        """
        if self._transform is None:
            g = self.graphs[i]
        else:
            g = self._transform(self.graphs[i])
        return (g, self.pmpds[i], self.labels[i])

    @property
    def save_name(self):
        return self.name + "_dgl_graph"

    @staticmethod
    def collate_fn(cur):
        graphs, pmpds, labels = zip(*cur)
        batched_graphs = batch_graphs(graphs)
        batched_pmpds = sp.block_diag(pmpds)
        batched_labels = np.concatenate(labels, axis=0)
        return batched_graphs, batched_pmpds, batched_labels


def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.asarray(mx.sum(1))
    mask = np.equal(rowsum, 0.0).flatten()
    rowsum[mask] = np.nan
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[mask] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def _encode_onehot(labels):
    classes = list(sorted(set(labels)))
    classes_dict = {
        c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)
    }
    labels_onehot = np.asarray(
        list(map(classes_dict.get, labels)), dtype=np.int32
    )
    return labels_onehot
