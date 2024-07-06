from __future__ import absolute_import

import os

import numpy as np

from .. import backend as F
from ..convert import graph as dgl_graph

from .dgl_dataset import DGLBuiltinDataset
from .utils import load_graphs, load_info, loadtxt, save_graphs, save_info


class LegacyTUDataset(DGLBuiltinDataset):
    r"""LegacyTUDataset contains lots of graph kernel datasets for graph classification.

    Parameters
    ----------
    name : str
        Dataset Name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``, can be the
        datasets name on `<https://chrsmrrs.github.io/datasets/docs/datasets/>`_.
    use_pandas : bool
        Numpy's file read function has performance issue when file is large,
        using pandas can be faster.
        Default: False
    hidden_size : int
        Some dataset doesn't contain features.
        Use constant node features initialization instead, with hidden size as ``hidden_size``.
        Default : 10
    max_allow_node : int
        Remove graphs that contains more nodes than ``max_allow_node``.
        Default : None
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    max_num_node : int
        Maximum number of nodes
    num_classes : int
        Number of classes
    num_labels : numpy.int64
        (DEPRECATED, use num_classes instead) Number of classes

    Notes
    -----
    LegacyTUDataset uses provided node feature by default. If no feature provided, it uses one-hot node label instead.
    If neither labels provided, it uses constant for node feature.

    The dataset sorts graphs by their labels.
    Shuffle is preferred before manual train/val split.

    Examples
    --------
    >>> data = LegacyTUDataset('DD')

    The dataset instance is an iterable

    >>> len(data)
    1178
    >>> g, label = data[1024]
    >>> g
    Graph(num_nodes=88, num_edges=410,
          ndata_schemes={'feat': Scheme(shape=(89,), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> label
    tensor(1)

    Batch the graphs and labels for mini-batch training

    >>> graphs, labels = zip(*[data[i] for i in range(16)])
    >>> batched_graphs = dgl.batch(graphs)
    >>> batched_labels = torch.tensor(labels)
    >>> batched_graphs
    Graph(num_nodes=9539, num_edges=47382,
          ndata_schemes={'feat': Scheme(shape=(89,), dtype=torch.float32), '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    """

    _url = r"https://www.chrsmrrs.com/graphkerneldatasets/{}.zip"

    def __init__(
        self,
        name,
        use_pandas=False,
        hidden_size=10,
        max_allow_node=None,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        url = self._url.format(name)
        self.hidden_size = hidden_size
        self.max_allow_node = max_allow_node
        self.use_pandas = use_pandas
        super(LegacyTUDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            hash_key=(name, use_pandas, hidden_size, max_allow_node),
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        self.data_mode = None

        if self.use_pandas:
            import pandas as pd

            DS_edge_list = self._idx_from_zero(
                pd.read_csv(
                    self._file_path("A"), delimiter=",", dtype=int, header=None
                ).values
            )
        else:
            DS_edge_list = self._idx_from_zero(
                np.genfromtxt(self._file_path("A"), delimiter=",", dtype=int)
            )

        DS_indicator = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_indicator"), dtype=int)
        )
        if os.path.exists(self._file_path("graph_labels")):
            DS_graph_labels = self._idx_from_zero(
                np.genfromtxt(self._file_path("graph_labels"), dtype=int)
            )
            self.num_labels = max(DS_graph_labels) + 1
            self.graph_labels = DS_graph_labels
        elif os.path.exists(self._file_path("graph_attributes")):
            DS_graph_labels = np.genfromtxt(
                self._file_path("graph_attributes"), dtype=float
            )
            self.num_labels = None
            self.graph_labels = DS_graph_labels
        else:
            raise Exception("Unknown graph label or graph attributes")

        g = dgl_graph(([], []))
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.graph_lists = [g.subgraph(node_idx) for node_idx in node_idx_list]

        try:
            DS_node_labels = self._idx_from_zero(
                np.loadtxt(self._file_path("node_labels"), dtype=int)
            )
            g.ndata["node_label"] = F.tensor(DS_node_labels)
            one_hot_node_labels = self._to_onehot(DS_node_labels)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata["feat"] = F.tensor(
                    one_hot_node_labels[idxs, :], F.float32
                )
            self.data_mode = "node_label"
        except IOError:
            print("No Node Label Data")

        try:
            DS_node_attr = np.loadtxt(
                self._file_path("node_attributes"), delimiter=","
            )
            if DS_node_attr.ndim == 1:
                DS_node_attr = np.expand_dims(DS_node_attr, -1)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata["feat"] = F.tensor(DS_node_attr[idxs, :], F.float32)
            self.data_mode = "node_attr"
        except IOError:
            print("No Node Attribute Data")

        if "feat" not in g.ndata.keys():
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata["feat"] = F.ones(
                    (g.num_nodes(), self.hidden_size), F.float32, F.cpu()
                )
            self.data_mode = "constant"
            if self.verbose:
                print(
                    "Use Constant one as Feature with hidden size {}".format(
                        self.hidden_size
                    )
                )

        # remove graphs that are too large by user given standard
        # optional pre-processing steop in conformity with Rex Ying's original
        # DiffPool implementation
        if self.max_allow_node:
            preserve_idx = []
            if self.verbose:
                print("original dataset length : ", len(self.graph_lists))
            for i, g in enumerate(self.graph_lists):
                if g.num_nodes() <= self.max_allow_node:
                    preserve_idx.append(i)
            self.graph_lists = [self.graph_lists[i] for i in preserve_idx]
            if self.verbose:
                print(
                    "after pruning graphs that are too big : ",
                    len(self.graph_lists),
                )
            self.graph_labels = [self.graph_labels[i] for i in preserve_idx]
            self.max_num_node = self.max_allow_node
        self.graph_labels = F.tensor(self.graph_labels)

    def save(self):
        label_dict = {"labels": self.graph_labels}
        info_dict = {
            "max_num_node": self.max_num_node,
            "num_labels": self.num_labels,
        }
        save_graphs(str(self.graph_path), self.graph_lists, label_dict)
        save_info(str(self.info_path), info_dict)

    def load(self):
        graphs, label_dict = load_graphs(str(self.graph_path))
        info_dict = load_info(str(self.info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict["labels"]
        self.max_num_node = info_dict["max_num_node"]
        self.num_labels = info_dict["num_labels"]

    @property
    def graph_path(self):
        return os.path.join(
            self.save_path, "legacy_tu_{}_{}.bin".format(self.name, self.hash)
        )

    @property
    def info_path(self):
        return os.path.join(
            self.save_path, "legacy_tu_{}_{}.pkl".format(self.name, self.hash)
        )

    def has_cache(self):
        if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
            Graph with node feature stored in ``feat`` field and node label in ``node_label`` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        if self._transform is not None:
            g = self._transform(g)
        return g, self.graph_labels[idx]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def _file_path(self, category):
        return os.path.join(
            self.raw_path, self.name, "{}_{}.txt".format(self.name, category)
        )

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _to_onehot(label_tensor):
        label_num = label_tensor.shape[0]
        assert np.min(label_tensor) == 0
        one_hot_tensor = np.zeros((label_num, np.max(label_tensor) + 1))
        one_hot_tensor[np.arange(label_num), label_tensor] = 1
        return one_hot_tensor

    def statistics(self):
        return (
            self.graph_lists[0].ndata["feat"].shape[1],
            self.num_labels,
            self.max_num_node,
        )

    @property
    def num_classes(self):
        return int(self.num_labels)


class TUDataset(DGLBuiltinDataset):
    r"""
    TUDataset contains lots of graph kernel datasets for graph classification.

    Parameters
    ----------
    name : str
        Dataset Name, such as ``ENZYMES``, ``DD``, ``COLLAB``, ``MUTAG``, can be the
        datasets name on `<https://chrsmrrs.github.io/datasets/docs/datasets/>`_.
    transform : callable, optional
        A transform that takes in a :class:`~dgl.DGLGraph` object and returns
        a transformed version. The :class:`~dgl.DGLGraph` object will be
        transformed before every access.

    Attributes
    ----------
    max_num_node : int
        Maximum number of nodes
    num_classes : int
        Number of classes
    num_labels : int
        (DEPRECATED, use num_classes instead) Number of classes

    Notes
    -----
    **IMPORTANT:** Some of the datasets have duplicate edges exist in the graphs, e.g.
    the edges in ``IMDB-BINARY`` are all duplicated.  DGL faithfully keeps the duplicates
    as per the original data.  Other frameworks such as PyTorch Geometric removes the
    duplicates by default.  You can remove the duplicate edges with :func:`dgl.to_simple`.

    Graphs may have node labels, node attributes, edge labels, and edge attributes,
    varing from different dataset.

    Labels are mapped to :math:`\lbrace 0,\cdots,n-1 \rbrace` where :math:`n` is the
    number of labels (some datasets have raw labels :math:`\lbrace -1, 1 \rbrace` which
    will be mapped to :math:`\lbrace 0, 1 \rbrace`). In previous versions, the minimum
    label was added so that :math:`\lbrace -1, 1 \rbrace` was mapped to
    :math:`\lbrace 0, 2 \rbrace`.

    The dataset sorts graphs by their labels.
    Shuffle is preferred before manual train/val split.

    Examples
    --------
    >>> data = TUDataset('DD')

    The dataset instance is an iterable

    >>> len(data)
    1178
    >>> g, label = data[1024]
    >>> g
    Graph(num_nodes=88, num_edges=410,
          ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'node_labels': Scheme(shape=(1,), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> label
    tensor([1])

    Batch the graphs and labels for mini-batch training

    >>> graphs, labels = zip(*[data[i] for i in range(16)])
    >>> batched_graphs = dgl.batch(graphs)
    >>> batched_labels = torch.tensor(labels)
    >>> batched_graphs
    Graph(num_nodes=9539, num_edges=47382,
          ndata_schemes={'node_labels': Scheme(shape=(1,), dtype=torch.int64), '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})

    """

    _url = r"https://www.chrsmrrs.com/graphkerneldatasets/{}.zip"

    def __init__(
        self,
        name,
        raw_dir=None,
        force_reload=False,
        verbose=False,
        transform=None,
    ):
        url = self._url.format(name)
        super(TUDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            force_reload=force_reload,
            verbose=verbose,
            transform=transform,
        )

    def process(self):
        DS_edge_list = self._idx_from_zero(
            loadtxt(self._file_path("A"), delimiter=",").astype(int)
        )
        DS_indicator = self._idx_from_zero(
            loadtxt(self._file_path("graph_indicator"), delimiter=",").astype(
                int
            )
        )

        if os.path.exists(self._file_path("graph_labels")):
            DS_graph_labels = self._idx_reset(
                loadtxt(self._file_path("graph_labels"), delimiter=",").astype(
                    int
                )
            )
            self.num_labels = int(max(DS_graph_labels) + 1)
            self.graph_labels = F.tensor(DS_graph_labels)
        elif os.path.exists(self._file_path("graph_attributes")):
            DS_graph_labels = loadtxt(
                self._file_path("graph_attributes"), delimiter=","
            ).astype(float)
            self.num_labels = None
            self.graph_labels = F.tensor(DS_graph_labels)
        else:
            raise Exception("Unknown graph label or graph attributes")

        g = dgl_graph(([], []))
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.attr_dict = {
            "node_labels": ("ndata", "node_labels"),
            "node_attributes": ("ndata", "node_attr"),
            "edge_labels": ("edata", "edge_labels"),
            "edge_attributes": ("edata", "node_labels"),
        }

        for filename, field_name in self.attr_dict.items():
            try:
                data = loadtxt(self._file_path(filename), delimiter=",")
                if "label" in filename:
                    data = F.tensor(self._idx_from_zero(data))
                else:
                    data = F.tensor(data)
                getattr(g, field_name[0])[field_name[1]] = data
            except IOError:
                pass

        self.graph_lists = [g.subgraph(node_idx) for node_idx in node_idx_list]

    @property
    def graph_path(self):
        return os.path.join(self.save_path, "tu_{}.bin".format(self.name))

    @property
    def info_path(self):
        return os.path.join(self.save_path, "tu_{}.pkl".format(self.name))

    def save(self):
        label_dict = {"labels": self.graph_labels}
        info_dict = {
            "max_num_node": self.max_num_node,
            "num_labels": self.num_labels,
        }
        save_graphs(str(self.graph_path), self.graph_lists, label_dict)
        save_info(str(self.info_path), info_dict)

    def load(self):
        graphs, label_dict = load_graphs(str(self.graph_path))
        info_dict = load_info(str(self.info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict["labels"]
        self.max_num_node = info_dict["max_num_node"]
        self.num_labels = info_dict["num_labels"]

    def has_cache(self):
        if os.path.exists(self.graph_path) and os.path.exists(self.info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the idx-th sample.

        Parameters
        ---------
        idx : int
            The sample index.

        Returns
        -------
        (:class:`dgl.DGLGraph`, Tensor)
            Graph with node feature stored in ``feat`` field and node label in ``node_labels`` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        if self._transform is not None:
            g = self._transform(g)
        return g, self.graph_labels[idx]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graph_lists)

    def _file_path(self, category):
        return os.path.join(
            self.raw_path, self.name, "{}_{}.txt".format(self.name, category)
        )

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def _idx_reset(idx_tensor):
        """Maps n unique labels to {0, ..., n-1} in an ordered fashion."""
        labels = np.unique(idx_tensor)
        relabel_map = {x: i for i, x in enumerate(labels)}
        new_idx_tensor = np.vectorize(relabel_map.get)(idx_tensor)
        return new_idx_tensor

    def statistics(self):
        return (
            self.graph_lists[0].ndata["feat"].shape[1],
            self.num_labels,
            self.max_num_node,
        )

    @property
    def num_classes(self):
        return self.num_labels
