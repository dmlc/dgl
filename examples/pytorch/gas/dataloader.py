import os

import dgl

import numpy as np
import scipy.io as sio
import torch as th
from dgl.data import DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url, load_graphs, save_graphs


class GASDataset(DGLBuiltinDataset):
    file_urls = {"pol": "dataset/GASPOL.zip", "gos": "dataset/GASGOS.zip"}

    def __init__(
        self, name, raw_dir=None, random_seed=717, train_size=0.7, val_size=0.1
    ):
        assert name in ["gos", "pol"], "Only supports 'gos' or 'pol'."
        self.seed = random_seed
        self.train_size = train_size
        self.val_size = val_size
        url = _get_dgl_url(self.file_urls[name])
        super(GASDataset, self).__init__(name=name, url=url, raw_dir=raw_dir)

    def process(self):
        """process raw data to graph, labels and masks"""
        data = sio.loadmat(
            os.path.join(self.raw_path, f"{self.name}_retweet_graph.mat")
        )

        adj = data["graph"].tocoo()
        num_edges = len(adj.row)
        row, col = adj.row[: int(num_edges / 2)], adj.col[: int(num_edges / 2)]

        graph = dgl.graph(
            (np.concatenate((row, col)), np.concatenate((col, row)))
        )
        news_labels = data["label"].squeeze()
        num_news = len(news_labels)

        node_feature = np.load(
            os.path.join(self.raw_path, f"{self.name}_node_feature.npy")
        )
        edge_feature = np.load(
            os.path.join(self.raw_path, f"{self.name}_edge_feature.npy")
        )[: int(num_edges / 2)]

        graph.ndata["feat"] = th.tensor(node_feature)
        graph.edata["feat"] = th.tensor(np.tile(edge_feature, (2, 1)))
        pos_news = news_labels.nonzero()[0]

        edge_labels = th.zeros(num_edges)
        edge_labels[graph.in_edges(pos_news, form="eid")] = 1
        edge_labels[graph.out_edges(pos_news, form="eid")] = 1
        graph.edata["label"] = edge_labels

        ntypes = th.ones(graph.num_nodes(), dtype=int)
        etypes = th.ones(graph.num_edges(), dtype=int)

        ntypes[graph.nodes() < num_news] = 0
        etypes[: int(num_edges / 2)] = 0

        graph.ndata["_TYPE"] = ntypes
        graph.edata["_TYPE"] = etypes

        hg = dgl.to_heterogeneous(graph, ["v", "u"], ["forward", "backward"])
        self._random_split(hg, self.seed, self.train_size, self.val_size)

        self.graph = hg

    @property
    def graph_path(self):
        return os.path.join(self.save_path, self.name + "_dgl_graph.bin")

    def save(self):
        """save the graph list and the labels"""
        save_graphs(str(self.graph_path), self.graph)

    def has_cache(self):
        """check whether there are processed data in `self.save_path`"""
        return os.path.exists(self.graph_path)

    def load(self):
        """load processed data from directory `self.save_path`"""
        graph, _ = load_graphs(str(self.graph_path))
        self.graph = graph[0]

    @property
    def num_classes(self):
        """Number of classes for each graph, i.e. number of prediction tasks."""
        return 2

    def __getitem__(self, idx):
        r"""Get graph object
        Parameters
        ----------
        idx : int
            Item index
        Returns
        -------
        :class:`dgl.DGLGraph`
        """
        assert idx == 0, "This dataset has only one graph"
        return self.graph

    def __len__(self):
        r"""Number of data examples
        Return
        -------
        int
        """
        return len(self.graph)

    def _random_split(self, graph, seed=717, train_size=0.7, val_size=0.1):
        """split the dataset into training set, validation set and testing set"""

        assert 0 <= train_size + val_size <= 1, (
            "The sum of valid training set size and validation set size "
            "must between 0 and 1 (inclusive)."
        )

        num_edges = graph.num_edges(etype="forward")
        index = np.arange(num_edges)

        index = np.random.RandomState(seed).permutation(index)
        train_idx = index[: int(train_size * num_edges)]
        val_idx = index[num_edges - int(val_size * num_edges) :]
        test_idx = index[
            int(train_size * num_edges) : num_edges - int(val_size * num_edges)
        ]
        train_mask = np.zeros(num_edges, dtype=np.bool_)
        val_mask = np.zeros(num_edges, dtype=np.bool_)
        test_mask = np.zeros(num_edges, dtype=np.bool_)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        graph.edges["forward"].data["train_mask"] = th.tensor(train_mask)
        graph.edges["forward"].data["val_mask"] = th.tensor(val_mask)
        graph.edges["forward"].data["test_mask"] = th.tensor(test_mask)
        graph.edges["backward"].data["train_mask"] = th.tensor(train_mask)
        graph.edges["backward"].data["val_mask"] = th.tensor(val_mask)
        graph.edges["backward"].data["test_mask"] = th.tensor(test_mask)
