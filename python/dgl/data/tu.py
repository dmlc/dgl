from __future__ import absolute_import
import numpy as np
import os
import random


from .dgl_dataset import DGLBuiltinDataset
from .utils import loadtxt, save_graphs, load_graphs, save_info, load_info
from .. import backend as F
from ..utils import retry_method_with_fix
from ..graph import DGLGraph

class LegacyTUDataset(DGLBuiltinDataset):
    r"""
    TUDataset contains lots of graph kernel datasets for graph classification.
    Use provided node feature by default. If no feature provided, use one-hot node label instead.
    If neither labels provided, use constant for node feature.

    Parameters
    ----------
    name : str
        Dataset Name, such as `ENZYMES`, `DD`, `COLLAB`
    use_pandas : bool
        Numpy's file read function has performance issue when file is large,
        using pandas can be faster.
        Default: False
    hidden_size : int
        Some dataset doesn't contain features.
        Use constant node features initialization instead, with hidden size as `hidden_size`.
        Default : 10
    max_allow_node : int
        Remove graphs that contains more nodes than `max_allow_node`.
        Default : None

    Returns
    -------
    LegacyTUDataset object as an iterable for dataloader.

    Examples
    --------
    >>> data = LegacyTUDataset('DD')
    >>> data[0]
    (DGLGraph(num_nodes=327, num_edges=1798,
             ndata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64), 'feat': Scheme(shape=(89,), dtype=torch.float64)}
             edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}), 0)
    """

    _url = r"https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, name, use_pandas=False,
                 hidden_size=10, max_allow_node=None,
                 raw_dir=None, force_reload=False, verbose=False):

        url = self._url.format(name)
        self.hidden_size = hidden_size
        self.max_allow_node = max_allow_node
        self.use_pandas = use_pandas
        super(LegacyTUDataset, self).__init__(name=name, url=url, raw_dir=raw_dir, force_reload=force_reload, verbose=verbose)

    def process(self, root_path):
        self.data_mode = None

        if self.use_pandas:
            import pandas as pd
            DS_edge_list = self._idx_from_zero(
                pd.read_csv(self._file_path("A"), delimiter=",", dtype=int, header=None).values)
        else:
            DS_edge_list = self._idx_from_zero(
                np.genfromtxt(self._file_path("A"), delimiter=",", dtype=int))

        DS_indicator = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(
            np.genfromtxt(self._file_path("graph_labels"), dtype=int))

        g = DGLGraph()
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.graph_lists = g.subgraphs(node_idx_list)
        self.num_labels = max(DS_graph_labels) + 1
        self.graph_labels = DS_graph_labels

        try:
            DS_node_labels = self._idx_from_zero(
                np.loadtxt(self._file_path("node_labels"), dtype=int))
            g.ndata['node_label'] = DS_node_labels
            one_hot_node_labels = self._to_onehot(DS_node_labels)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = one_hot_node_labels[idxs, :]
            self.data_mode = "node_label"
        except IOError:
            print("No Node Label Data")

        try:
            DS_node_attr = np.loadtxt(
                self._file_path("node_attributes"), delimiter=",")
            if DS_node_attr.ndim == 1:
                DS_node_attr = np.expand_dims(DS_node_attr, -1)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = DS_node_attr[idxs, :]
            self.data_mode = "node_attr"
        except IOError:
            print("No Node Attribute Data")

        if 'feat' not in g.ndata.keys():
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = np.ones((g.number_of_nodes(), hidden_size))
            self.data_mode = "constant"
            if self.verbose:
                print("Use Constant one as Feature with hidden size {}".format(hidden_size))

        # remove graphs that are too large by user given standard
        # optional pre-processing steop in conformity with Rex Ying's original
        # DiffPool implementation
        if self.max_allow_node:
            preserve_idx = []
            if self.verbose:
                print("original dataset length : ", len(self.graph_lists))
            for (i, g) in enumerate(self.graph_lists):
                if g.number_of_nodes() <= self.max_allow_node:
                    preserve_idx.append(i)
            self.graph_lists = [self.graph_lists[i] for i in preserve_idx]
            if self.verbose:
                print("after pruning graphs that are too big : ", len(self.graph_lists))
            self.graph_labels = [self.graph_labels[i] for i in preserve_idx]
            self.max_num_node = self.max_allow_node

    def save(self):
        graph_path = os.path.join(self.save_path, 'legacy_tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'legacy_tu_{}.pkl'.format(self.name))
        label_dict = {'labels': F.tensor(self.graph_labels)}
        info_dict = {'max_num_node': self.max_num_node,
                     'num_labels': self.num_labels}
        save_graphs(str(graph_path), self.graph_lists, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(self.save_path, 'legacy_tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'legacy_tu_{}.pkl'.format(self.name))
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict['labels']
        self.max_num_node = info_dict['max_num_node']
        self.num_labels = info_dict['num_labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'legacy_tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'legacy_tu_{}.pkl'.format(self.name))
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field and node label in `node_label` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

    def _file_path(self, category):
        return os.path.join(self.raw_path, self.name,
                            "{}_{}.txt".format(self.name, category))

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
        return self.graph_lists[0].ndata['feat'].shape[1],\
            self.num_labels,\
            self.max_num_node

class TUDataset(DGLBuiltinDataset):
    r"""
    TUDataset contains lots of graph kernel datasets for graph classification.
    Graphs may have node labels, node attributes, edge labels, and edge attributes,
    varing from different dataset.

    Parameters
    ----------
    name : str
        Dataset Name, such as `ENZYMES`, `DD`, `COLLAB`, `MUTAG`, can be the 
        datasets name on https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets.

    Returns
    -------
    TUDataset object as an iterable for dataloader.

    Examples
    --------
    >>> data = TUDataset('DD')
    >>> data[0]
    (DGLGraph(num_nodes=327, num_edges=1798,
             ndata_schemes={'node_labels': Scheme(shape=(1,), dtype=torch.int64), '_ID': Scheme(shape=(), dtype=torch.int64)}
             edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)}), tensor([0]))
    """

    _url = r"https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=False):
        url = self._url.format(name)
        super(TUDataset, self).__init__(name=name, url=url,
                                        raw_dir=raw_dir, force_reload=force_reload,
                                        verbose=verbose)
    
    def process(self, root_path):
        DS_edge_list = self._idx_from_zero(
            loadtxt(self._file_path("A"), delimiter=",").astype(int))
        DS_indicator = self._idx_from_zero(
            loadtxt(self._file_path("graph_indicator"), delimiter=",").astype(int))
        DS_graph_labels = self._idx_from_zero(
            loadtxt(self._file_path("graph_labels"), delimiter=",").astype(int))

        g = DGLGraph()
        g.add_nodes(int(DS_edge_list.max()) + 1)
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(np.max(DS_indicator) + 1):
            node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(node_idx[0])
            if len(node_idx[0]) > self.max_num_node:
                self.max_num_node = len(node_idx[0])

        self.num_labels = max(DS_graph_labels) + 1
        self.graph_labels = DS_graph_labels

        self.attr_dict = {
            'node_labels': ('ndata', 'node_labels'),
            'node_attributes': ('ndata', 'node_attr'),
            'edge_labels': ('edata', 'edge_labels'),
            'edge_attributes': ('edata', 'node_labels'),
        }

        for filename, field_name in self.attr_dict.items():
            try:
                data = loadtxt(self._file_path(filename),
                               delimiter=',').astype(int)
                if 'label' in filename:
                    data = self._idx_from_zero(data)
                getattr(g, field_name[0])[field_name[1]] = data
            except IOError:
                pass

        self.graph_lists = g.subgraphs(node_idx_list)
        for g in self.graph_lists:
            g.copy_from_parent()

    def save(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        label_dict = {'labels': F.tensor(self.graph_labels)}
        info_dict = {'max_num_node': self.max_num_node,
                     'num_labels': self.num_labels}
        save_graphs(str(graph_path), self.graph_lists, label_dict)
        save_info(str(info_path), info_dict)

    def load(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'tu_{}.pkl'.format(self.name))
        graphs, label_dict = load_graphs(str(graph_path))
        info_dict = load_info(str(info_path))

        self.graph_lists = graphs
        self.graph_labels = label_dict['labels']
        self.max_num_node = info_dict['max_num_node']
        self.num_labels = info_dict['num_labels']

    def has_cache(self):
        graph_path = os.path.join(self.save_path, 'tu_{}.bin'.format(self.name))
        info_path = os.path.join(self.save_path, 'legacy_tu_{}.pkl'.format(self.name))
        if os.path.exists(graph_path) and os.path.exists(info_path):
            return True
        return False

    def __getitem__(self, idx):
        """Get the i^th sample.
        Paramters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field and node label in `node_label` if available.
            And its label.
        """
        g = self.graph_lists[idx]
        return g, self.graph_labels[idx]

    def __len__(self):
        return len(self.graph_lists)

    def _file_path(self, category):
        return os.path.join(self.raw_path, self.name,
                            "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    def statistics(self):
        return self.graph_lists[0].ndata['feat'].shape[1], \
            self.num_labels, \
            self.max_num_node
