from __future__ import absolute_import
import numpy as np
import dgl
import os
from .utils import download, extract_archive, get_download_dir, _get_dgl_url
import networkx as nx

class TUDataset(object):

    _url= r"https://ls11-www.cs.uni-dortmund.de/people/morris/graphkerneldatasets/{}.zip"

    def __init__(self, name, use_node_attr=False, use_node_label=False):
        # kwargs for now is for diffpool specifically.
        self.name = name
        self.extract_dir = self._download()
        DS_edge_list = self._idx_from_zero(np.loadtxt(self._file_path("A"), delimiter=",", dtype=int))
        DS_indicator = self._idx_from_zero(np.loadtxt(self._file_path("graph_indicator"), dtype=int))
        DS_graph_labels = self._idx_from_zero(np.loadtxt(self._file_path("graph_labels"), dtype=int))

        g = dgl.DGLGraph() # megagraph contains all nodes
        g.add_nodes(len(DS_indicator))
        g.add_edges(DS_edge_list[:, 0], DS_edge_list[:, 1])
        g.add_edges(DS_edge_list[:, 1], DS_edge_list[:, 0])
        # assume original list is only one-dimensional
        self.max_degrees = int(max(list(g.in_degrees())))

        node_idx_list = []
        self.max_num_node = 0
        for idx in range(len(DS_graph_labels)):
            subgraph_node_idx = np.where(DS_indicator == idx)
            node_idx_list.append(subgraph_node_idx[0])
            if len(subgraph_node_idx[0]) > self.max_num_node:
                self.max_num_node = len(subgraph_node_idx[0])

        self.graph_lists = g.subgraphs(node_idx_list)

        self.graph_labels = DS_graph_labels

        if use_node_label:
            DS_node_labels = self._idx_from_zero(np.loadtxt(self._file_path("node_labels"), dtype=int))
            self.max_node_label = max(DS_node_labels)
            for idxs, g in zip(node_idx_list, self.graph_lists):
                # by default we make node label one-hot. Assume label is
                # one-dim.
                node_label_embedding = self.one_hotify(DS_node_labels[list(idxs), ...], pad=True,
                                                        result_dim = self.max_node_label)
                g.ndata['node_label'] = node_label_embedding

        if use_node_attr:
            # assume node attribute dim unified
            DS_node_attr = np.loadtxt(self._file_path("node_attributes"), delimiter=",")
            for idxs, g in zip(node_idx_list, self.graph_lists):
                g.ndata['feat'] = DS_node_attr[list(idxs), ...]




    def __len__(self):
        return len(node_idx_list)

    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]

    @staticmethod
    def collate_fn(batch):
        graphs, labels = zip(*batch)
        batched_graphs = dgl.batch(graphs)
        batched_label = np.concatenate(labels, axis=0)
        return batched_graphs, batched_labels

    def _download(self):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(download_dir, "tu_{}.zip".format(self.name))
        download(self._url.format(self.name), path=zip_file_path)
        extract_dir = os.path.join(download_dir, "tu_{}".format(self.name))
        extract_archive(zip_file_path, extract_dir)
        return extract_dir

    def _file_path(self, category):
        return os.path.join(self.extract_dir, self.name, "{}_{}.txt".format(self.name, category))

    @staticmethod
    def _idx_from_zero(idx_tensor):
        return idx_tensor - np.min(idx_tensor)

    @staticmethod
    def one_hotify(labels, pad=False, result_dim=None):
        num_instances = len(labels)
        if not pad:
            dim_embedding = np.max(labels) + 1 #zero-indexed assumed
        else:
            assert result_dim, "result_dim for padding one hot embedding not set!"
            dim_embedding = result_dim + 1
        embeddings = np.zeros((num_instances, dim_embedding))
        embeddings[np.arange(num_instances),labels] = 1

        return embeddings

    def statistics(self):
        return self.grpah_lists[0].ndata['feat'].shape[1]

class DiffpoolDataset(TUDataset):
    def __init__(self, name, use_node_attr, use_node_label, mode='train',
                 train_ratio=0.8, test_ratio=0.1, **kwargs):
        super(DiffpoolDataset, self).__init__(name, use_node_attr,
                                            use_node_label)
        self.kwargs = kwargs
        self.use_node_attr = use_node_attr
        self.mode = mode
        self._preprocess()
        print("_proprocess for diffpool done")

        # train vs val vs test split
        train_idx = int(len(self.graph_lists)*train_ratio)
        test_idx = int(len(self.graph_lists)*(1-test_ratio))

        self.train_graphs = self.graph_lists[:train_idx]
        self.val_graphs = self.graph_lists[train_idx:test_idx]
        self.test_graphs = self.graph_lists[test_idx:]

        self.train_labels = self.graph_labels[:train_idx]
        self.val_labels = self.graph_labels[train_idx:test_idx]
        self.test_labels = self.graph_labels[test_idx:]

        # report dataset statistics
        print("Num of training graphs: ", len(self.train_labels))
        print("Num of validation graphs: ", len(self.val_labels))
        print("Num of testing graphs: ", len(self.test_labels))

    def set_mode(self, new_mode):
        self.mode = new_mode

    def _preprocess(self):
        """
        diffpool specific data partition, pre-process and shuffling
        """
        # adjacency degree normalization -- not done here
        if self.kwargs['feature_mode'] == 'id':
            for g in self.graph_lists:
                id_list = np.arange(g.number_of_nodes)
                g.ndata['feat'] = self.one_hotify(id_list, pad=True,
                                                      result_dim =
                                                      self.max_num_node)
        elif self.kwargs['feature_mode'] == 'deg-num':
            for g in self.graph_lists:
                g.ndata['feat'] = np.expand_dims(g.in_degrees(), axis=1)

        elif self.kwargs['feature_mode'] == 'deg':
            # max degree is disabled.
            for g in self.graph_lists:
                degs = list(g.in_degrees())
                degs_one_hot = self.one_hotify(degs, pad=True, result_dim =
                                                self.max_degrees)
                g.ndata['feat'] = np.concatenate((g.ndata['feat'],
                                                    degs_one_hot), axis=1)
        elif self.kwargs['feature_mode'] == 'struct':
            for g in self.graph_lists:
                degs = list(g.in_degrees())
                degs_one_hot = self.one_hotify(degs, pad=True, result_dim =
                                                self.max_degrees)
                nxg = g.to_networkx().to_undirected()
                clustering_coeffs = np.array(list(nx.clustering(nxg).values()))
                clustering_embedding = np.expand_dims(clustering_coeffs,
                                                        axis=1)
                struct_feats = np.concatenate((degs_one_hot,
                                                clustering_embedding),
                                                axis=1)
                if self.use_node_attr:
                    g.ndata['feat'] = np.concatenate((struct_feats,
                                                        g.ndata['feat']),
                                                        axis=1)
                else:
                    g.ndata['feat'] = struct_feats

        assert 'feat' in self.graph_lists[0].ndata, "node feature not initialized!"

        if self.kwargs['assign_feat'] == 'id':
            for g in self.graph_lists:
                id_list = np.arange(g.number_of_nodes())
                g.ndata['a_feat'] = self.one_hotify(id_list, pad=True,
                                                    result_dim=self.max_num_node)
        else:
            for g in self.graph_lists:
                id_list = np.arange(g.number_of_nodes())
                id_embedding = self.one_hotify(id_list, pad=True,
                                                result_dim=self.max_num_node)
                g.ndata['a_feat'] = np.concatenate((id_embedding,
                                                    g.ndata['feat']),
                                                    axis=1)
        # sanity check
        assert self.graph_lists[0].ndata['feat'].shape[1] == self.graph_lists[1].ndata['feat'].shape[1]

    def __len__(self):
        if self.mode == 'train':
            assert len(self.train_graphs)
            return len(self.train_graphs)

        elif self.mode == 'val':
            assert len(self.val_graphs)
            return len(self.val_graphs)

        elif self.mode == 'test':
            assert len(self.test_graphs)
            return len(self.test_graphs)

        else:
            raise NotImplementedError

    def __getitem__(self, idx):
        if self.mode == 'train':
            return self.train_graphs[idx], self.train_labels[idx]

        elif self.mode == 'val':
            return self.val_graphs[idx], self.val_labels[idx]

        elif self.mode == 'test':
            return self.test_graphs[idx], self.test_labels[idx]

        else:
            print("warning -- reading dataset without train/val/test split")
            return self.graph_lists[idx], self.graph_labels[idx]

    def statistics(self):
        print("DEBUGGGG")
        print(self.graph_labels[0])
        print(len(self.graph_labels))
        return self.graph_lists[0].ndata['feat'].shape[1], self.graph_lists[0].ndata['a_feat'].shape[1], self.graph_labels[0].shape
