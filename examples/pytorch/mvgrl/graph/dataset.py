""" Code adapted from https://github.com/kavehhassani/mvgrl """
import os
import re
from collections import Counter

import dgl

import networkx as nx
import numpy as np
import torch as th
from dgl.data import DGLDataset
from scipy.linalg import fractional_matrix_power, inv

""" Compute Personalized Page Ranking"""


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])  # A^ = A + I_n
    d = np.diag(np.sum(a, 1))  # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)  # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)  # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv(
        (np.eye(a.shape[0]) - (1 - alpha) * at)
    )  # a(I_n-(1-a)A~)^-1


def download(dataset, datadir):
    os.makedirs(datadir)
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/{0}.zip".format(
        dataset
    )
    zipfile = os.path.basename(url)
    os.system("wget {0}; unzip {1}".format(url, zipfile))
    os.system("mv {0}/* {1}".format(dataset, datadir))
    os.system("rm -r {0}".format(dataset))
    os.system("rm {0}".format(zipfile))


def process(dataset):
    src = os.path.join(os.path.dirname(__file__), "data")
    prefix = os.path.join(src, dataset, dataset)

    # assign each node to the corresponding graph
    graph_node_dict = {}
    with open("{0}_graph_indicator.txt".format(prefix), "r") as f:
        for idx, line in enumerate(f):
            graph_node_dict[idx + 1] = int(line.strip("\n"))

    node_labels = []
    if os.path.exists("{0}_node_labels.txt".format(prefix)):
        with open("{0}_node_labels.txt".format(prefix), "r") as f:
            for line in f:
                node_labels += [int(line.strip("\n")) - 1]
            num_unique_node_labels = max(node_labels) + 1
    else:
        print("No node labels")

    node_attrs = []
    if os.path.exists("{0}_node_attributes.txt".format(prefix)):
        with open("{0}_node_attributes.txt".format(prefix), "r") as f:
            for line in f:
                node_attrs.append(
                    np.array(
                        [
                            float(attr)
                            for attr in re.split("[,\s]+", line.strip("\s\n"))
                            if attr
                        ],
                        dtype=float,
                    )
                )
    else:
        print("No node attributes")

    graph_labels = []
    unique_labels = set()
    with open("{0}_graph_labels.txt".format(prefix), "r") as f:
        for line in f:
            val = int(line.strip("\n"))
            if val not in unique_labels:
                unique_labels.add(val)
            graph_labels.append(val)
    label_idx_dict = {val: idx for idx, val in enumerate(unique_labels)}
    graph_labels = np.array([label_idx_dict[l] for l in graph_labels])

    adj_list = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    index_graph = {idx: [] for idx in range(1, len(graph_labels) + 1)}
    with open("{0}_A.txt".format(prefix), "r") as f:
        for line in f:
            u, v = tuple(map(int, line.strip("\n").split(",")))
            adj_list[graph_node_dict[u]].append((u, v))
            index_graph[graph_node_dict[u]] += [u, v]

    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs, pprs = [], []
    for idx in range(1, 1 + len(adj_list)):
        graph = nx.from_edgelist(adj_list[idx])

        graph.graph["label"] = graph_labels[idx - 1]
        for u in graph.nodes():
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                graph.nodes[u]["label"] = node_label_one_hot
            if len(node_attrs) > 0:
                graph.nodes[u]["feat"] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            graph.graph["feat_dim"] = node_attrs[0].shape[0]

        # relabeling
        mapping = {}
        for node_idx, node in enumerate(graph.nodes()):
            mapping[node] = node_idx

        graphs.append(nx.relabel_nodes(graph, mapping))
        pprs.append(compute_ppr(graph, alpha=0.2))

    if "feat_dim" in graphs[0].graph:
        pass
    else:
        max_deg = max([max(dict(graph.degree).values()) for graph in graphs])
        for graph in graphs:
            for u in graph.nodes(data=True):
                f = np.zeros(max_deg + 1)
                f[graph.degree[u[0]]] = 1.0
                if "label" in u[1]:
                    f = np.concatenate(
                        (np.array(u[1]["label"], dtype=float), f)
                    )
                graph.nodes[u[0]]["feat"] = f
    return graphs, pprs


def load(dataset):
    basedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.join(basedir, "data", dataset)

    if not os.path.exists(datadir):
        download(dataset, datadir)
        graphs, diff = process(dataset)
        feat, adj, labels = [], [], []

        for idx, graph in enumerate(graphs):
            adj.append(nx.to_numpy_array(graph))
            labels.append(graph.graph["label"])
            feat.append(
                np.array(list(nx.get_node_attributes(graph, "feat").values()))
            )

        adj, diff, feat, labels = (
            np.array(adj),
            np.array(diff),
            np.array(feat),
            np.array(labels),
        )

        np.save(f"{datadir}/adj.npy", adj)
        np.save(f"{datadir}/diff.npy", diff)
        np.save(f"{datadir}/feat.npy", feat)
        np.save(f"{datadir}/labels.npy", labels)
    else:
        adj = np.load(f"{datadir}/adj.npy", allow_pickle=True)
        diff = np.load(f"{datadir}/diff.npy", allow_pickle=True)
        feat = np.load(f"{datadir}/feat.npy", allow_pickle=True)
        labels = np.load(f"{datadir}/labels.npy", allow_pickle=True)

    n_graphs = adj.shape[0]

    graphs = []
    diff_graphs = []
    lbls = []

    for i in range(n_graphs):
        a = adj[i]
        edge_indexes = a.nonzero()

        graph = dgl.graph(edge_indexes)
        graph = graph.add_self_loop()
        graph.ndata["feat"] = th.tensor(feat[i]).float()

        diff_adj = diff[i]
        diff_indexes = diff_adj.nonzero()
        diff_weight = th.tensor(diff_adj[diff_indexes]).float()

        diff_graph = dgl.graph(diff_indexes)
        diff_graph.edata["edge_weight"] = diff_weight
        label = labels[i]
        graphs.append(graph)
        diff_graphs.append(diff_graph)
        lbls.append(label)

    labels = th.tensor(lbls)

    dataset = TUDataset(graphs, diff_graphs, labels)
    return dataset


class TUDataset(DGLDataset):
    def __init__(self, graphs, diff_graphs, labels):
        super(TUDataset, self).__init__(name="tu")
        self.graphs = graphs
        self.diff_graphs = diff_graphs
        self.labels = labels

    def process(self):
        return

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.diff_graphs[idx], self.labels[idx]
