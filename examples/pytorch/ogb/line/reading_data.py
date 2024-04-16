import os
import pickle
import random
import time

import dgl

import numpy as np
import scipy.sparse as sp
import torch
from dgl.data.utils import (
    _get_dgl_url,
    download,
    extract_archive,
    get_download_dir,
)
from torch.utils.data import DataLoader


def ReadTxtNet(file_path="", undirected=True):
    """Read the txt network file.
    Notations: The network is unweighted.

    Parameters
    ----------
    file_path str : path of network file
    undirected bool : whether the edges are undirected

    Return
    ------
    net dict : a dict recording the connections in the graph
    node2id dict : a dict mapping the nodes to their embedding indices
    id2node dict : a dict mapping nodes embedding indices to the nodes
    """
    if file_path == "youtube" or file_path == "blog":
        name = file_path
        dir = get_download_dir()
        zip_file_path = "{}/{}.zip".format(dir, name)
        download(
            _get_dgl_url(
                os.path.join("dataset/DeepWalk/", "{}.zip".format(file_path))
            ),
            path=zip_file_path,
        )
        extract_archive(zip_file_path, "{}/{}".format(dir, name))
        file_path = "{}/{}/{}-net.txt".format(dir, name, name)

    node2id = {}
    id2node = {}
    cid = 0

    src = []
    dst = []
    weight = []
    net = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            tup = list(map(int, line.strip().split(" ")))
            assert len(tup) in [
                2,
                3,
            ], "The format of network file is unrecognizable."
            if len(tup) == 3:
                n1, n2, w = tup
            elif len(tup) == 2:
                n1, n2 = tup
                w = 1
            if n1 not in node2id:
                node2id[n1] = cid
                id2node[cid] = n1
                cid += 1
            if n2 not in node2id:
                node2id[n2] = cid
                id2node[cid] = n2
                cid += 1

            n1 = node2id[n1]
            n2 = node2id[n2]
            if n1 not in net:
                net[n1] = {n2: w}
                src.append(n1)
                dst.append(n2)
                weight.append(w)
            elif n2 not in net[n1]:
                net[n1][n2] = w
                src.append(n1)
                dst.append(n2)
                weight.append(w)

            if undirected:
                if n2 not in net:
                    net[n2] = {n1: w}
                    src.append(n2)
                    dst.append(n1)
                    weight.append(w)
                elif n1 not in net[n2]:
                    net[n2][n1] = w
                    src.append(n2)
                    dst.append(n1)
                    weight.append(w)

    print("node num: %d" % len(net))
    print("edge num: %d" % len(src))
    assert max(net.keys()) == len(net) - 1, "error reading net, quit"

    sm = sp.coo_matrix((np.array(weight), (src, dst)), dtype=np.float32)

    return net, node2id, id2node, sm


def net2graph(net_sm):
    """Transform the network to DGL graph

    Return
    ------
    G DGLGraph : graph by DGL
    """
    start = time.time()
    G = dgl.DGLGraph(net_sm)
    end = time.time()
    t = end - start
    print("Building DGLGraph in %.2fs" % t)
    return G


def make_undirected(G):
    G.add_edges(G.edges()[1], G.edges()[0])
    return G


def find_connected_nodes(G):
    nodes = torch.nonzero(G.out_degrees(), as_tuple=False).squeeze(-1)
    return nodes


class LineDataset:
    def __init__(
        self,
        net_file,
        batch_size,
        num_samples,
        negative=5,
        gpus=[0],
        fast_neg=True,
        ogbl_name="",
        load_from_ogbl=False,
        ogbn_name="",
        load_from_ogbn=False,
    ):
        """This class has the following functions:
        1. Transform the txt network file into DGL graph;
        2. Generate random walk sequences for the trainer;
        3. Provide the negative table if the user hopes to sample negative
        nodes according to nodes' degrees;

        Parameter
        ---------
        net_file str : path of the dgl network file
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        num_walks int : number of walks for each node
        batch_size int : number of node sequences in each batch
        negative int : negative samples for each positve node pair
        fast_neg bool : whether do negative sampling inside a batch
        """
        self.batch_size = batch_size
        self.negative = negative
        self.num_samples = num_samples
        self.num_procs = len(gpus)
        self.fast_neg = fast_neg

        if load_from_ogbl:
            assert (
                len(gpus) == 1
            ), "ogb.linkproppred is not compatible with multi-gpu training."
            from load_dataset import load_from_ogbl_with_name

            self.G = load_from_ogbl_with_name(ogbl_name)
        elif load_from_ogbn:
            assert (
                len(gpus) == 1
            ), "ogb.linkproppred is not compatible with multi-gpu training."
            from load_dataset import load_from_ogbn_with_name

            self.G = load_from_ogbn_with_name(ogbn_name)
        else:
            self.G = dgl.load_graphs(net_file)[0][0]
        self.G = make_undirected(self.G)
        print("Finish reading graph")

        self.num_nodes = self.G.num_nodes()

        start = time.time()
        seeds = np.random.choice(
            np.arange(self.G.num_edges()), self.num_samples, replace=True
        )  # edge index
        self.seeds = torch.split(
            torch.LongTensor(seeds),
            int(np.ceil(self.num_samples / self.num_procs)),
            0,
        )
        end = time.time()
        t = end - start
        print("generate %d samples in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        self.valid_nodes = find_connected_nodes(self.G)
        if not fast_neg:
            node_degree = self.G.out_degrees(self.valid_nodes).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=int)
            self.neg_table = []

            for idx, node in enumerate(self.valid_nodes):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=int)
            del node_degree

    def create_sampler(self, i):
        """create random walk sampler"""
        return EdgeSampler(self.G, self.seeds[i])

    def save_mapping(self, map_file):
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)


class EdgeSampler(object):
    def __init__(self, G, seeds):
        self.G = G
        self.seeds = seeds
        self.edges = torch.cat(
            (self.G.edges()[0].unsqueeze(0), self.G.edges()[1].unsqueeze(0)), 0
        ).t()

    def sample(self, seeds):
        """seeds torch.LongTensor : a batch of indices of edges"""
        return self.edges[torch.LongTensor(seeds)]
