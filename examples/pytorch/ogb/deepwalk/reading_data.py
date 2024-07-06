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
from utils import shuffle_walks


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
    G = dgl.from_scipy(net_sm)
    end = time.time()
    t = end - start
    print("Building DGLGraph in %.2fs" % t)
    return G


def make_undirected(G):
    G.add_edges(G.edges()[1], G.edges()[0])
    return G


def find_connected_nodes(G):
    nodes = G.out_degrees().nonzero().squeeze(-1)
    return nodes


class DeepwalkDataset:
    def __init__(
        self,
        net_file,
        map_file,
        walk_length,
        window_size,
        num_walks,
        batch_size,
        negative=5,
        gpus=[0],
        fast_neg=True,
        ogbl_name="",
        load_from_ogbl=False,
    ):
        """This class has the following functions:
        1. Transform the txt network file into DGL graph;
        2. Generate random walk sequences for the trainer;
        3. Provide the negative table if the user hopes to sample negative
        nodes according to nodes' degrees;

        Parameter
        ---------
        net_file str : path of the txt network file
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        num_walks int : number of walks for each node
        batch_size int : number of node sequences in each batch
        negative int : negative samples for each positve node pair
        fast_neg bool : whether do negative sampling inside a batch
        """
        self.walk_length = walk_length
        self.window_size = window_size
        self.num_walks = num_walks
        self.batch_size = batch_size
        self.negative = negative
        self.num_procs = len(gpus)
        self.fast_neg = fast_neg

        if load_from_ogbl:
            assert (
                len(gpus) == 1
            ), "ogb.linkproppred is not compatible with multi-gpu training (CUDA error)."
            from load_dataset import load_from_ogbl_with_name

            self.G = load_from_ogbl_with_name(ogbl_name)
            self.G = make_undirected(self.G)
        else:
            self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(net_file)
            self.save_mapping(map_file)
            self.G = net2graph(self.sm)

        self.num_nodes = self.G.num_nodes()

        # random walk seeds
        start = time.time()
        self.valid_seeds = find_connected_nodes(self.G)
        if len(self.valid_seeds) != self.num_nodes:
            print(
                "WARNING: The node ids are not serial. Some nodes are invalid."
            )

        seeds = torch.cat([torch.LongTensor(self.valid_seeds)] * num_walks)
        self.seeds = torch.split(
            shuffle_walks(seeds),
            int(
                np.ceil(len(self.valid_seeds) * self.num_walks / self.num_procs)
            ),
            0,
        )
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = self.G.out_degrees(self.valid_seeds).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=int)
            self.neg_table = []

            for idx, node in enumerate(self.valid_seeds):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=int)
            del node_degree

    def create_sampler(self, i):
        """create random walk sampler"""
        return DeepwalkSampler(self.G, self.seeds[i], self.walk_length)

    def save_mapping(self, map_file):
        """save the mapping dict that maps node IDs to embedding indices"""
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)


class DeepwalkSampler(object):
    def __init__(self, G, seeds, walk_length):
        """random walk sampler

        Parameter
        ---------
        G dgl.Graph : the input graph
        seeds torch.LongTensor : starting nodes
        walk_length int : walk length
        """
        self.G = G
        self.seeds = seeds
        self.walk_length = walk_length

    def sample(self, seeds):
        walks = dgl.sampling.random_walk(
            self.G, seeds, length=self.walk_length - 1
        )[0]
        return walks
