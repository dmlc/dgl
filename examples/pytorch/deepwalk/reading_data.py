import os
import numpy as np
import scipy.sparse as sp
import pickle
import torch
from torch.utils.data import DataLoader
from dgl.data.utils import download, _get_dgl_url, get_download_dir, extract_archive
import random
import time
import dgl
from utils import shuffle_walks
#np.random.seed(3141592653)

def ReadTxtNet(file_path="", undirected=True):
    """ Read the txt network file. 
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
    if file_path == 'youtube' or file_path == 'blog':
        name = file_path
        dir = get_download_dir()
        zip_file_path='{}/{}.zip'.format(dir, name)
        download(_get_dgl_url(os.path.join('dataset/DeepWalk/', '{}.zip'.format(file_path))), path=zip_file_path)
        extract_archive(zip_file_path,
                        '{}/{}'.format(dir, name))
        file_path = "{}/{}/{}-net.txt".format(dir, name, name)

    node2id = {}
    id2node = {}
    cid = 0

    src = []
    dst = []
    net = {}
    with open(file_path, "r") as f:
        for line in f.readlines():
            n1, n2 = list(map(int, line.strip().split(" ")[:2]))
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
                net[n1] = {n2: 1}
                src.append(n1)
                dst.append(n2)
            elif n2 not in net[n1]:
                net[n1][n2] = 1
                src.append(n1)
                dst.append(n2)
            
            if undirected:
                if n2 not in net:
                    net[n2] = {n1: 1}
                    src.append(n2)
                    dst.append(n1)
                elif n1 not in net[n2]:
                    net[n2][n1] = 1
                    src.append(n2)
                    dst.append(n1)

    print("node num: %d" % len(net))
    print("edge num: %d" % len(src))
    assert max(net.keys()) == len(net) - 1, "error reading net, quit"

    sm = sp.coo_matrix(
        (np.ones(len(src)), (src, dst)),
        dtype=np.float32)

    return net, node2id, id2node, sm

def net2graph(net_sm):
    """ Transform the network to DGL graph

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

class DeepwalkDataset:
    def __init__(self, 
            net_file,
            map_file,
            walk_length=80,
            window_size=5,
            num_walks=10,
            batch_size=32,
            negative=5,
            gpus=[0],
            fast_neg=True,
            ):
        """ This class has the following functions:
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
        self.net, self.node2id, self.id2node, self.sm = ReadTxtNet(net_file)
        self.save_mapping(map_file)
        self.G = net2graph(self.sm)

        # random walk seeds
        start = time.time()
        seeds = torch.cat([torch.LongTensor(self.G.nodes())] * num_walks)
        self.seeds = torch.split(shuffle_walks(seeds), int(np.ceil(len(self.net) * self.num_walks / self.num_procs)), 0)
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = np.array(list(map(lambda x: len(self.net[x]), self.net.keys())))
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=np.int)
            self.neg_table = []
            for idx, node in enumerate(self.net.keys()):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.long)
            del node_degree

    def create_sampler(self, gpu_id):
        """ Still in construction...

        Several mode:
        1. do true negative sampling.
          1.1 from random walk sequence
          1.2 from node degree distribution
          return the sampled node ids
        2. do false negative sampling from random walk sequence
          save GPU, faster
          return the node indices in the sequences
        """
        return DeepwalkSampler(self.G, self.seeds[gpu_id], self.walk_length)

    def save_mapping(self, map_file):
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)

class DeepwalkSampler(object):
    def __init__(self, G, seeds, walk_length):
        self.G = G
        self.seeds = seeds
        self.walk_length = walk_length
    
    def sample(self, seeds):
        walks = dgl.contrib.sampling.random_walk(self.G, seeds, 
            1, self.walk_length-1)
        return walks
