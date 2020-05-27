import numpy as np
import dgl
import torch
import random
import time
np.random.seed(3141592653)

def ReadTxtNet(file_path=""):
    """ Read the txt network file to a dict

    Return
    ------
    net dict : a dict recording the connections in the graph
    node2id dict : a dict mapping the nodes to their embedding indices 
    id2node dict : a dict mapping nodes embedding indices to the nodes
    """
    node2id = {}
    id2node = {}
    cid = 0

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
            try:
                net[n1][n2] = 1
            except:
                net[n1] = {n2: 1}
            try:
                net[n2][n1] = 1
            except:
                net[n2] = {n1: 1}

    print("node num: %d" % len(net))
    print("edge num: %d" % (sum(list(map(lambda i: len(net[i]), net.keys())))/2))
    if max(net.keys()) != len(net) - 1:
        print("error reading net, quit")
        exit(1)
    return net, node2id, id2node

def net2graph(net):
    """ Transform the network dict to DGL graph

    Return 
    ------
    G DGLGraph : graph by DGL
    """
    G = dgl.DGLGraph()
    G.add_nodes(len(net))
    for i in net:
        G.add_edges(i, list(net[i].keys()))
    return G

class DeepwalkDataset:
    def __init__(self, 
            net_file,
            walk_length=80,
            window_size=5,
            num_walks=10,
            batch_size=32,
            negative=5,
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
        self.net, self.node2id, self.id2node = ReadTxtNet(net_file)
        self.G = net2graph(self.net)

        # random walk
        start = time.time()
        walks = dgl.contrib.sampling.random_walk(self.G, self.G.nodes(), 
                self.num_walks, self.walk_length-1)
        self.walks = walks.view(-1, self.walk_length)
        end = time.time()
        t = end - start
        print("%d walks in %.2fs" % (len(self.walks), t))

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
        