import os
import sys
import networkx as nx
import numpy as np
import csv
from statistics import mean
import random
import time

import dgl
import torch as th
from dgl.data import register_data_args, load_data
from load_graph import load_reddit, load_ogb
from dgl.sparse import libra_vertex_cut
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from scipy.io import mmread
import libra2dgl
import requests

def download_proteins():
    print("Downloading dataset...")
    print("This might a take while..")
    url = "https://portal.nersc.gov/project/m1982/GNN/"
    file_name = "subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
    url = url + file_name
    try:
        r = requests.get(url)
    except:
        print("Error: can't download Proteins dataset!! Aborting..")
        
    with open("proteins.mtx", "wb") as handle:
        handle.write(r.content)
    

def proteins_mtx2dgl():
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a = mmread('proteins.mtx')
    coo = a.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    g = dgl.DGLGraph()
    g.add_edges(u,v)

    n = g.number_of_nodes()
    feat_size = 128         ## arbitrary number
    feats = th.empty([n, feat_size], dtype=th.float32)

    ## arbitrary numbers
    train_size = 1000000
    test_size = 500000
    val_size = 5000
    nlabels = 256

    train_mask = th.zeros(n, dtype=th.bool)
    test_mask  = th.zeros(n, dtype=th.bool)
    val_mask   = th.zeros(n, dtype=th.bool)
    label      = th.zeros(n, dtype=th.int64)

    for i in range(train_size):
        train_mask[i] = True

    for i in range(test_size):
        test_mask[train_size + i] = True

    for i in range(val_size):
        val_mask[train_size + test_size + i] = True

    for i in range(n):
        label[i] = random.choice(range(nlabels))

    g.ndata['feat'] = feats
    g.ndata['train_mask'] = train_mask
    g.ndata['test_mask'] = test_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['label'] = label

    print(g)
    return g


def save(g, dataset):
    print("Saving dataset..")
    part_dir = os.path.join("./" + dataset)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    save_tensors(node_feat_file, g.ndata)
    save_graphs(part_graph_file, [g])
    #print("Graph saved successfully !!")
    

def load_proteins(dataset):    
    part_dir = dataset
    graph_file = os.path.join(part_dir + "/graph.dgl")

    if not os.path.exists("proteins.mtx"):
        download_proteins()
    if not os.path.exists(graph_file):        
        g = proteins_mtx2dgl()
        save(g, dataset)
    
    graph = load_graphs(graph_file)[0][0]    
    return graph


class Args:
    def __init__(self, dataset):
        self.dataset = dataset
            
                
def leastload(weights_array):
    result = np.where(weights_array == np.amin(weights_array))
    return random.choice(result[0])


def vertex_cut_partition(num_community, dataset, prefix):
    args = Args(dataset)
    print("Input dataset: ", args.dataset)
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        G,_ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        G,_ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'proteins':
        G = load_proteins('proteins')
    elif args.dataset == 'ogbn-arxiv':
        print("Loading ogbn-arxiv")
        G, _ = load_ogb('ogbn-arxiv')                                
    else:
        try:
            G = load_data(args)[0]
        except:
            print("Dataset {} not found !!!".format(dataset))
            sys.exit(1)

    print("Done loading the graph.", flush=True)
    
    N_n = G.number_of_nodes()  # number of nodes
    N_c = num_community
    N_e = G.number_of_edges()
    community_list = [[] for i in range(N_c)]
    
    in_d = G.in_degrees()
    out_d = G.out_degrees()
    node_degree = in_d + out_d
    edgenum_unassigned = node_degree.clone()
    replication_list = []
        
    u_t,v_t = G.edges()
    weight_ = th.ones(u_t.shape[0], dtype=th.float32)
    community_weights = th.zeros(N_c, dtype=th.float32)

    self_loop = 0
    for i in range(len(u_t)):
        if u_t[i] == v_t[i]:
            self_loop += 1

    print("#self loops in the dataset: ", self_loop)

    del G

    ## call to C/C++ code
    out = th.zeros(u_t.shape[0], dtype=th.int32)
    libra_vertex_cut(N_c, node_degree, edgenum_unassigned, community_weights,
                     u_t, v_t, weight_, out, N_n, N_e, prefix)
        
    return int(community_weights.max())


if __name__ == "__main__":    
    if len(sys.argv) != 2:
        print("Error: Input dataset required !!")
        sys.exit(1)

    no_papers = True
    prefix = ""
    dataset = sys.argv[1]
    print("dataset: ", dataset)
    index = 0
    if dataset == 'cora':
        resultdir = os.path.join(prefix, 'Libra_result_cora')
    elif dataset == 'pubmed':
        resultdir = os.path.join(prefix, 'Libra_result_pubmed')
    elif dataset == 'citeseer':
        resultdir = os.path.join(prefix, 'Libra_result_citeseer')
    elif dataset == 'reddit':          
        resultdir = os.path.join(prefix, 'Libra_result_reddit')
    elif dataset == 'ogbn-products':   
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-products')
        index = 1
    elif dataset == 'ogbn-papers100M':
        no_papers = False
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-papers100M')
        index = 3
    elif dataset == 'proteins':        
        resultdir = os.path.join(prefix, 'Libra_result_proteins')
        index = 2
    elif dataset == 'ogbn-arxiv':      
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-arxiv')
    else:
        print("Error: dataset not found !!")
        sys.exit(1)

    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        print("Error: Could not create directory: ", resultdir)
        
    l = [[2,4,8,16], [2,4,8,16,32,64],[2,4,8,16,32,64],[32,64,128]]    
    print("Output is stored in ", resultdir, flush=True)
    #print("Partition range: ", l[index])
    print("Generating ", l[index], " partitions...", flush=True)
    #for num_community in [2, 4, 8, 32] :#, 16, 32, 64]:

    tic = time.time()
    for num_community in l[index]:        
        print("####################################################################")
        print("Executing parititons: ", num_community)
        ltic = time.time()
        try:
            resultdir_libra2dgl = os.path.join(resultdir, str(num_community) + "Communities")
            os.makedirs(resultdir_libra2dgl, mode=0o775, exist_ok=True)
        except:
            print("Error: Could not create sub-directory: ", resultdir_libra2dgl)
            sys.exit(1)

        ## Libra partitioning
        max_weightsum  = vertex_cut_partition(num_community, sys.argv[1], resultdir)
                
        print(" ** Converting libra partitions to dgl graphs **")
        libra2dgl.run(dataset, resultdir_libra2dgl, num_community, no_papers)
        print("Conversion libra2dgl completed !!!")
        ltoc = time.time()
        print("Time taken by {} partitions {:0.4f} sec".format(num_community, ltoc - ltic))
        print()

    toc = time.time()
    print("Generated ", l[index], " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Partitioning completed successfully !!!")
