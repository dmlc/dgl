import os
import sys
import networkx as nx
import numpy as np
import csv
from statistics import mean
import random

import sys
import dgl
import torch as th
from dgl.data import register_data_args, load_data
from load_graph import load_reddit, load_ogb
from dgl.sparse import libra_vertex_cut
from dgl.data.utils import load_tensors, load_graphs
import libra2dgl


def load_ucb(dataset):
    part_dir = os.path.join("/cold_storage/omics/gnn/graph_partitioning/cagnet_datasets/" + dataset)
    #node_feats_file = os.path.join(part_dir + "/node_feat.dgl")
    graph_file = os.path.join(part_dir + "/graph.dgl")
    
    #node_feats = load_tensors(node_feats_file)
    graph = load_graphs(graph_file)[0][0]
    #graph.ndata = node_feats
    
    return graph


class args_:
    def __init__(self, dataset):
        self.dataset = dataset
            
                
def leastload(weights_array):
    result = np.where(weights_array == np.amin(weights_array))
    return random.choice(result[0])


def vertexCut_v2(num_community, dataset, prefix):    #gexf file    
    args = args_(dataset)
    print("Input dataset: ", args.dataset)
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        G,_ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        G,_ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'proteins':
        G = load_ucb('proteins')
    elif args.dataset == 'ogbn-arxiv':
        print("Loading ogbn-arxiv")
        G, _ = load_ogb('ogbn-arxiv')                                
    else:
        try:
            G = load_data(args)[0]
        except:
            print("Dataset {} not found !!!".format(dataset))
            sys.exit(1)

    print("Done loading...", flush=True)

    N_n = G.number_of_nodes()  # number of nodes
    N_c = num_community
    N_e = G.number_of_edges()
    community_list = [[] for i in range(N_c)]
    
    in_d = G.in_degrees()
    out_d = G.out_degrees()
    node_degree = in_d + out_d
    edgenum_unassigned = node_degree.clone()
    replication_list = []
        
    u_,v_ = G.edges()
    weight_ = th.ones(u_.shape[0], dtype=th.float32)
    community_weights = th.zeros(N_c, dtype=th.float32)

    self_loop = 0
    for i in range(len(u_)):
        if u_[i] == v_[i]:
            self_loop += 1

    print("#self loops in the dataset: ", self_loop)

    del G

    ## call to the c code
    out = th.zeros(u_.shape[0], dtype=th.int32)
    libra_vertex_cut(N_c, node_degree, edgenum_unassigned, community_weights,
                     u_, v_, weight_, out, N_n, N_e, prefix)

        
    return int(community_weights.max())


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Error: Input_dataset required !!")
        sys.exit(1)

    no_papers = True
    prefix = ""
    dataset = sys.argv[1]
    print("dataset: ", dataset)
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
    elif dataset == 'ogbn-papers100M':
        no_papers = False
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-papers100M')
    elif dataset == 'proteins':        
        resultdir = os.path.join(prefix, 'Libra_result_proteins')
    elif dataset == 'ogbn-arxiv':      
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-arxiv')
    else:
        print("Error: No output directory created as dataset not found !!")
        sys.exit(1)

    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        print("Error: Could not create directory: ", resultdir)
        
        
    print("Output is stored in ", resultdir, flush=True)
    for num_community in [2, 4, 8] :#, 16, 32, 64]:
        print("Executing parititons: ", num_community)

        try:
            resultdir_ = os.path.join(resultdir, str(num_community) + "Communities")
            os.makedirs(resultdir_, mode=0o775, exist_ok=True)
        except:
            print("Error: Could not create sub-directory: ", resultdir_)

        max_weightsum  = vertexCut_v2(num_community, sys.argv[1], resultdir)
        
        #print('\nThe max weight sum and replication number for community= {} is: {}, {}'.
        #      format(num_community, max_weightsum, 0))
        
        print("\nConverting libra output to dgl graphs:")
        libra2dgl.run(dataset, resultdir_, num_community, no_papers)
        print("Conversion libra2dgl completed !!!")
        print()

    print("Partitioning completed successfully !!!")
