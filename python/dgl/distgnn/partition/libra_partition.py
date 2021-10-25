"""
Copyright (c) 2021 Intel Corporation
 \file distgnn/partition/libra_partition.py
 \brief Libra - Vertex-cut based graph partitioner for distirbuted training
 \author Vasimuddin Md <vasimuddin.md@intel.com>,
         Guixiang Ma <guixiang.ma@intel.com>
         Sanchit Misra <sanchit.misra@intel.com>,
         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,         
         Sasikanth Avancha <sasikanth.avancha@intel.com>
         Nesreen K. Ahmed <nesreen.k.ahmed@intel.com>
 \cite Distributed Power-law Graph Computing: Theoretical and Empirical Analysis
"""

import os
import sys
import networkx as nx
import numpy as np
import csv
from statistics import mean
import random
import time

import dgl
import json
from dgl import DGLGraph
import torch as th
from dgl.data import register_data_args, load_data
from load_graph import load_reddit, load_ogb
from dgl.sparse import libra_vertex_cut
from dgl.sparse import libra2dgl_build_dict
from dgl.sparse import libra2dgl_set_lf
from dgl.sparse import libra2dgl_build_adjlist
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from scipy.io import mmread
from dgl.base import DGLError
import requests

## \brief This function reports replication per node
def rep_per_node(prefix, nc):
    ifile = os.path.join(prefix, 'replicationlist.csv')
    f = open(ifile, "r")
    r_dt = {}

    fline = f.readline()
    for line in f:
        if line[0] == '#':
            raise DGLError("Error: Read hash in rep_per_node func.")
    
        node = line.strip('\n')
        if r_dt.get(node, -100) == -100:
            r_dt[node] = 1
        else:
            r_dt[node] += 1

    f.close()
    ## checks
    for v in r_dt.values():
        if v >= nc:
            raise DGLError("Error: Unexpected event in rep_per_node func.")

    return r_dt


## \brief This function converts libra output to dgl format
def main_libra2dgl(resultdir, dataset, nc):
    """
    Converts the output from Libra partitioning to DGL/DistGNN graph input.
    It builds dictionaries to assign local IDs to nodes in the partitions as well
    as it build a database to keep track of the location of clone nodes in the remote
    partitions.

    Parameters
    ----------
    resultdir : Location where partitions in dgl format are stored
    dataset : Dataset name
    nc : Number of partitions

    Output
    ------
    Creates partX folder in resultdir location for each partition X

    Notes
    -----
    This output is directly used as input to DistGNN
    
    """
    tedges = 1615685872   ## total edges
    max_c = 1024   ## max partitions supported
    factor = 1.2
    
    ## for pre-allocated tensor size
    hash_edges = [int((tedges/i)*factor) for i in range(1, max_c + 1)]
    
    ## load graph for the feature gather
    args = Args(dataset)

    print("Loading data...", flush=True)
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        g_orig, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        g_orig, _ = load_ogb('ogbn-papers100M')
    elif args.dataset == 'proteins':
        print("Loading proteins")
        g_orig = load_proteins('proteins')
    elif args.dataset == 'ogbn-arxiv':
        print("Loading ogbn-arxiv")
        g_orig, _ = load_ogb('ogbn-arxiv')
    else:
        g_orig = load_data(args)[0]

    print("Done loading data.", flush=True)
    a,b = g_orig.edges()

    N_n = g_orig.number_of_nodes()
    print("Number of nodes in the graph: ", N_n)
    node_map = th.zeros(nc, dtype=th.int32)
    indices = th.zeros(N_n, dtype=th.int32)
    lftensor = th.zeros(N_n, dtype=th.int32)
    gdt_key = th.zeros(N_n, dtype=th.int32)
    gdt_value = th.zeros([N_n, nc], dtype=th.int32)
    offset = th.zeros(1, dtype=th.int32)
    ldt_ar = []
    
    gg = [DGLGraph() for i in range(nc)]
    part_nodes = []

    ## Iterator over number of partitions
    for i in range(nc):
        g = gg[i]
        fsize = hash_edges[nc]

        hash_nodes = th.zeros(2, dtype=th.int32)
        a = th.zeros(fsize, dtype=th.int64)
        b = th.zeros(fsize, dtype=th.int64)
        ldt_key = th.zeros(fsize, dtype=th.int64)
        ldt_ar.append(ldt_key)

        ## building node, parition dictionary
        ## Assign local node ids and mapping to global node ids
        libra2dgl_build_dict(a, b, indices, ldt_key, gdt_key, gdt_value,
                             node_map, offset, nc, i, fsize, hash_nodes,
                             resultdir)

        num_nodes = int(hash_nodes[0])
        num_edges = int(hash_nodes[1])
        part_nodes.append(num_nodes)
        
        g.add_edges(a[0:num_edges], b[0:num_edges])

    ########################################################
    ## fixing lf - 1-level tree for the split-nodes
    libra2dgl_set_lf(gdt_key, gdt_value, lftensor, nc, N_n)
    ########################################################
    graph_name = dataset
    part_method = 'Libra'
    num_parts = nc   ## number of paritions/communities
    num_hops = 0
    node_map_val = node_map.tolist()
    edge_map_val = 0
    out_path = resultdir

    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g_orig.number_of_nodes(),
                     'num_edges': g_orig.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val}
    ############################################################

    for i in range(nc):
        g = gg[0]
        num_nodes = part_nodes[i]
        adj        = th.zeros([num_nodes, nc - 1], dtype=th.int32)
        inner_node = th.zeros(num_nodes, dtype=th.int32)
        lf         = th.zeros(num_nodes, dtype=th.int32)
        ldt = ldt_ar[0]

        try:
            feat = g_orig.ndata['feat']
        except:
            feat = g_orig.ndata['features']

        try:
            labels = g_orig.ndata['label']
        except:
            labels = g_orig.ndata['labels']

        trainm = g_orig.ndata['train_mask']
        testm = g_orig.ndata['test_mask']
        valm = g_orig.ndata['val_mask']

        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes, feat_size], dtype=feat.dtype)

        glabels = th.zeros(num_nodes, dtype=labels.dtype)
        gtrainm = th.zeros(num_nodes, dtype=trainm.dtype)
        gtestm = th.zeros(num_nodes, dtype=testm.dtype)
        gvalm = th.zeros(num_nodes, dtype=valm.dtype)

        ## build remote node databse per local node
        ## gather feats, train, test, val, and labels for each partition
        libra2dgl_build_adjlist(feat, gfeat, adj, inner_node, ldt, gdt_key,
                                gdt_value, node_map, lf, lftensor, num_nodes,
                                nc, i, feat_size, labels, trainm, testm, valm,
                                glabels, gtrainm, gtestm, gvalm, feat.shape[0])
        

        g.ndata['adj'] = adj   ## databse of remote clones
        g.ndata['inner_node'] = inner_node  ## split node '0' else '1'
        g.ndata['feat'] = gfeat    ## gathered features
        g.ndata['lf'] = lf  ## 1-level tree among split nodes

        g.ndata['label'] = glabels
        g.ndata['train_mask'] = gtrainm
        g.ndata['test_mask'] = gtestm
        g.ndata['val_mask'] = gvalm

        lf = g.ndata['lf']
        print("Writing partition {} to file".format(i), flush=True)
        
        part = g
        part_id = i
        part_dir = os.path.join(out_path, "part" + str(part_id))
        node_feat_file = os.path.join(part_dir, "node_feat.dgl")
        edge_feat_file = os.path.join(part_dir, "edge_feat.dgl")
        part_graph_file = os.path.join(part_dir, "graph.dgl")
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
        os.makedirs(part_dir, mode=0o775, exist_ok=True)
        save_tensors(node_feat_file, part.ndata)
        save_graphs(part_graph_file, [part])
        
        del g
        del gg[0]
        del ldt
        del ldt_ar[0]
        
    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)
        
    return gg, node_map


## \brief Driver function for Libra output to DGL graph format conversion
def run_libra2dgl(dataset, resultdir, nc):
    th.set_printoptions(threshold=10)
        
    print("Dataset: ", dataset, flush=True)
    print("Result location: ",resultdir, flush=True)
    print("number of parititons: ", nc)

    r_dt = rep_per_node(resultdir, nc)
    partition_ar, node_map =  main_libra2dgl(resultdir, dataset, nc)


## \brief This function download Proteins dataset
def download_proteins():
    print("Downloading dataset...")
    print("This might a take while..")
    url = "https://portal.nersc.gov/project/m1982/GNN/"
    file_name = "subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
    url = url + file_name
    try:
        r = requests.get(url)
    except:
        raise DGLError("Error: Failed to download Proteins dataset!! Aborting..")
        
    with open("proteins.mtx", "wb") as handle:
        handle.write(r.content)

        
## \brief This function converts Proteins dataset from mtx to dgl format
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

    #print(g)
    return g

## \brief This function saves input dataset to dgl format
def save(g, dataset):
    print("Saving dataset..")
    part_dir = os.path.join("./" + dataset)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    save_tensors(node_feat_file, g.ndata)
    save_graphs(part_graph_file, [g])
    print("Graph saved successfully !!")
    

    
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
    """
    Performs vertex-cut based grpah partitioning
    Parameters
    ----------
    num_community : Number of partitions to create
    dataset : Input graph name to partition
    prefix : Output location

    Output
    ------
    Creates X partition folder as XCommunities (say, X=2, so, 2Communities)
    XCommunities contains communityZ.txt file per parition Z
    Each such file contains list of edges assigned to that partition.

    """
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
            raise DGLError("Error: Dataset {} not found !!!".format(dataset))

    print("Done loading the graph.", flush=True)
    
    N_n = G.number_of_nodes()  # number of nodes
    N_c = num_community     ## number of partitions/communities
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
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num-parts', type=int, default=2)
    argparser.add_argument('--out-dir', type=str, default='\"\"')
    args = argparser.parse_args()

    dataset = args.dataset
    nc = args.num_parts
    prefix = args.out_dir
    
    print("dataset: ", dataset)
    print("num partitions: ", nc)
    print("output location: ", prefix)

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
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-papers100M')
        index = 3
    elif dataset == 'proteins':
        resultdir = os.path.join(prefix, 'Libra_result_proteins')
        index = 2
    elif dataset == 'ogbn-arxiv':
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-arxiv')
    else:
        raise DGLError("Error: Input dataset {}  not found !!", dataset)
        
    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        raise DGLError("Error: Could not create directory: ", resultdir)

    ## Partitions per dataset 
    l = [[2,4,8,16], [2,4,8,16,32,64],[2,4,8,16,32,64],[32,64,128]]
    print("Output is stored in ", resultdir, flush=True)
    #print("Generating ", l[index], " partitions...", flush=True)
    print("Generating ", nc, " partitions...", flush=True)

    tic = time.time()
    #for num_community in l[index]:
    for i in range(1):
        num_community = nc    ## num communities or num partitions
        print("####################################################################")
        print("Executing parititons: ", num_community)
        ltic = time.time()
        try:
            resultdir_libra2dgl = os.path.join(resultdir, str(num_community) + "Communities")
            os.makedirs(resultdir_libra2dgl, mode=0o775, exist_ok=True)
        except:
            raise DGLError("Error: Could not create sub-directory: ", resultdir_libra2dgl)

        ## Libra partitioning
        max_weightsum  = vertex_cut_partition(num_community, dataset, resultdir)

        print(" ** Converting libra partitions to dgl graphs **")
        libra2dgl.run(dataset, resultdir_libra2dgl, num_community)
        print("Conversion libra2dgl completed !!!")
        ltoc = time.time()
        print("Time taken by {} partitions {:0.4f} sec".format(num_community, ltoc - ltic))
        print()

    toc = time.time()
    #print("Generated ", l[index], " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Generated ", nc, " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Partitioning completed successfully !!!")



def partition_graph(dataset, nc, prefix):
    print("dataset: ", dataset)
    print("num partitions: ", nc)
    print("output location: ", prefix)
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
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-papers100M')
        index = 3
    elif dataset == 'proteins':
        resultdir = os.path.join(prefix, 'Libra_result_proteins')
        index = 2
    elif dataset == 'ogbn-arxiv':
        resultdir = os.path.join(prefix, 'Libra_result_ogbn-arxiv')
    else:
        raise DGLError("Error: Input dataset {}  not found !!", dataset)
        
    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        raise DGLError("Error: Could not create directory: ", resultdir)

    ## Partitions per dataset 
    l = [[2,4,8,16], [2,4,8,16,32,64],[2,4,8,16,32,64],[32,64,128]]
    print("Output is stored in ", resultdir, flush=True)
    #print("Generating ", l[index], " partitions...", flush=True)
    print("Generating ", nc, " partitions...", flush=True)

    tic = time.time()
    #for num_community in l[index]: ## residual code for creating bunch of partitions
    for i in range(1):
        num_community = nc    ## num communities or num partitions
        print("####################################################################")
        print("Executing parititons: ", num_community)
        ltic = time.time()
        try:
            resultdir_libra2dgl = os.path.join(resultdir, str(num_community) + "Communities")
            os.makedirs(resultdir_libra2dgl, mode=0o775, exist_ok=True)
        except:
            raise DGLError("Error: Could not create sub-directory: ", resultdir_libra2dgl)

        ## Libra partitioning
        max_weightsum  = vertex_cut_partition(num_community, dataset, resultdir)

        print(" ** Converting libra partitions to dgl graphs **")
        #libra2dgl.run(dataset, resultdir_libra2dgl, num_community)
        run_libra2dgl(dataset, resultdir_libra2dgl, num_community)
        print("Conversion libra2dgl completed !!!")
        ltoc = time.time()
        print("Time taken by {} partitions {:0.4f} sec".format(num_community, ltoc - ltic))
        print()

    toc = time.time()
    #print("Generated ", l[index], " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Generated ", nc, " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Partitioning completed successfully !!!")
