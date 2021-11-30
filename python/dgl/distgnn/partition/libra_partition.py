r"""
Copyright (c) 2021 Intel Corporation
 \file distgnn/partition/libra_partition.py
 \brief Libra - Vertex-cut based graph partitioner for distributed training
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
from dgl.sparse import libra2dgl_set_lr
from dgl.sparse import libra2dgl_build_adjlist
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors
from dgl.distgnn.tools import load_proteins
from scipy.io import mmread
from dgl.base import DGLError
import requests


class Args:
    def __init__(self, dataset):
        self.dataset = dataset


def libra_partition(num_community, dataset, resultdir):
    """
    Performs vertex-cut based graph partitioning and converts the partitioning
    output to DGL input format
    Parameters
    ----------
    num_community : Number of partitions to create
    dataset : Input graph name to partition
    prefix : Output location

    Output
    ------
    Creates X partition folder as XCommunities (say, X=2, so, 2Communities)
    XCommunities contains communityZ.txt file per partition Z
    Each such file contains a list of edges assigned to that partition.
    The folder also contains partX folders containing DGL graphs for the partitions;
    these graph files are used as input DistGNN.
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

    N_n = G.number_of_nodes()   # number of nodes
    N_c = num_community      ## number of partitions/communities
    N_e = G.number_of_edges()
    community_list = [[] for i in range(N_c)]
    print("Number of nodes in the graph: ", N_n)
    print("Number of edges in the graph: ", N_e)

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
    #del G

    ## call to C/C++ code
    out = th.zeros(u_t.shape[0], dtype=th.int32)
    libra_vertex_cut(N_c, node_degree, edgenum_unassigned, community_weights,
                     u_t, v_t, weight_, out, N_n, N_e, resultdir)

    print("Max partition size: ", int(community_weights.max()))
    print(" ** Converting libra partitions to dgl graphs **")
    fsize = int(community_weights.max()) + 1024   ## max edges in partition
    print("fsize: ", fsize)

    node_map = th.zeros(N_c, dtype=th.int64)
    indices = th.zeros(N_n, dtype=th.int64)
    lrtensor = th.zeros(N_n, dtype=th.int64)
    gdt_key = th.zeros(N_n, dtype=th.int64)
    gdt_value = th.zeros([N_n, N_c], dtype=th.int64)
    offset = th.zeros(1, dtype=th.int64)
    ldt_ar = []

    gg = [DGLGraph() for i in range(N_c)]
    part_nodes = []

    ## Iterator over number of partitions
    for i in range(N_c):
        g = gg[i]

        a = th.zeros(fsize, dtype=th.int64)
        b = th.zeros(fsize, dtype=th.int64)
        ldt_key = th.zeros(fsize, dtype=th.int64)
        ldt_ar.append(ldt_key)

        ## building node, parition dictionary
        ## Assign local node ids and mapping to global node ids
        ret = libra2dgl_build_dict(a, b, indices, ldt_key, gdt_key, gdt_value,
                             node_map, offset, N_c, i, fsize, resultdir)

        num_nodes = int(ret[0])
        num_edges = int(ret[1])
        # print("ret values: {} {}".format(int(ret[0]), int(ret[1])))
        part_nodes.append(num_nodes)

        g.add_edges(a[0:num_edges], b[0:num_edges])

    ########################################################
    ## fixing lr - 1-level tree for the split-nodes
    libra2dgl_set_lr(gdt_key, gdt_value, lrtensor, N_c, N_n)
    ########################################################
    graph_name = dataset
    part_method = 'Libra'
    num_parts = N_c   ## number of paritions/communities
    num_hops = 0
    node_map_val = node_map.tolist()
    edge_map_val = 0
    out_path = resultdir

    part_metadata = {'graph_name': graph_name,
                     'num_nodes': G.number_of_nodes(),
                     'num_edges': G.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_map': node_map_val,
                     'edge_map': edge_map_val}
    ############################################################

    for i in range(N_c):
        g = gg[0]
        num_nodes = part_nodes[i]
        adj        = th.zeros([num_nodes, N_c - 1], dtype=th.int64)
        inner_node = th.zeros(num_nodes, dtype=th.int32)
        lr         = th.zeros(num_nodes, dtype=th.int64)
        ldt = ldt_ar[0]

        try:
            feat = G.ndata['feat']
        except:
            feat = G.ndata['features']

        try:
            labels = G.ndata['label']
        except:
            labels = G.ndata['labels']

        trainm = G.ndata['train_mask'].int()
        testm = G.ndata['test_mask'].int()
        valm = G.ndata['val_mask'].int()

        feat_size = feat.shape[1]
        gfeat = th.zeros([num_nodes, feat_size], dtype=feat.dtype)

        glabels = th.zeros(num_nodes, dtype=labels.dtype)
        gtrainm = th.zeros(num_nodes, dtype=trainm.dtype)
        gtestm = th.zeros(num_nodes, dtype=testm.dtype)
        gvalm = th.zeros(num_nodes, dtype=valm.dtype)

        ## build remote node databse per local node
        ## gather feats, train, test, val, and labels for each partition
        libra2dgl_build_adjlist(feat, gfeat, adj, inner_node, ldt, gdt_key,
                                gdt_value, node_map, lr, lrtensor, num_nodes,
                                N_c, i, feat_size, labels, trainm, testm, valm,
                                glabels, gtrainm, gtestm, gvalm, feat.shape[0])


        g.ndata['adj'] = adj    ## database of remote clones
        g.ndata['inner_node'] = inner_node   ## split node '0' else '1'
        g.ndata['feat'] = gfeat    ## gathered features
        g.ndata['lf'] = lr   ## 1-level tree among split nodes

        g.ndata['label'] = glabels
        g.ndata['train_mask'] = gtrainm
        g.ndata['test_mask'] = gtestm
        g.ndata['val_mask'] = gvalm

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

    print("Conversion libra2dgl completed !!!")


def partition_graph(dataset, N_c, prefix):
    print("dataset: ", dataset)
    print("num partitions: ", N_c)
    print("output location: ", prefix)

    out_dir = 'Libra_result_' + dataset
    resultdir = os.path.join(prefix, out_dir)

    ## create ouptut directory
    try:
        os.makedirs(resultdir, mode=0o775, exist_ok=True)
    except:
        raise DGLError("Error: Could not create directory: ", resultdir)

    ## Partitions per dataset
    print("Output is stored in ", resultdir, flush=True)
    print("Generating ", N_c, " partitions...", flush=True)

    tic = time.time()
    for i in range(1):
        num_community = N_c    ## num communities or num partitions
        print("####################################################################")
        print("Executing parititons: ", num_community)
        ltic = time.time()
        try:
            resultdir = os.path.join(resultdir, str(num_community) + "Communities")
            os.makedirs(resultdir, mode=0o775, exist_ok=True)
        except:
            raise DGLError("Error: Could not create sub-directory: ", resultdir)

        ## Libra partitioning
        libra_partition(num_community, dataset, resultdir)

        ltoc = time.time()
        print("Time taken by {} partitions {:0.4f} sec".format(num_community, ltoc - ltic))
        print()

    toc = time.time()
    print("Generated ", N_c, " partitions in {:0.4f} sec".format(toc - tic), flush=True)
    print("Partitioning completed successfully !!!")
