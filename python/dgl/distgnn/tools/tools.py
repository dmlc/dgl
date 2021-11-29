r"""
Copyright (c) 2021 Intel Corporation
 \file distgnn/tools/tools.py
 \brief Tools for use in DistGNN
 \author Vasimuddin Md <vasimuddin.md@intel.com>
"""

import os
import sys
import csv
import random
import time

import dgl
import json
from dgl import DGLGraph
import torch as th
from scipy.io import mmread
from dgl.base import DGLError
import requests
from dgl.data.utils import load_graphs, save_graphs, load_tensors, save_tensors

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
    ## load
    graph = load_graphs(graph_file)[0][0]
    return graph
