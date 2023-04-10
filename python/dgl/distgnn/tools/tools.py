r"""
Copyright (c) 2021 Intel Corporation
 \file distgnn/tools/tools.py
 \brief Tools for use in Libra graph partitioner.
 \author Vasimuddin Md <vasimuddin.md@intel.com>
"""

import os
import random

import requests
import torch as th
from scipy.io import mmread

import dgl
from dgl.base import DGLError
from dgl.data.utils import load_graphs, save_graphs, save_tensors


def rep_per_node(prefix, num_community):
    """
    Used on Libra partitioned data.
    This function reports number of split-copes per node (replication) of
    a partitioned graph
    Parameters
    ----------
    prefix: Partition folder location (contains replicationlist.csv)
    num_community: number of partitions or communities
    """
    ifile = os.path.join(prefix, "replicationlist.csv")
    fhandle = open(ifile, "r")
    r_dt = {}

    fline = fhandle.readline()  ## reading first line, contains the comment.
    print(fline)
    for line in fhandle:
        if line[0] == "#":
            raise DGLError("[Bug] Read Hash char in rep_per_node func.")

        node = line.strip("\n")
        if r_dt.get(node, -100) == -100:
            r_dt[node] = 1
        else:
            r_dt[node] += 1

    fhandle.close()
    ## sanity checks
    for v in r_dt.values():
        if v >= num_community:
            raise DGLError(
                "[Bug] Unexpected event in rep_per_node() in tools.py."
            )

    return r_dt


def download_proteins():
    """
    Downloads the proteins dataset
    """
    print("Downloading dataset...")
    print("This might a take while..")
    url = "https://portal.nersc.gov/project/m1982/GNN/"
    file_name = "subgraph3_iso_vs_iso_30_70length_ALL.m100.propermm.mtx"
    url = url + file_name
    try:
        req = requests.get(url)
    except:
        raise DGLError(
            "Error: Failed to download Proteins dataset!! Aborting.."
        )

    with open("proteins.mtx", "wb") as handle:
        handle.write(req.content)


def proteins_mtx2dgl():
    """
    This function converts Proteins dataset from mtx to dgl format.
    """
    print("Converting mtx2dgl..")
    print("This might a take while..")
    a_mtx = mmread("proteins.mtx")
    coo = a_mtx.tocoo()
    u = th.tensor(coo.row, dtype=th.int64)
    v = th.tensor(coo.col, dtype=th.int64)
    g = dgl.DGLGraph()

    g.add_edges(u, v)

    n = g.num_nodes()
    feat_size = 128  ## arbitrary number
    feats = th.empty([n, feat_size], dtype=th.float32)

    ## arbitrary numbers
    train_size = 1000000
    test_size = 500000
    val_size = 5000
    nlabels = 256

    train_mask = th.zeros(n, dtype=th.bool)
    test_mask = th.zeros(n, dtype=th.bool)
    val_mask = th.zeros(n, dtype=th.bool)
    label = th.zeros(n, dtype=th.int64)

    for i in range(train_size):
        train_mask[i] = True

    for i in range(test_size):
        test_mask[train_size + i] = True

    for i in range(val_size):
        val_mask[train_size + test_size + i] = True

    for i in range(n):
        label[i] = random.choice(range(nlabels))

    g.ndata["feat"] = feats
    g.ndata["train_mask"] = train_mask
    g.ndata["test_mask"] = test_mask
    g.ndata["val_mask"] = val_mask
    g.ndata["label"] = label

    return g


def save(g, dataset):
    """
    This function saves input dataset to dgl format
    Parameters
    ----------
    g : graph to be saved
    dataset : output folder name
    """
    print("Saving dataset..")
    part_dir = os.path.join("./" + dataset)
    node_feat_file = os.path.join(part_dir, "node_feat.dgl")
    part_graph_file = os.path.join(part_dir, "graph.dgl")
    os.makedirs(part_dir, mode=0o775, exist_ok=True)
    save_tensors(node_feat_file, g.ndata)
    save_graphs(part_graph_file, [g])
    print("Graph saved successfully !!")


def load_proteins(dataset):
    """
    This function downloads, converts, and load Proteins graph dataset
    Parameter
    ---------
    dataset: output folder name
    """
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
