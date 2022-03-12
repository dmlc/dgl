r"""
Copyright (c) 2021 Intel Corporation
 \file Graph partitioning
 \brief Calls Libra - Vertex-cut based graph partitioner for distirbuted training
 \author Vasimuddin Md <vasimuddin.md@intel.com>,
         Guixiang Ma <guixiang.ma@intel.com>
         Sanchit Misra <sanchit.misra@intel.com>,
         Ramanarayan Mohanty <ramanarayan.mohanty@intel.com>,
         Sasikanth Avancha <sasikanth.avancha@intel.com>
         Nesreen K. Ahmed <nesreen.k.ahmed@intel.com>
"""


import os
import sys
import numpy as np
import csv
from statistics import mean
import random
import time
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from load_graph import load_ogb
import dgl
from dgl.data import load_data
from dgl.distgnn.partition import partition_graph
from dgl.distgnn.tools import load_proteins
from dgl.base import DGLError


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num-parts', type=int, default=2)
    argparser.add_argument('--out-dir', type=str, default='./')
    args = argparser.parse_args()

    dataset = args.dataset
    num_community = args.num_parts
    out_dir = 'Libra_result_' + dataset   ## "Libra_result_" prefix is mandatory
    resultdir = os.path.join(args.out_dir, out_dir)

    print("Input dataset for partitioning: ", dataset)
    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        G, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        G, _ = load_ogb('ogbn-papers100M')
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

    partition_graph(num_community, G, resultdir)
