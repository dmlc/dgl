"""
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
import dgl
from dgl.distgnn.partition import partition_graph

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='cora')
    argparser.add_argument('--num-parts', type=int, default=2)
    argparser.add_argument('--out-dir', type=str, default='./')
    args = argparser.parse_args()

    dataset = args.dataset
    nc = args.num_parts
    prefix = args.out_dir

    partition_graph(dataset, nc, prefix)
