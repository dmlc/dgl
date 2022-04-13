
import sys
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from read_nodes import include_recv_proc_nodes, include_recv_proc_edges, \
                        read_nodes_file, read_edge_file, \
                        read_node_features_file, read_edge_features_file
from read_metis_partitions import read_metis_partitions
from proc_init import run, init_process

def log_params(params): 
    print('Input Dir: ', params.input_dir)
    print('Graph Name: ', params.graph_name)
    print('Schema File: ', params.schema)
    print('No. partitions: ', params.num_parts)
    print('No. node weights: ', params.num_node_weights)
    print('Workspace dir: ', params.workspace)
    print('Node Attr Type: ', params.node_attr_dtype)
    print('Edge Attr Dtype: ', params.edge_attr_dtype)
    print('Output Dir: ', params.output)
    print('Removed Edges File: ', params.removed_edges)
    print('WorldSize: ', params.world_size)
    print('Nodes File: ', params.nodes_file)
    print('Edges File: ', params.edges_file)
    print('Node feats: ', params.node_feats_file)
    print('Edge feats: ', params.edge_feats_file)
    print('Metis partitions: ', params.metis_partitions)


def localRun(params): 
    log_params(params)
    size = params.world_size
    processes = []
    mp.set_start_method("spawn")

    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct graph partitions')
    parser.add_argument('--input-dir', required=True, type=str,
                     help='The directory path that contains the partition results.')
    parser.add_argument('--graph-name', required=True, type=str,
                     help='The graph name')
    parser.add_argument('--schema', required=True, type=str,
                     help='The schema of the graph')
    parser.add_argument('--num-parts', required=True, type=int,
                     help='The number of partitions')
    parser.add_argument('--num-node-weights', required=True, type=int,
                     help='The number of node weights used by METIS.')
    parser.add_argument('--workspace', type=str, default='/tmp',
                    help='The directory to store the intermediate results')
    parser.add_argument('--node-attr-dtype', type=str, default=None,
                    help='The data type of the node attributes')
    parser.add_argument('--edge-attr-dtype', type=str, default=None,
                    help='The data type of the edge attributes')
    parser.add_argument('--output', required=True, type=str,
                    help='The output directory of the partitioned results')
    parser.add_argument('--removed-edges', help='a file that contains the removed self-loops and duplicated edges',
                    default=None, type=str)

    parser.add_argument('--world-size', help='no. of processes to spawn',
                    default=1, type=int, required=True)
    parser.add_argument('--nodes-file', help='filename of the nodes metadata', 
                    default=None, type=str, required=True)
    parser.add_argument('--edges-file', help='filename of the nodes metadata', 
                    default=None, type=str, required=True)
    parser.add_argument('--node-feats-file', help='filename of the nodes features', 
                    default=None, type=str, required=True)
    parser.add_argument('--edge-feats-file', help='filename of the nodes metadata', 
                    default=None, type=str )
    parser.add_argument('--metis-partitions', help='filename of the output of dgl_part2 (metis partitions)',
                    default=None, type=str)
    params = parser.parse_args()

    localRun(params)
