import numpy as np
import argparse
import signal
import dgl
from dgl import backend as F
from dgl.data.utils import load_graphs, save_graphs
from dgl.contrib.dist_graph import partition_graph
import pickle

def main():
    parser = argparse.ArgumentParser(description='Partition a graph')
    parser.add_argument('--data', required=True, type=str,
                        help='The file path of the input graph in the DGL format.')
    parser.add_argument('--graph-name', required=True, type=str,
                        help='The graph name')
    parser.add_argument('-k', '--num-parts', required=True, type=int,
                        help='The number of partitions')
    parser.add_argument('--num-hops', type=int, default=1,
                        help='The number of hops of HALO nodes we include in a partition')
    parser.add_argument('-m', '--method', required=True, type=str,
                        help='The partitioning method: random, metis')
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='The output directory of the partitioned results')
    args = parser.parse_args()
    glist, _ = load_graphs(args.data)
    g = glist[0]
    partition_graph(g, args.graph_name, args.num_parts, args.output,
                    num_hops=args.num_hops, part_method=args.method)


if __name__ == '__main__':
    main()
