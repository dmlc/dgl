from pathlib import Path
import dgl
from dgl.distributed import partition_graph, load_partition, load_partition_book
import os
from dgl.distributed.dist_graph import DistGraphServer
from ogb.nodeproppred import DglNodePropPredDataset
import argparse

def gen_partition(num_parts, num_hops=1, reshuffle=True):
    # os.symlink('/tmp/dataset/', os.path.join(os.getcwd(), 'dataset'))
    data = DglNodePropPredDataset(name="ogbn-products")
    g = data[0][0]
    num_hops = 1
    folder_name = 'partition_data'
    partition_graph(g, folder_name, num_parts, "{}/".format(folder_name),
                    num_hops=num_hops, part_method='metis', reshuffle=reshuffle)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-parts", type=int, help="Number of partitions")
    args = parser.parse_args()
    gen_partition(args.num_parts)
