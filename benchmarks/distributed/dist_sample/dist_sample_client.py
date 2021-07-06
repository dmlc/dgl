from pathlib import Path
import dgl
from dgl.distributed import partition_graph, load_partition, load_partition_book
import os
from dgl.distributed import DistGraph
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.distributed import sample_neighbors
import argparse
import numpy as np
import time


def start_client(machine_id):
    os.environ['DGL_DIST_MODE'] = 'distributed'
    datadir = Path(os.environ.get(
        "PARTITION_DATA_BASE_PATH", "partition_data"))
    _, _, _, gpb, _, _, _ = load_partition(
        datadir / 'partition_data.json', machine_id)
    dgl.distributed.initialize("ip_config.txt")

    dist_graph = DistGraph("partition_data", gpb=gpb)
    print("Finish initialize client")

    num_nodes = dist_graph.num_nodes()
    sample_batch_size = 2048
    fanout = 3
    num_iterations = 200
    # dry run
    for i in range(50):
        seed_nodes = np.random.randint(0, num_nodes, sample_batch_size)
        sampled_graph = sample_neighbors(dist_graph, seed_nodes, fanout)

    seed_nodes = np.random.randint(0, num_nodes, (200, sample_batch_size))
    t0 = time.time()
    for i in range(num_iterations):
        sampled_graph = sample_neighbors(dist_graph, seed_nodes[i], fanout)
    t1 = time.time()
    per_iter_time = (t1-t0)/num_iterations
    print(f"Time: {per_iter_time}")
    dgl.distributed.exit_client()
    return per_iter_time


parser = argparse.ArgumentParser()
parser.add_argument("--machine-id", type=int, help="Machine id")
args = parser.parse_args()
start_client(args.machine_id)
