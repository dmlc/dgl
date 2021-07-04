from pathlib import Path
import dgl
from dgl.distributed import partition_graph, load_partition, load_partition_book
import os
from dgl.distributed import DistGraph
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.distributed import sample_neighbors
import argparse

def start_client(machine_id):
    os.environ['DGL_DIST_MODE'] = 'distributed'
    datadir = Path(os.environ.get("PARTITION_DATA_BASE_PATH", "partition_data"))
    print(datadir)
    print(datadir / 'partition_data.json')
    _, _, _, gpb, _, _, _ = load_partition(datadir / 'partition_data.json', machine_id)
    dgl.distributed.initialize("ip_config.txt")
    dist_graph = DistGraph("partition_data", gpb=gpb)
    try:
        sampled_graph = sample_neighbors(dist_graph, [0, 10, 99, 66, 1024, 2008], 3)
    except Exception as e:
        print(e)
        sampled_graph = None
    print(sampled_graph)
    dgl.distributed.exit_client()
    return sampled_graph

parser = argparse.ArgumentParser()
parser.add_argument("--machine-id", type=int, help="Machine id")
args = parser.parse_args()
start_client(args.machine_id)