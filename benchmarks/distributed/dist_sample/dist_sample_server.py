from pathlib import Path
import dgl
from dgl.distributed import partition_graph, load_partition, load_partition_book
import os
from dgl.distributed.dist_graph import DistGraphServer
from ogb.nodeproppred import DglNodePropPredDataset
import argparse

def start_server(server_id, num_servers, num_clients):
    os.environ['DGL_DIST_MODE'] = 'distributed'
    folder_name = os.environ.get("PARTITION_DATA_BASE_PATH", "partition_data")
    
    print("Before construct graph")
    g = DistGraphServer(server_id, "ip_config.txt", num_servers, num_clients,
                        Path(folder_name) / 'partition_data.json',
                        graph_format='csc')
    print("After construct graph")
    g.start()


parser = argparse.ArgumentParser()
parser.add_argument("--server-id", type=int, help="Machine id")
parser.add_argument("--num-servers", type=int, help="Number of servers per machine")
parser.add_argument("--num-clients", type=int, help="Number of clients")
args = parser.parse_args()
start_server(args.server_id, args.num_servers, args.num_clients)