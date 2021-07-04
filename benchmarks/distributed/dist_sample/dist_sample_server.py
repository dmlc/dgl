from pathlib import Path
import dgl
from dgl.distributed import partition_graph, load_partition, load_partition_book
import os
from dgl.distributed.dist_graph import DistGraphServer
from ogb.nodeproppred import DglNodePropPredDataset


def start_server(server_id, num_servers, num_clients):
    folder_name = os.environ.get("PARTITION_DATA_BASE_PATH", "partition_data")
    g = DistGraphServer(server_id, "ip_config.txt", num_servers, num_clients,
                        Path(folder_name) / 'partition_data.json',
                        graph_format='csc')
    g.start()