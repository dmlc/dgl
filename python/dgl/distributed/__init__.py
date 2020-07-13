"""DGL distributed."""
import os
import sys

from .dist_graph import DistGraphServer, DistGraph, DistTensor, node_split, edge_split
from .partition import partition_graph, load_partition, load_partition_book
from .graph_partition_book import GraphPartitionBook, RangePartitionBook, PartitionPolicy
from .sparse_emb import SparseAdagrad, SparseNodeEmbedding

from .rpc import *
from .rpc_server import start_server
from .rpc_client import connect_to_server, finalize_client, shutdown_servers
from .kvstore import KVServer, KVClient
from .server_state import ServerState
from .graph_services import sample_neighbors, in_subgraph

if os.environ.get('DGL_ROLE', 'client') == 'server':
    serv_id = int(os.environ.get('DGL_SERVER_ID'))
    ip_conf = os.environ.get('DGL_IP_CONFIG')
    conf_path = os.environ.get('DGL_CONF_PATH')
    num_clients = int(os.environ.get('DGL_NUM_CLIENT'))
    serv = DistGraphServer(serv_id, ip_conf, num_clients, conf_path)
    serv.start()
    sys.exit()
