"""DGL distributed."""
import os
import sys

from .dist_graph import DistGraphServer, DistGraph, DistTensor, node_split, edge_split
from .partition import partition_graph, load_partition, load_partition_book
from .graph_partition_book import GraphPartitionBook, RangePartitionBook, PartitionPolicy
from .sparse_emb import SparseAdagrad, DistEmbedding

from .rpc import *
from .rpc_server import start_server
from .rpc_client import connect_to_server, exit_client
from .kvstore import KVServer, KVClient
from .server_state import ServerState
from .graph_services import sample_neighbors, in_subgraph

if os.environ.get('DGL_ROLE', 'client') == 'server':
    assert os.environ.get('DGL_SERVER_ID') is not None, \
            'Please define DGL_SERVER_ID to run DistGraph server'
    assert os.environ.get('DGL_IP_CONFIG') is not None, \
            'Please define DGL_IP_CONFIG to run DistGraph server'
    assert os.environ.get('DGL_NUM_CLIENT') is not None, \
            'Please define DGL_NUM_CLIENT to run DistGraph server'
    assert os.environ.get('DGL_CONF_PATH') is not None, \
            'Please define DGL_CONF_PATH to run DistGraph server'
    SERV = DistGraphServer(int(os.environ.get('DGL_SERVER_ID')),
                           os.environ.get('DGL_IP_CONFIG'),
                           int(os.environ.get('DGL_NUM_CLIENT')),
                           os.environ.get('DGL_CONF_PATH'))
    SERV.start()
    sys.exit()
