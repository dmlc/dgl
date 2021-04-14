"""DGL distributed module contains classes and functions to support
distributed graph neural network training and inference in a cluster of
machines.

This includes a few submodules:

* distributed data structures including distributed graph, distributed tensor
  and distributed embeddings.
* distributed sampling.
* distributed workload split at runtime.
* graph partition.

"""
import os
import sys

from .dist_graph import DistGraphServer, DistGraph, node_split, edge_split
from .dist_tensor import DistTensor
from .partition import partition_graph, load_partition, load_partition_book
from .graph_partition_book import GraphPartitionBook, PartitionPolicy
from .sparse_emb import SparseAdagrad, DistEmbedding

from .rpc import *
from .rpc_server import start_server
from .rpc_client import connect_to_server
from .dist_context import initialize, exit_client
from .kvstore import KVServer, KVClient
from .server_state import ServerState
from .dist_dataloader import DistDataLoader
from .graph_services import sample_neighbors, in_subgraph, find_edges
