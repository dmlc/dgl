"""Server data"""

from .._ffi.function import _init_api

# Remove C++ bindings for now, since not used


class ServerState:
    """Data stored in one DGL server.

    In a distributed setting, DGL partitions all data associated with the graph
    (e.g., node and edge features, graph structure, etc.) to multiple partitions,
    each handled by one DGL server. Hence, the ServerState class includes all
    the data associated with a graph partition.

    Under some setup, users may want to deploy servers in a heterogeneous way
    -- servers are further divided into special groups for fetching/updating
    node/edge data and for sampling/querying on graph structure respectively.
    In this case, the ServerState can be configured to include only node/edge
    data or graph structure.

    Each machine can have multiple server and client processes, but only one
    server is the *master* server while all the others are backup servers. All
    clients and backup servers share the state of the master server via shared
    memory, which means the ServerState class must be serializable and large
    bulk data (e.g., node/edge features) must be stored in NDArray to leverage
    shared memory.

    Attributes
    ----------
    kv_store : KVServer
        reference for KVServer
    graph : DGLHeteroGraph
        Graph structure of one partition
    total_num_nodes : int
        Total number of nodes
    total_num_edges : int
        Total number of edges
    partition_book : GraphPartitionBook
        Graph Partition book
    """

    def __init__(self, kv_store, local_g, partition_book):
        self._kv_store = kv_store
        self._graph = local_g
        self.partition_book = partition_book
        self._roles = {}

    @property
    def roles(self):
        """Roles of the client processes"""
        return self._roles

    @property
    def kv_store(self):
        """Get data store."""
        return self._kv_store

    @kv_store.setter
    def kv_store(self, kv_store):
        self._kv_store = kv_store

    @property
    def graph(self):
        """Get graph data."""
        return self._graph

    @graph.setter
    def graph(self, graph):
        self._graph = graph


_init_api("dgl.distributed.server_state")
