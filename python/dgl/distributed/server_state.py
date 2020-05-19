# pylint: skip-file

"""Server data"""

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api

@register_object('server_state.ServerState')
class ServerState(ObjectBase):
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
    kv_store : dict[str, Tensor]
        Key value store for tensor data
    graph : DGLHeteroGraph
        Graph structure of one partition
    total_num_nodes : int
        Total number of nodes
    total_num_edges : int
        Total number of edges
    """
    @property
    def kv_store(self):
        """Get KV store."""
        return _CAPI_DGLRPCServerStateGetKVStore(self)

    @property
    def graph(self):
        """Get graph."""
        return _CAPI_DGLRPCServerStateGetGraph(self)

    @property
    def total_num_nodes(self):
        """Get total number of nodes."""
        return _CAPI_DGLRPCServerStateGetTotalNumNodes(self)

    @property
    def total_num_edges(self):
        """Get total number of edges."""
        return _CAPI_DGLRPCServerStateGetTotalNumEdges(self)

def get_server_state():
    """Get server state data.

    If the process is a server, this stores necessary
    server-side data. Otherwise, the process is a client and it stores a cache
    of the server co-located with the client (if available). When the client
    invokes a RPC to the co-located server, it can thus perform computation
    locally without an actual remote call.

    Returns
    -------
    ServerState
        Server state data
    """
    return _CAPI_DGLRPCGetServerState()

_init_api("dgl.distributed.server_state")
