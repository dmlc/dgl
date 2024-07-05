"""Define distributed graph."""

import gc

import os
from collections import namedtuple
from collections.abc import Mapping, MutableMapping

import numpy as np
import torch

from .. import backend as F, graphbolt as gb, heterograph_index
from .._ffi.ndarray import empty_shared_mem
from ..base import ALL, DGLError, EID, ETYPE, is_all, NID
from ..convert import graph as dgl_graph, heterograph as dgl_heterograph
from ..frame import infer_scheme

from ..heterograph import DGLGraph
from ..ndarray import exist_shared_mem_array
from ..transforms import compact_graphs
from . import graph_services, role, rpc
from .dist_tensor import DistTensor
from .graph_partition_book import (
    _etype_str_to_tuple,
    EdgePartitionPolicy,
    get_shared_mem_partition_book,
    HeteroDataName,
    NodePartitionPolicy,
    parse_hetero_data_name,
    PartitionPolicy,
)
from .graph_services import (
    find_edges as dist_find_edges,
    in_degrees as dist_in_degrees,
    out_degrees as dist_out_degrees,
)
from .kvstore import get_kvstore, KVServer
from .partition import (
    load_partition,
    load_partition_book,
    load_partition_feats,
    RESERVED_FIELD_DTYPE,
)
from .rpc_server import start_server
from .server_state import ServerState
from .shared_mem_utils import (
    _get_edata_path,
    _get_ndata_path,
    _to_shared_mem,
    DTYPE_DICT,
)

INIT_GRAPH = 800001
QUERY_IF_USE_GRAPHBOLT = 800002
ADD_EDGE_ATTRIBUTE_FROM_KV = 800003
ADD_EDGE_ATTRIBUTE_FROM_SHARED_MEM = 800004


class InitGraphRequest(rpc.Request):
    """Init graph on the backup servers.

    When the backup server starts, they don't load the graph structure.
    This request tells the backup servers that they can map to the graph structure
    with shared memory.
    """

    def __init__(self, graph_name):
        self._graph_name = graph_name

    def __getstate__(self):
        return self._graph_name

    def __setstate__(self, state):
        self._graph_name = state

    def process_request(self, server_state):
        if server_state.graph is None:
            server_state.graph = _get_graph_from_shared_mem(
                self._graph_name, server_state.use_graphbolt
            )
        return InitGraphResponse(self._graph_name)


class InitGraphResponse(rpc.Response):
    """Ack the init graph request"""

    def __init__(self, graph_name):
        self._graph_name = graph_name

    def __getstate__(self):
        return self._graph_name

    def __setstate__(self, state):
        self._graph_name = state


class QueryIfUseGraphBoltRequest(rpc.Request):
    """Query if use GraphBolt."""

    def __getstate__(self):
        return None

    def __setstate__(self, state):
        pass

    def process_request(self, server_state):
        return QueryIfUseGraphBoltResponse(server_state.use_graphbolt)


class QueryIfUseGraphBoltResponse(rpc.Response):
    """Ack the query request about if use GraphBolt."""

    def __init__(self, use_graphbolt):
        self._use_graphbolt = use_graphbolt

    def __getstate__(self):
        return self._use_graphbolt

    def __setstate__(self, state):
        self._use_graphbolt = state


def _copy_data_to_shared_mem(data, name):
    """Copy data to shared memory."""
    # [TODO] Copy data to shared memory.
    assert data.dtype == torch.float32, "Only float32 is supported."
    data_type = F.reverse_data_type_dict[F.dtype(data)]
    shared_data = empty_shared_mem(name, True, data.shape, data_type)
    dlpack = shared_data.to_dlpack()
    ret = F.zerocopy_from_dlpack(dlpack)
    rpc.copy_data_to_shared_memory(ret, data)
    return ret


def _copy_data_from_shared_mem(name, shape):
    """Copy data from shared memory."""
    data_type = F.reverse_data_type_dict[F.float32]
    data = empty_shared_mem(name, False, shape, data_type)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


class AddEdgeAttributeFromKVRequest(rpc.Request):
    """Add edge attribute from kvstore to local GraphBolt partition."""

    def __init__(self, name, kv_names):
        self._name = name
        self._kv_names = kv_names

    def __getstate__(self):
        return self._name, self._kv_names

    def __setstate__(self, state):
        self._name, self._kv_names = state

    def process_request(self, server_state):
        # For now, this is only used to add prob/mask data to the graph.
        name = self._name
        g = server_state.graph
        if name not in g.edge_attributes:
            # Fetch target data from kvstore.
            kv_store = server_state.kv_store
            data = [
                kv_store.data_store[kv_name] if kv_name else None
                for kv_name in self._kv_names
            ]
            # Due to data type limitation in GraphBolt's sampling, we only support float32.
            data_type = torch.float32
            gpb = server_state.partition_book
            # Initialize the edge attribute.
            num_edges = g.total_num_edges
            attr_data = torch.zeros(num_edges, dtype=data_type)
            # Map data from kvstore to the local partition for inner edges only.
            num_inner_edges = gpb.metadata()[gpb.partid]["num_edges"]
            homo_eids = g.edge_attributes[EID][:num_inner_edges]
            etype_ids, typed_eids = gpb.map_to_per_etype(homo_eids)
            for etype_id, c_etype in enumerate(gpb.canonical_etypes):
                curr_indices = torch.nonzero(etype_ids == etype_id).squeeze()
                curr_typed_eids = typed_eids[curr_indices]
                curr_local_eids = gpb.eid2localeid(
                    curr_typed_eids, gpb.partid, etype=c_etype
                )
                if data[etype_id] is None:
                    continue
                attr_data[curr_indices] = data[etype_id][curr_local_eids].to(
                    data_type
                )
            # Copy data to shared memory.
            attr_data = _copy_data_to_shared_mem(attr_data, "__edge__" + name)
            g.add_edge_attribute(name, attr_data)
        return AddEdgeAttributeFromKVResponse(name)


class AddEdgeAttributeFromKVResponse(rpc.Response):
    """Ack the request of adding edge attribute."""

    def __init__(self, name):
        self._name = name

    def __getstate__(self):
        return self._name

    def __setstate__(self, state):
        self._name = state


class AddEdgeAttributeFromSharedMemRequest(rpc.Request):
    """Add edge attribute from shared memory to local GraphBolt partition."""

    def __init__(self, name):
        self._name = name

    def __getstate__(self):
        return self._name

    def __setstate__(self, state):
        self._name = state

    def process_request(self, server_state):
        name = self._name
        g = server_state.graph
        if name not in g.edge_attributes:
            data = _copy_data_from_shared_mem(
                "__edge__" + name, (g.total_num_edges,)
            )
            g.add_edge_attribute(name, data)
        return AddEdgeAttributeFromSharedMemResponse(name)


class AddEdgeAttributeFromSharedMemResponse(rpc.Response):
    """Ack the request of adding edge attribute from shared memory."""

    def __init__(self, name):
        self._name = name

    def __getstate__(self):
        return self._name

    def __setstate__(self, state):
        self._name = state


def _copy_graph_to_shared_mem(g, graph_name, graph_format, use_graphbolt):
    if use_graphbolt:
        return g.copy_to_shared_memory(graph_name)
    new_g = g.shared_memory(graph_name, formats=graph_format)
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    new_g.ndata["inner_node"] = _to_shared_mem(
        g.ndata["inner_node"], _get_ndata_path(graph_name, "inner_node")
    )
    new_g.ndata[NID] = _to_shared_mem(
        g.ndata[NID], _get_ndata_path(graph_name, NID)
    )

    new_g.edata["inner_edge"] = _to_shared_mem(
        g.edata["inner_edge"], _get_edata_path(graph_name, "inner_edge")
    )
    new_g.edata[EID] = _to_shared_mem(
        g.edata[EID], _get_edata_path(graph_name, EID)
    )
    # for heterogeneous graph, we need to put ETYPE into KVStore
    # for homogeneous graph, ETYPE does not exist
    if ETYPE in g.edata:
        new_g.edata[ETYPE] = _to_shared_mem(
            g.edata[ETYPE],
            _get_edata_path(graph_name, ETYPE),
        )
    return new_g


def _get_shared_mem_ndata(g, graph_name, name):
    """Get shared-memory node data from DistGraph server.

    This is called by the DistGraph client to access the node data in the DistGraph server
    with shared memory.
    """
    shape = (g.num_nodes(),)
    dtype = RESERVED_FIELD_DTYPE[name]
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(
        _get_ndata_path(graph_name, name), False, shape, dtype
    )
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


def _get_shared_mem_edata(g, graph_name, name):
    """Get shared-memory edge data from DistGraph server.

    This is called by the DistGraph client to access the edge data in the DistGraph server
    with shared memory.
    """
    shape = (g.num_edges(),)
    dtype = RESERVED_FIELD_DTYPE[name]
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(
        _get_edata_path(graph_name, name), False, shape, dtype
    )
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)


def _exist_shared_mem_array(graph_name, name):
    return exist_shared_mem_array(_get_edata_path(graph_name, name))


def _get_graph_from_shared_mem(graph_name, use_graphbolt):
    """Get the graph from the DistGraph server.

    The DistGraph server puts the graph structure of the local partition in the shared memory.
    The client can access the graph structure and some metadata on nodes and edges directly
    through shared memory to reduce the overhead of data access.
    """
    if use_graphbolt:
        return gb.load_from_shared_memory(graph_name)
    g, ntypes, etypes = heterograph_index.create_heterograph_from_shared_memory(
        graph_name
    )
    if g is None:
        return None
    g = DGLGraph(g, ntypes, etypes)

    g.ndata["inner_node"] = _get_shared_mem_ndata(g, graph_name, "inner_node")
    g.ndata[NID] = _get_shared_mem_ndata(g, graph_name, NID)

    g.edata["inner_edge"] = _get_shared_mem_edata(g, graph_name, "inner_edge")
    g.edata[EID] = _get_shared_mem_edata(g, graph_name, EID)

    # heterogeneous graph has ETYPE
    if _exist_shared_mem_array(graph_name, ETYPE):
        g.edata[ETYPE] = _get_shared_mem_edata(g, graph_name, ETYPE)
    return g


NodeSpace = namedtuple("NodeSpace", ["data"])
EdgeSpace = namedtuple("EdgeSpace", ["data"])


class HeteroNodeView(object):
    """A NodeView class to act as G.nodes for a DistGraph."""

    __slots__ = ["_graph"]

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        assert isinstance(key, str)
        return NodeSpace(data=NodeDataView(self._graph, key))


class HeteroEdgeView(object):
    """An EdgeView class to act as G.edges for a DistGraph."""

    __slots__ = ["_graph"]

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        assert isinstance(key, str) or (
            isinstance(key, tuple) and len(key) == 3
        ), f"Expect edge type in string or triplet of string, but got {key}."
        return EdgeSpace(data=EdgeDataView(self._graph, key))


class NodeDataView(MutableMapping):
    """The data view class when dist_graph.ndata[...].data is called."""

    __slots__ = ["_graph", "_data"]

    def __init__(self, g, ntype=None):
        self._graph = g
        if ntype is None or len(g.ntypes) == 1:
            self._data = g._ndata_store
        else:
            if ntype not in g.ntypes:
                raise DGLError(f"Node type {ntype} does not exist.")
            self._data = g._ndata_store[ntype]

    def _get_names(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        self._data[key] = val

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        # The number of node data may change. Let's count it every time we need them.
        # It's not called frequently. It should be fine.
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        reprs = {}
        for name in self._data:
            dtype = F.dtype(self._data[name])
            shape = F.shape(self._data[name])
            reprs[name] = "DistTensor(shape={}, dtype={})".format(
                str(shape), str(dtype)
            )
        return repr(reprs)


class EdgeDataView(MutableMapping):
    """The data view class when G.edges[...].data is called."""

    __slots__ = ["_graph", "_data"]

    def __init__(self, g, etype=None):
        self._graph = g
        if etype is None or len(g.canonical_etypes) == 1:
            self._data = g._edata_store
        else:
            c_etype = g.to_canonical_etype(etype)
            self._data = g._edata_store[c_etype]

    def _get_names(self):
        return list(self._data.keys())

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        self._data[key] = val

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        # The number of edge data may change. Let's count it every time we need them.
        # It's not called frequently. It should be fine.
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __repr__(self):
        reprs = {}
        for name in self._data:
            dtype = F.dtype(self._data[name])
            shape = F.shape(self._data[name])
            reprs[name] = "DistTensor(shape={}, dtype={})".format(
                str(shape), str(dtype)
            )
        return repr(reprs)


def _format_partition(graph, graph_format):
    """Format the partition to the specified format."""
    if isinstance(graph, gb.FusedCSCSamplingGraph):
        return graph
    # formatting dtype
    # TODO(Rui) Formatting forcely is not a perfect solution.
    #   We'd better store all dtypes when mapping to shared memory
    #   and map back with original dtypes.
    for k, dtype in RESERVED_FIELD_DTYPE.items():
        if k in graph.ndata:
            graph.ndata[k] = F.astype(graph.ndata[k], dtype)
        if k in graph.edata:
            graph.edata[k] = F.astype(graph.edata[k], dtype)
    # Create the graph formats specified the users.
    print(
        "Start to create specified graph formats which may take "
        "non-trivial time."
    )
    graph = graph.formats(graph_format)
    graph.create_formats_()
    print(f"Finished creating specified graph formats: {graph_format}")
    return graph


class DistGraphServer(KVServer):
    """The DistGraph server.

    This DistGraph server loads the graph data and sets up a service so that trainers and
    samplers can read data of a graph partition (graph structure, node data and edge data)
    from remote machines. A server is responsible for one graph partition.

    Currently, each machine runs only one main server with a set of backup servers to handle
    clients' requests. The main server and the backup servers all handle the requests for the same
    graph partition. They all share the partition data (graph structure and node/edge data) with
    shared memory.

    By default, the partition data is shared with the DistGraph clients that run on
    the same machine. However, a user can disable shared memory option. This is useful for the case
    that a user wants to run the server and the client on different machines.

    Parameters
    ----------
    server_id : int
        The server ID (start from 0).
    ip_config : str
        Path of IP configuration file.
    num_servers : int
        Server count on each machine.
    num_clients : int
        Total number of client nodes.
    part_config : string
        The path of the config file generated by the partition tool.
    disable_shared_mem : bool
        Disable shared memory.
    graph_format : str or list of str
        The graph formats.
    use_graphbolt : bool
        Whether to load GraphBolt partition. Default: False.
    """

    def __init__(
        self,
        server_id,
        ip_config,
        num_servers,
        num_clients,
        part_config,
        disable_shared_mem=False,
        graph_format=("csc", "coo"),
        use_graphbolt=False,
    ):
        super(DistGraphServer, self).__init__(
            server_id=server_id,
            ip_config=ip_config,
            num_servers=num_servers,
            num_clients=num_clients,
        )
        self.ip_config = ip_config
        self.num_servers = num_servers
        self.use_graphbolt = use_graphbolt
        # Load graph partition data.
        if self.is_backup_server():
            # The backup server doesn't load the graph partition. It'll initialized afterwards.
            self.gpb, graph_name, ntypes, etypes = load_partition_book(
                part_config, self.part_id
            )
            self.client_g = None
        else:
            # Loading of node/edge_feats are deferred to lower the peak memory consumption.
            (
                self.client_g,
                _,
                _,
                self.gpb,
                graph_name,
                ntypes,
                etypes,
            ) = load_partition(
                part_config,
                self.part_id,
                load_feats=False,
                use_graphbolt=use_graphbolt,
            )
            print("load " + graph_name)
            self.client_g = _format_partition(self.client_g, graph_format)
            if not disable_shared_mem:
                self.client_g = _copy_graph_to_shared_mem(
                    self.client_g, graph_name, graph_format, use_graphbolt
                )

        if not disable_shared_mem:
            self.gpb.shared_memory(graph_name)
        assert self.gpb.partid == self.part_id
        for ntype in ntypes:
            node_name = HeteroDataName(True, ntype, "")
            self.add_part_policy(
                PartitionPolicy(node_name.policy_str, self.gpb)
            )
        for etype in etypes:
            edge_name = HeteroDataName(False, etype, "")
            self.add_part_policy(
                PartitionPolicy(edge_name.policy_str, self.gpb)
            )

        if not self.is_backup_server():
            node_feats, _ = load_partition_feats(
                part_config, self.part_id, load_nodes=True, load_edges=False
            )
            for name in node_feats:
                # The feature name has the following format: node_type + "/" + feature_name to avoid
                # feature name collision for different node types.
                ntype, feat_name = name.split("/")
                data_name = HeteroDataName(True, ntype, feat_name)
                self.init_data(
                    name=str(data_name),
                    policy_str=data_name.policy_str,
                    data_tensor=node_feats[name],
                )
                self.orig_data.add(str(data_name))
            # Let's free once node features are copied to shared memory
            del node_feats
            gc.collect()
            _, edge_feats = load_partition_feats(
                part_config, self.part_id, load_nodes=False, load_edges=True
            )
            for name in edge_feats:
                # The feature name has the following format: edge_type + "/" + feature_name to avoid
                # feature name collision for different edge types.
                etype, feat_name = name.split("/")
                etype = _etype_str_to_tuple(etype)
                data_name = HeteroDataName(False, etype, feat_name)
                self.init_data(
                    name=str(data_name),
                    policy_str=data_name.policy_str,
                    data_tensor=edge_feats[name],
                )
                self.orig_data.add(str(data_name))
            # Let's free once edge features are copied to shared memory
            del edge_feats
            gc.collect()

    def start(self):
        """Start graph store server."""
        # start server
        server_state = ServerState(
            kv_store=self,
            local_g=self.client_g,
            partition_book=self.gpb,
            use_graphbolt=self.use_graphbolt,
        )
        print(
            "start graph service on server {} for part {}".format(
                self.server_id, self.part_id
            )
        )
        start_server(
            server_id=self.server_id,
            ip_config=self.ip_config,
            num_servers=self.num_servers,
            num_clients=self.num_clients,
            server_state=server_state,
        )


class DistGraph:
    """The class for accessing a distributed graph.

    This class provides a subset of DGLGraph APIs for accessing partitioned graph data in
    distributed GNN training and inference. Thus, its main use case is to work with
    distributed sampling APIs to generate mini-batches and perform forward and
    backward computation on the mini-batches.

    The class can run in two modes: the standalone mode and the distributed mode.

    * When a user runs the training script normally, ``DistGraph`` will be in the standalone mode.
      In this mode, the input data must be constructed by
      :py:meth:`~dgl.distributed.partition.partition_graph` with only one partition. This mode is
      used for testing and debugging purpose. In this mode, users have to provide ``part_config``
      so that ``DistGraph`` can load the input graph.
    * When a user runs the training script with the distributed launch script, ``DistGraph`` will
      be set into the distributed mode. This is used for actual distributed training. All data of
      partitions are loaded by the ``DistGraph`` servers, which are created by DGL's launch script.
      ``DistGraph`` connects with the servers to access the partitioned graph data.

    Currently, the ``DistGraph`` servers and clients run on the same set of machines
    in the distributed mode. ``DistGraph`` uses shared-memory to access the partition data
    in the local machine. This gives the best performance for distributed training

    Users may want to run ``DistGraph`` servers and clients on separate sets of machines.
    In this case, a user may want to disable shared memory by passing
    ``disable_shared_mem=False`` when creating ``DistGraphServer``. When shared memory is disabled,
    a user has to pass a partition book.

    Parameters
    ----------
    graph_name : str
        The name of the graph. This name has to be the same as the one used for
        partitioning a graph in :py:meth:`dgl.distributed.partition.partition_graph`.
    gpb : GraphPartitionBook, optional
        The partition book object. Normally, users do not need to provide the partition book.
        This argument is necessary only when users want to run server process and trainer
        processes on different machines.
    part_config : str, optional
        The path of partition configuration file generated by
        :py:meth:`dgl.distributed.partition.partition_graph`. It's used in the standalone mode.

    Examples
    --------
    The example shows the creation of ``DistGraph`` in the standalone mode.

    >>> dgl.distributed.partition_graph(g, 'graph_name', 1, num_hops=1, part_method='metis',
    ...                                 out_path='output/')
    >>> g = dgl.distributed.DistGraph('graph_name', part_config='output/graph_name.json')

    The example shows the creation of ``DistGraph`` in the distributed mode.

    >>> g = dgl.distributed.DistGraph('graph-name')

    The code below shows the mini-batch training using ``DistGraph``.

    >>> def sample(seeds):
    ...     seeds = th.LongTensor(np.asarray(seeds))
    ...     frontier = dgl.distributed.sample_neighbors(g, seeds, 10)
    ...     return dgl.to_block(frontier, seeds)
    >>> dataloader = dgl.distributed.DistDataLoader(dataset=nodes, batch_size=1000,
    ...                                             collate_fn=sample, shuffle=True)
    >>> for block in dataloader:
    ...     feat = g.ndata['features'][block.srcdata[dgl.NID]]
    ...     labels = g.ndata['labels'][block.dstdata[dgl.NID]]
    ...     pred = model(block, feat)

    Note
    ----
    DGL's distributed training by default runs server processes and trainer processes on the same
    set of machines. If users need to run them on different sets of machines, it requires
    manually setting up servers and trainers. The setup is not fully tested yet.
    """

    def __init__(self, graph_name, gpb=None, part_config=None):
        self.graph_name = graph_name
        self._added_edge_attributes = []  # For prob/mask sampling on GB.
        if os.environ.get("DGL_DIST_MODE", "standalone") == "standalone":
            # "GraphBolt is not supported in standalone mode."
            self._use_graphbolt = False
            assert (
                part_config is not None
            ), "When running in the standalone model, the partition config file is required"
            self._client = get_kvstore()
            assert (
                self._client is not None
            ), "Distributed module is not initialized. Please call dgl.distributed.initialize."
            # Load graph partition data.
            g, node_feats, edge_feats, self._gpb, _, _, _ = load_partition(
                part_config, 0
            )
            assert (
                self._gpb.num_partitions() == 1
            ), "The standalone mode can only work with the graph data with one partition"
            if self._gpb is None:
                self._gpb = gpb
            self._g = g
            for name in node_feats:
                # The feature name has the following format: node_type + "/" + feature_name.
                ntype, feat_name = name.split("/")
                self._client.add_data(
                    str(HeteroDataName(True, ntype, feat_name)),
                    node_feats[name],
                    NodePartitionPolicy(self._gpb, ntype=ntype),
                )
            for name in edge_feats:
                # The feature name has the following format: edge_type + "/" + feature_name.
                etype, feat_name = name.split("/")
                etype = _etype_str_to_tuple(etype)
                self._client.add_data(
                    str(HeteroDataName(False, etype, feat_name)),
                    edge_feats[name],
                    EdgePartitionPolicy(self._gpb, etype=etype),
                )
            self._client.map_shared_data(self._gpb)
            rpc.set_num_client(1)
        else:
            # Query the main server about whether GraphBolt is used.
            rpc.send_request(0, QueryIfUseGraphBoltRequest())
            self._use_graphbolt = rpc.recv_response()._use_graphbolt

            self._init(gpb)
            # Tell the backup servers to load the graph structure from shared memory.
            for server_id in range(self._client.num_servers):
                rpc.send_request(server_id, InitGraphRequest(graph_name))
            for server_id in range(self._client.num_servers):
                rpc.recv_response()
            self._client.barrier()

        self._init_ndata_store()
        self._init_edata_store()
        self._init_metadata()

    def _init(self, gpb):
        self._client = get_kvstore()
        assert (
            self._client is not None
        ), "Distributed module is not initialized. Please call dgl.distributed.initialize."
        self._g = _get_graph_from_shared_mem(
            self.graph_name, self._use_graphbolt
        )
        self._gpb = get_shared_mem_partition_book(self.graph_name)
        if self._gpb is None:
            self._gpb = gpb
        self._client.map_shared_data(self._gpb)

    def _init_ndata_store(self):
        """Initialize node data store."""
        self._ndata_store = {}
        for ntype in self.ntypes:
            names = self._get_ndata_names(ntype)
            data = {}
            for name in names:
                assert name.is_node()
                policy = PartitionPolicy(
                    name.policy_str, self.get_partition_book()
                )
                dtype, shape, _ = self._client.get_data_meta(str(name))
                # We create a wrapper on the existing tensor in the kvstore.
                data[name.get_name()] = DistTensor(
                    shape,
                    dtype,
                    name.get_name(),
                    part_policy=policy,
                    attach=False,
                )
            if len(self.ntypes) == 1:
                self._ndata_store = data
            else:
                self._ndata_store[ntype] = data

    def _init_edata_store(self):
        """Initialize edge data store."""
        self._edata_store = {}
        for etype in self.canonical_etypes:
            names = self._get_edata_names(etype)
            data = {}
            for name in names:
                assert name.is_edge()
                policy = PartitionPolicy(
                    name.policy_str, self.get_partition_book()
                )
                dtype, shape, _ = self._client.get_data_meta(str(name))
                # We create a wrapper on the existing tensor in the kvstore.
                data[name.get_name()] = DistTensor(
                    shape,
                    dtype,
                    name.get_name(),
                    part_policy=policy,
                    attach=False,
                )
            if len(self.canonical_etypes) == 1:
                self._edata_store = data
            else:
                self._edata_store[etype] = data

    def _init_metadata(self):
        self._num_nodes = 0
        self._num_edges = 0
        for part_md in self._gpb.metadata():
            self._num_nodes += int(part_md["num_nodes"])
            self._num_edges += int(part_md["num_edges"])

        # When we store node/edge types in a list, they are stored in the order of type IDs.
        self._ntype_map = {ntype: i for i, ntype in enumerate(self.ntypes)}
        self._etype_map = {
            etype: i for i, etype in enumerate(self.canonical_etypes)
        }

    def __getstate__(self):
        return (
            self.graph_name,
            self._gpb,
            self._use_graphbolt,
            self._added_edge_attributes,
        )

    def __setstate__(self, state):
        (
            self.graph_name,
            gpb,
            self._use_graphbolt,
            self._added_edge_attributes,
        ) = state
        self._init(gpb)

        self._init_ndata_store()
        self._init_edata_store()
        self._init_metadata()

        # For prob/mask sampling on GB only.
        if self._use_graphbolt and len(self._added_edge_attributes) > 0:
            # Add edge attribute from main server's shared memory.
            for name in self._added_edge_attributes:
                data = _copy_data_from_shared_mem(
                    "__edge__" + name, (self.local_partition.total_num_edges,)
                )
                self.local_partition.add_edge_attribute(name, data)

    @property
    def local_partition(self):
        """Return the local partition on the client

        DistGraph provides a global view of the distributed graph. Internally,
        it may contains a partition of the graph if it is co-located with
        the server. When servers and clients run on separate sets of machines,
        this returns None.

        Returns
        -------
        DGLGraph
            The local partition
        """
        return self._g

    @property
    def nodes(self):
        """Return a node view"""
        return HeteroNodeView(self)

    @property
    def edges(self):
        """Return an edge view"""
        return HeteroEdgeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        Returns
        -------
        NodeDataView
            The data view in the distributed graph storage.
        """
        assert (
            len(self.ntypes) == 1
        ), "ndata only works for a graph with one node type."
        return NodeDataView(self)

    @property
    def edata(self):
        """Return the data view of all the edges.

        Returns
        -------
        EdgeDataView
            The data view in the distributed graph storage.
        """
        assert (
            len(self.etypes) == 1
        ), "edata only works for a graph with one edge type."
        return EdgeDataView(self)

    @property
    def idtype(self):
        """The dtype of graph index

        Returns
        -------
        backend dtype object
            th.int32/th.int64 or tf.int32/tf.int64 etc.

        See Also
        --------
        long
        int
        """
        # TODO(da?): describe when self._g is None and idtype shouldn't be called.
        # For GraphBolt partition, we use the global node ID's dtype.
        return (
            self.get_partition_book().global_nid_dtype
            if self._use_graphbolt
            else F.int64
        )

    @property
    def device(self):
        """Get the device context of this graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1])
        ... })
        >>> print(g.device)
        device(type='cpu')
        >>> g = g.to('cuda:0')
        >>> print(g.device)
        device(type='cuda', index=0)

        Returns
        -------
        Device context object
        """
        # TODO(da?): describe when self._g is None and device shouldn't be called.
        return F.cpu()

    def is_pinned(self):
        """Check if the graph structure is pinned to the page-locked memory.

        Returns
        -------
        bool
            True if the graph structure is pinned.
        """
        # (Xin Yao): Currently we don't support pinning a DistGraph.
        return False

    @property
    def ntypes(self):
        """Return the list of node types of this graph.

        Returns
        -------
        list of str

        Examples
        --------

        >>> g = DistGraph("test")
        >>> g.ntypes
        ['_U']
        """
        return self._gpb.ntypes

    @property
    def etypes(self):
        """Return the list of edge types of this graph.

        Returns
        -------
        list of str

        Examples
        --------

        >>> g = DistGraph("test")
        >>> g.etypes
        ['_E']
        """
        return self._gpb.etypes

    @property
    def canonical_etypes(self):
        """Return all the canonical edge types in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        Returns
        -------
        list[(str, str, str)]
            All the canonical edge type triplets in a list.

        Notes
        -----
        DGL internally assigns an integer ID for each edge type. The returned
        edge type names are sorted according to their IDs.

        See Also
        --------
        etypes

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = DistGraph("test")
        >>> g.canonical_etypes
        [('user', 'follows', 'user'),
         ('user', 'follows', 'game'),
         ('user', 'plays', 'game')]
        """
        return self._gpb.canonical_etypes

    def to_canonical_etype(self, etype):
        """Convert an edge type to the corresponding canonical edge type in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        The function expects the given edge type name can uniquely identify a canonical edge
        type. DGL will raise error if this is not the case.

        Parameters
        ----------
        etype : str or (str, str, str)
            If :attr:`etype` is an edge type (str), it returns the corresponding canonical edge
            type in the graph. If :attr:`etype` is already a canonical edge type,
            it directly returns the input unchanged.

        Returns
        -------
        (str, str, str)
            The canonical edge type corresponding to the edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = DistGraph("test")
        >>> g.canonical_etypes
        [('user', 'follows', 'user'),
         ('user', 'follows', 'game'),
         ('user', 'plays', 'game')]

        >>> g.to_canonical_etype('plays')
        ('user', 'plays', 'game')
        >>> g.to_canonical_etype(('user', 'plays', 'game'))
        ('user', 'plays', 'game')

        See Also
        --------
        canonical_etypes
        """
        return self._gpb.to_canonical_etype(etype)

    def get_ntype_id(self, ntype):
        """Return the ID of the given node type.

        ntype can also be None. If so, there should be only one node type in the
        graph.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if len(self._ntype_map) != 1:
                raise DGLError(
                    "Node type name must be specified if there are more than one "
                    "node types."
                )
            return 0
        return self._ntype_map[ntype]

    def get_etype_id(self, etype):
        """Return the id of the given edge type.

        etype can also be None. If so, there should be only one edge type in the
        graph.

        Parameters
        ----------
        etype : str or tuple of str
            Edge type

        Returns
        -------
        int
        """
        if etype is None:
            if len(self._etype_map) != 1:
                raise DGLError(
                    "Edge type name must be specified if there are more than one "
                    "edge types."
                )
            return 0
        etype = self.to_canonical_etype(etype)
        return self._etype_map[etype]

    def number_of_nodes(self, ntype=None):
        """Alias of :func:`num_nodes`"""
        return self.num_nodes(ntype)

    def number_of_edges(self, etype=None):
        """Alias of :func:`num_edges`"""
        return self.num_edges(etype)

    def num_nodes(self, ntype=None):
        """Return the total number of nodes in the distributed graph.

        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type. If not given (default), it returns the total number of nodes of all types.

        Returns
        -------
        int
            The number of nodes

        Examples
        --------
        >>> g = dgl.distributed.DistGraph('ogb-product')
        >>> print(g.num_nodes())
        2449029
        """
        if ntype is None:
            if len(self.ntypes) == 1:
                return self._gpb._num_nodes(self.ntypes[0])
            else:
                return sum(
                    [self._gpb._num_nodes(ntype) for ntype in self.ntypes]
                )
        return self._gpb._num_nodes(ntype)

    def num_edges(self, etype=None):
        """Return the total number of edges in the distributed graph.

        Parameters
        ----------
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            If not provided, return the total number of edges regardless of the types
            in the graph.

        Returns
        -------
        int
            The number of edges

        Examples
        --------
        >>> g = dgl.distributed.DistGraph('ogb-product')
        >>> print(g.num_edges())
        123718280
        """
        if etype is None:
            return sum(
                [
                    self._gpb._num_edges(c_etype)
                    for c_etype in self.canonical_etypes
                ]
            )
        return self._gpb._num_edges(etype)

    def out_degrees(self, u=ALL):
        """Return the out-degree(s) of the given nodes.

        It computes the out-degree(s).
        It does not support heterogeneous graphs yet.

        Parameters
        ----------
        u : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

            If not given, return the in-degrees of all the nodes.

        Returns
        -------
        int or Tensor
            The out-degree(s) of the node(s) in a Tensor. The i-th element is the out-degree
            of the i-th input node. If :attr:`v` is an ``int``, return an ``int`` too.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for all nodes.

        >>> g.out_degrees()
        tensor([2, 2, 0, 0])

        Query for nodes 1 and 2.

        >>> g.out_degrees(torch.tensor([1, 2]))
        tensor([2, 0])

        See Also
        --------
        in_degrees
        """
        if is_all(u):
            u = F.arange(0, self.num_nodes())
        return dist_out_degrees(self, u)

    def in_degrees(self, v=ALL):
        """Return the in-degree(s) of the given nodes.

        It computes the in-degree(s).
        It does not support heterogeneous graphs yet.

        Parameters
        ----------
        v : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

            If not given, return the in-degrees of all the nodes.

        Returns
        -------
        int or Tensor
            The in-degree(s) of the node(s) in a Tensor. The i-th element is the in-degree
            of the i-th input node. If :attr:`v` is an ``int``, return an ``int`` too.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for all nodes.

        >>> g.in_degrees()
        tensor([0, 2, 1, 1])

        Query for nodes 1 and 2.

        >>> g.in_degrees(torch.tensor([1, 2]))
        tensor([2, 1])

        See Also
        --------
        out_degrees
        """
        if is_all(v):
            v = F.arange(0, self.num_nodes())
        return dist_in_degrees(self, v)

    def node_attr_schemes(self):
        """Return the node feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g.node_attr_schemes()
        {'h': Scheme(shape=(4,), dtype=torch.float32)}

        See Also
        --------
        edge_attr_schemes
        """
        schemes = {}
        for key in self.ndata:
            schemes[key] = infer_scheme(self.ndata[key])
        return schemes

    def edge_attr_schemes(self):
        """Return the edge feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature.

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.

        Examples
        --------
        The following uses PyTorch backend.

        >>> g.edge_attr_schemes()
        {'h': Scheme(shape=(4,), dtype=torch.float32)}

        See Also
        --------
        node_attr_schemes
        """
        schemes = {}
        for key in self.edata:
            schemes[key] = infer_scheme(self.edata[key])
        return schemes

    def rank(self):
        """The rank of the current DistGraph.

        This returns a unique number to identify the DistGraph object among all of
        the client processes.

        Returns
        -------
        int
            The rank of the current DistGraph.
        """
        return role.get_global_rank()

    def find_edges(self, edges, etype=None):
        """Given an edge ID array, return the source
        and destination node ID array ``s`` and ``d``.  ``s[i]`` and ``d[i]``
        are source and destination node ID for edge ``eid[i]``.

        Parameters
        ----------
        edges : Int Tensor
            Each element is an ID. The tensor must have the same device type
              and ID data type as the graph's.

        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        tensor
            The source node ID array.
        tensor
            The destination node ID array.
        """
        if etype is None:
            assert (
                len(self.etypes) == 1
            ), "find_edges requires etype for heterogeneous graphs."

        gpb = self.get_partition_book()
        if len(gpb.etypes) > 1:
            edges = gpb.map_to_homo_eid(edges, etype)
        src, dst = dist_find_edges(self, edges)
        if len(gpb.ntypes) > 1:
            _, src = gpb.map_to_per_ntype(src)
            _, dst = gpb.map_to_per_ntype(dst)
        return src, dst

    def edge_subgraph(self, edges, relabel_nodes=True, store_ids=True):
        """Return a subgraph induced on the given edges.

        An edge-induced subgraph is equivalent to creating a new graph using the given
        edges. In addition to extracting the subgraph, DGL also copies the features
        of the extracted nodes and edges to the resulting graph. The copy is *lazy*
        and incurs data movement only when needed.

        If the graph is heterogeneous, DGL extracts a subgraph per relation and composes
        them as the resulting graph. Thus, the resulting graph has the same set of relations
        as the input one.

        Parameters
        ----------
        edges : Int Tensor or dict[(str, str, str), Int Tensor]
            The edges to form the subgraph. Each element is an edge ID. The tensor must have
            the same device type and ID data type as the graph's.

            If the graph is homogeneous, one can directly pass an Int Tensor.
            Otherwise, the argument must be a dictionary with keys being edge types
            and values being the edge IDs in the above formats.
        relabel_nodes : bool, optional
            If True, it will remove the isolated nodes and relabel the incident nodes in the
            extracted subgraph.
        store_ids : bool, optional
            If True, it will store the raw IDs of the extracted edges in the ``edata`` of the
            resulting graph under name ``dgl.EID``; if ``relabel_nodes`` is ``True``, it will
            also store the raw IDs of the incident nodes in the ``ndata`` of the resulting
            graph under name ``dgl.NID``.

        Returns
        -------
        G : DGLGraph
            The subgraph.
        """
        if isinstance(edges, dict):
            # TODO(zhengda) we need to directly generate subgraph of all relations with
            # one invocation.
            subg = {}
            for etype, edge in edges.items():
                etype = self.to_canonical_etype(etype)
                subg[etype] = self.find_edges(edge, etype)
            num_nodes = {ntype: self.num_nodes(ntype) for ntype in self.ntypes}
            subg = dgl_heterograph(subg, num_nodes_dict=num_nodes)
            for etype in edges:
                subg.edges[etype].data[EID] = edges[etype]
        else:
            assert len(self.etypes) == 1
            subg = self.find_edges(edges)
            subg = dgl_graph(subg, num_nodes=self.num_nodes())
            subg.edata[EID] = edges

        if relabel_nodes:
            subg = compact_graphs(subg)
        assert store_ids, "edge_subgraph always stores original node/edge IDs."
        return subg

    def get_partition_book(self):
        """Get the partition information.

        Returns
        -------
        GraphPartitionBook
            Object that stores all graph partition information.
        """
        return self._gpb

    def get_node_partition_policy(self, ntype):
        """Get the partition policy for a node type.

        When creating a new distributed tensor, we need to provide a partition policy
        that indicates how to distribute data of the distributed tensor in a cluster
        of machines. When we load a distributed graph in the cluster, we have pre-defined
        partition policies for each node type and each edge type. By providing
        the node type, we can reference to the pre-defined partition policy for the node type.

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        PartitionPolicy
            The partition policy for the node type.
        """
        return NodePartitionPolicy(self.get_partition_book(), ntype)

    def get_edge_partition_policy(self, etype):
        """Get the partition policy for an edge type.

        When creating a new distributed tensor, we need to provide a partition policy
        that indicates how to distribute data of the distributed tensor in a cluster
        of machines. When we load a distributed graph in the cluster, we have pre-defined
        partition policies for each node type and each edge type. By providing
        the edge type, we can reference to the pre-defined partition policy for the edge type.

        Parameters
        ----------
        etype : str or (str, str, str)
            The edge type

        Returns
        -------
        PartitionPolicy
            The partition policy for the edge type.
        """
        etype = self.to_canonical_etype(etype)
        return EdgePartitionPolicy(self.get_partition_book(), etype)

    def barrier(self):
        """Barrier for all client nodes.

        This API blocks the current process untill all the clients invoke this API.
        Please use this API with caution.
        """
        self._client.barrier()

    def sample_neighbors(
        self,
        seed_nodes,
        fanout,
        edge_dir="in",
        prob=None,
        exclude_edges=None,
        replace=False,
        etype_sorted=True,
        output_device=None,
    ):
        # pylint: disable=unused-argument
        """Sample neighbors from a distributed graph."""
        if exclude_edges is not None:
            # Convert exclude edge IDs to homogeneous edge IDs.
            gpb = self.get_partition_book()
            if isinstance(exclude_edges, Mapping):
                exclude_eids = []
                for c_etype, eids in exclude_edges.items():
                    exclude_eids.append(gpb.map_to_homo_eid(eids, c_etype))
                exclude_edges = torch.cat(exclude_eids)
        if len(self.etypes) > 1:
            frontier = graph_services.sample_etype_neighbors(
                self,
                seed_nodes,
                fanout,
                replace=replace,
                etype_sorted=etype_sorted,
                prob=prob,
                exclude_edges=exclude_edges,
                use_graphbolt=self._use_graphbolt,
            )
        else:
            frontier = graph_services.sample_neighbors(
                self,
                seed_nodes,
                fanout,
                replace=replace,
                prob=prob,
                exclude_edges=exclude_edges,
                use_graphbolt=self._use_graphbolt,
            )
        return frontier

    def _get_ndata_names(self, ntype=None):
        """Get the names of all node data."""
        names = self._client.gdata_name_list()
        ndata_names = []
        for name in names:
            name = parse_hetero_data_name(name)
            right_type = (
                (name.get_type() == ntype) if ntype is not None else True
            )
            if name.is_node() and right_type:
                ndata_names.append(name)
        return ndata_names

    def _get_edata_names(self, etype=None):
        """Get the names of all edge data."""
        if etype is not None:
            etype = self.to_canonical_etype(etype)
        names = self._client.gdata_name_list()
        edata_names = []
        for name in names:
            name = parse_hetero_data_name(name)
            right_type = (
                (name.get_type() == etype) if etype is not None else True
            )
            if name.is_edge() and right_type:
                edata_names.append(name)
        return edata_names

    def add_edge_attribute(self, name):
        """Add an edge attribute into GraphBolt partition from edge data.

        Parameters
        ----------
        name : str
            The name of the edge attribute.
        """
        # Sanity checks.
        if not self._use_graphbolt:
            raise DGLError("GraphBolt is not used.")

        # Send add request to main server on the same machine.
        kv_names = [
            (
                self.edges[etype].data[name].kvstore_key
                if name in self.edges[etype].data
                else None
            )
            for etype in self.canonical_etypes
        ]
        rpc.send_request(
            self._client._main_server_id,
            AddEdgeAttributeFromKVRequest(name, kv_names),
        )
        # Wait for the response.
        assert rpc.recv_response()._name == name
        # Send add request to local backup servers.
        for i in range(self._client.group_count - 1):
            server_id = (
                self._client.machine_id * self._client.group_count + i + 1
            )
            rpc.send_request(
                server_id, AddEdgeAttributeFromSharedMemRequest(name)
            )
        # Receive response from local backup servers.
        for _ in range(self._client.group_count - 1):
            response = rpc.recv_response()
            assert response._name == name
        # Add edge attribute from main server's shared memory.
        data = _copy_data_from_shared_mem(
            "__edge__" + name, (self.local_partition.total_num_edges,)
        )
        self.local_partition.add_edge_attribute(name, data)
        # Sync local clients.
        self._client.barrier()

        # Save the edge attribute into state. This is required by separate samplers.
        self._added_edge_attributes.append(name)


def _get_overlap(mask_arr, ids):
    """Select the IDs given a boolean mask array.

    The boolean mask array indicates all of the IDs to be selected. We want to
    find the overlap between the IDs selected by the boolean mask array and
    the ID array.

    Parameters
    ----------
    mask_arr : 1D tensor
        A boolean mask array.
    ids : 1D tensor
        A vector with IDs.

    Returns
    -------
    1D tensor
        The selected IDs.
    """
    if isinstance(mask_arr, DistTensor):
        masks = mask_arr[ids]
        return F.boolean_mask(ids, masks)
    else:
        masks = F.gather_row(F.tensor(mask_arr), ids)
        return F.boolean_mask(ids, masks)


def _split_local(partition_book, rank, elements, local_eles):
    """Split the input element list with respect to data locality."""
    num_clients = role.get_num_trainers()
    num_client_per_part = num_clients // partition_book.num_partitions()
    if rank is None:
        rank = role.get_trainer_rank()
    assert (
        rank < num_clients
    ), "The input rank ({}) is incorrect. #Trainers: {}".format(
        rank, num_clients
    )
    # all ranks of the clients in the same machine are in a contiguous range.
    client_id_in_part = rank % num_client_per_part
    local_eles = _get_overlap(elements, local_eles)

    # get a subset for the local client.
    size = len(local_eles) // num_client_per_part
    # if this isn't the last client in the partition.
    if client_id_in_part + 1 < num_client_per_part:
        return local_eles[
            (size * client_id_in_part) : (size * (client_id_in_part + 1))
        ]
    else:
        return local_eles[(size * client_id_in_part) :]


def _even_offset(n, k):
    """Split an array of length n into k segments and the difference of thier length is
    at most 1. Return the offset of each segment.
    """
    eles_per_part = n // k
    offset = np.array([0] + [eles_per_part] * k, dtype=int)
    offset[1 : n - eles_per_part * k + 1] += 1
    return np.cumsum(offset)


def _split_even_to_part(partition_book, elements):
    """Split the input element list evenly."""
    # here we divide the element list as evenly as possible. If we use range partitioning,
    # the split results also respect the data locality. Range partitioning is the default
    # strategy.
    # TODO(zhengda) we need another way to divide the list for other partitioning strategy.
    if isinstance(elements, DistTensor):
        nonzero_count = elements.count_nonzero()
    else:
        elements = F.tensor(elements)
        nonzero_count = F.count_nonzero(elements)
    # compute the offset of each split and ensure that the difference of each partition size
    # is 1.
    offsets = _even_offset(nonzero_count, partition_book.num_partitions())
    assert offsets[-1] == nonzero_count

    # Get the elements that belong to the partition.
    partid = partition_book.partid
    left, right = offsets[partid], offsets[partid + 1]

    x = y = 0
    num_elements = len(elements)
    block_size = num_elements // partition_book.num_partitions()
    part_eles = F.tensor([], dtype=elements.dtype)
    # compute the nonzero tensor of each partition instead of whole tensor to save memory
    for idx in range(0, num_elements, block_size):
        nonzero_block = F.nonzero_1d(
            elements[idx : min(idx + block_size, num_elements)]
        )
        x = y
        y += len(nonzero_block)
        if y > left and x < right:
            start = max(x, left) - x
            end = min(y, right) - x
            tmp = nonzero_block[start:end] + idx
            part_eles = F.cat((part_eles, tmp), 0)
        elif x >= right:
            break

    return part_eles


def _split_random_within_part(partition_book, rank, part_eles):
    # If there are more than one client in a partition, we need to randomly select a subset of
    # elements in the partition for a client. We have to make sure that the set of elements
    # for different clients are disjoint.

    num_clients = role.get_num_trainers()
    num_client_per_part = num_clients // partition_book.num_partitions()
    if num_client_per_part == 1:
        return part_eles
    if rank is None:
        rank = role.get_trainer_rank()
    assert (
        rank < num_clients
    ), "The input rank ({}) is incorrect. #Trainers: {}".format(
        rank, num_clients
    )
    client_id_in_part = rank % num_client_per_part
    offset = _even_offset(len(part_eles), num_client_per_part)

    # We set the random seed for each partition, so that each process (client) in a partition
    # permute the elements in a partition in the same way, so each process gets a disjoint subset
    # of elements.
    np.random.seed(partition_book.partid)
    rand_idx = np.random.permutation(len(part_eles))
    rand_idx = rand_idx[
        offset[client_id_in_part] : offset[client_id_in_part + 1]
    ]
    idx, _ = F.sort_1d(F.tensor(rand_idx))
    return F.gather_row(part_eles, idx)


def _split_by_trainer_id(
    partition_book,
    part_eles,
    trainer_id,
    num_client_per_part,
    client_id_in_part,
):
    # TODO(zhengda): MXNet cannot deal with empty tensors, which makes the implementation
    # much more difficult. Let's just use numpy for the computation for now. We just
    # perform operations on vectors. It shouldn't be too difficult.
    trainer_id = F.asnumpy(trainer_id)
    part_eles = F.asnumpy(part_eles)
    part_id = trainer_id // num_client_per_part
    trainer_id = trainer_id % num_client_per_part
    local_eles = part_eles[
        np.nonzero(part_id[part_eles] == partition_book.partid)[0]
    ]
    # these are the Ids of the local elements in the partition. The Ids are global Ids.
    remote_eles = part_eles[
        np.nonzero(part_id[part_eles] != partition_book.partid)[0]
    ]
    # these are the Ids of the remote nodes in the partition. The Ids are global Ids.
    local_eles_idx = np.concatenate(
        [
            np.nonzero(trainer_id[local_eles] == i)[0]
            for i in range(num_client_per_part)
        ],
        # trainer_id[local_eles] is the trainer ids of local nodes in the partition and we
        # pick out the indices where the node belongs to each trainer i respectively, and
        # concatenate them.
        axis=0,
    )
    # `local_eles_idx` is used to sort `local_eles` according to `trainer_id`. It is a
    # permutation of 0...(len(local_eles)-1)
    local_eles = local_eles[local_eles_idx]

    # evenly split local nodes to trainers
    local_offsets = _even_offset(len(local_eles), num_client_per_part)
    # evenly split remote nodes to trainers
    remote_offsets = _even_offset(len(remote_eles), num_client_per_part)

    client_local_eles = local_eles[
        local_offsets[client_id_in_part] : local_offsets[client_id_in_part + 1]
    ]
    client_remote_eles = remote_eles[
        remote_offsets[client_id_in_part] : remote_offsets[
            client_id_in_part + 1
        ]
    ]
    client_eles = np.concatenate(
        [client_local_eles, client_remote_eles], axis=0
    )
    return F.tensor(client_eles)


def node_split(
    nodes,
    partition_book=None,
    ntype="_N",
    rank=None,
    force_even=True,
    node_trainer_ids=None,
):
    """Split nodes and return a subset for the local rank.

    This function splits the input nodes based on the partition book and
    returns a subset of nodes for the local rank. This method is used for
    dividing workloads for distributed training.

    The input nodes are stored as a vector of masks. The length of the vector is
    the same as the number of nodes in a graph; 1 indicates that the vertex in
    the corresponding location exists.

    There are two strategies to split the nodes. By default, it splits the nodes
    in a way to maximize data locality. That is, all nodes that belong to a process
    are returned. If ``force_even`` is set to true, the nodes are split evenly so
    that each process gets almost the same number of nodes.

    When ``force_even`` is True, the data locality is still preserved if a graph is partitioned
    with Metis and the node/edge IDs are shuffled.
    In this case, majority of the nodes returned for a process are the ones that
    belong to the process. If node/edge IDs are not shuffled, data locality is not guaranteed.

    Parameters
    ----------
    nodes : 1D tensor or DistTensor
        A boolean mask vector that indicates input nodes.
    partition_book : GraphPartitionBook, optional
        The graph partition book
    ntype : str, optional
        The node type of the input nodes.
    rank : int, optional
        The rank of a process. If not given, the rank of the current process is used.
    force_even : bool, optional
        Force the nodes are split evenly.
    node_trainer_ids : 1D tensor or DistTensor, optional
        If not None, split the nodes to the trainers on the same machine according to
        trainer IDs assigned to each node. Otherwise, split randomly.

    Returns
    -------
    1D-tensor
        The vector of node IDs that belong to the rank.
    """
    if not isinstance(nodes, DistTensor):
        assert (
            partition_book is not None
        ), "Regular tensor requires a partition book."
    elif partition_book is None:
        partition_book = nodes.part_policy.partition_book

    assert len(nodes) == partition_book._num_nodes(
        ntype
    ), "The length of boolean mask vector should be the number of nodes in the graph."
    if rank is None:
        rank = role.get_trainer_rank()
    if force_even:
        num_clients = role.get_num_trainers()
        num_client_per_part = num_clients // partition_book.num_partitions()
        assert (
            num_clients % partition_book.num_partitions() == 0
        ), "The total number of clients should be multiple of the number of partitions."
        part_nid = _split_even_to_part(partition_book, nodes)
        if num_client_per_part == 1:
            return part_nid
        elif node_trainer_ids is None:
            return _split_random_within_part(partition_book, rank, part_nid)
        else:
            trainer_id = node_trainer_ids[0 : len(node_trainer_ids)]
            max_trainer_id = F.as_scalar(F.reduce_max(trainer_id)) + 1

            if max_trainer_id > num_clients:
                # We hope the partition scheme with trainer_id could be used when the number of
                # trainers is less than the `num_trainers_per_machine` previously assigned during
                # partitioning.
                assert max_trainer_id % num_clients == 0
                trainer_id //= max_trainer_id // num_clients

            client_id_in_part = rank % num_client_per_part
            return _split_by_trainer_id(
                partition_book,
                part_nid,
                trainer_id,
                num_client_per_part,
                client_id_in_part,
            )
    else:
        # Get all nodes that belong to the rank.
        local_nids = partition_book.partid2nids(
            partition_book.partid, ntype=ntype
        )
        return _split_local(partition_book, rank, nodes, local_nids)


def edge_split(
    edges,
    partition_book=None,
    etype="_E",
    rank=None,
    force_even=True,
    edge_trainer_ids=None,
):
    """Split edges and return a subset for the local rank.

    This function splits the input edges based on the partition book and
    returns a subset of edges for the local rank. This method is used for
    dividing workloads for distributed training.

    The input edges can be stored as a vector of masks. The length of the vector is
    the same as the number of edges in a graph; 1 indicates that the edge in
    the corresponding location exists.

    There are two strategies to split the edges. By default, it splits the edges
    in a way to maximize data locality. That is, all edges that belong to a process
    are returned. If ``force_even`` is set to true, the edges are split evenly so
    that each process gets almost the same number of edges.

    When ``force_even`` is True, the data locality is still preserved if a graph is partitioned
    with Metis and the node/edge IDs are shuffled.
    In this case, majority of the nodes returned for a process are the ones that
    belong to the process. If node/edge IDs are not shuffled, data locality is not guaranteed.

    Parameters
    ----------
    edges : 1D tensor or DistTensor
        A boolean mask vector that indicates input edges.
    partition_book : GraphPartitionBook, optional
        The graph partition book
    etype : str or (str, str, str), optional
        The edge type of the input edges.
    rank : int, optional
        The rank of a process. If not given, the rank of the current process is used.
    force_even : bool, optional
        Force the edges are split evenly.
    edge_trainer_ids : 1D tensor or DistTensor, optional
        If not None, split the edges to the trainers on the same machine according to
        trainer IDs assigned to each edge. Otherwise, split randomly.

    Returns
    -------
    1D-tensor
        The vector of edge IDs that belong to the rank.
    """
    if not isinstance(edges, DistTensor):
        assert (
            partition_book is not None
        ), "Regular tensor requires a partition book."
    elif partition_book is None:
        partition_book = edges.part_policy.partition_book
    assert len(edges) == partition_book._num_edges(
        etype
    ), "The length of boolean mask vector should be the number of edges in the graph."
    if rank is None:
        rank = role.get_trainer_rank()
    if force_even:
        num_clients = role.get_num_trainers()
        num_client_per_part = num_clients // partition_book.num_partitions()
        assert (
            num_clients % partition_book.num_partitions() == 0
        ), "The total number of clients should be multiple of the number of partitions."
        part_eid = _split_even_to_part(partition_book, edges)
        if num_client_per_part == 1:
            return part_eid
        elif edge_trainer_ids is None:
            return _split_random_within_part(partition_book, rank, part_eid)
        else:
            trainer_id = edge_trainer_ids[0 : len(edge_trainer_ids)]
            max_trainer_id = F.as_scalar(F.reduce_max(trainer_id)) + 1

            if max_trainer_id > num_clients:
                # We hope the partition scheme with trainer_id could be used when the number of
                # trainers is less than the `num_trainers_per_machine` previously assigned during
                # partitioning.
                assert max_trainer_id % num_clients == 0
                trainer_id //= max_trainer_id // num_clients

            client_id_in_part = rank % num_client_per_part
            return _split_by_trainer_id(
                partition_book,
                part_eid,
                trainer_id,
                num_client_per_part,
                client_id_in_part,
            )
    else:
        # Get all edges that belong to the rank.
        local_eids = partition_book.partid2eids(
            partition_book.partid, etype=etype
        )
        return _split_local(partition_book, rank, edges, local_eids)


rpc.register_service(INIT_GRAPH, InitGraphRequest, InitGraphResponse)
rpc.register_service(
    QUERY_IF_USE_GRAPHBOLT,
    QueryIfUseGraphBoltRequest,
    QueryIfUseGraphBoltResponse,
)
rpc.register_service(
    ADD_EDGE_ATTRIBUTE_FROM_KV,
    AddEdgeAttributeFromKVRequest,
    AddEdgeAttributeFromKVResponse,
)
rpc.register_service(
    ADD_EDGE_ATTRIBUTE_FROM_SHARED_MEM,
    AddEdgeAttributeFromSharedMemRequest,
    AddEdgeAttributeFromSharedMemResponse,
)
