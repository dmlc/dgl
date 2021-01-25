"""Define distributed graph."""

from collections.abc import MutableMapping
from collections import namedtuple

import os
import numpy as np

from ..heterograph import DGLHeteroGraph
from .. import heterograph_index
from .. import backend as F
from ..base import NID, EID, NTYPE, ETYPE
from .kvstore import KVServer, get_kvstore
from .._ffi.ndarray import empty_shared_mem
from ..frame import infer_scheme
from .partition import load_partition, load_partition_book
from .graph_partition_book import PartitionPolicy, get_shared_mem_partition_book
from .graph_partition_book import HeteroDataName, parse_hetero_data_name
from .graph_partition_book import NodePartitionPolicy, EdgePartitionPolicy
from .shared_mem_utils import _to_shared_mem, _get_ndata_path, _get_edata_path, DTYPE_DICT
from . import rpc
from . import role
from .server_state import ServerState
from .rpc_server import start_server
from .graph_services import find_edges as dist_find_edges
from .dist_tensor import DistTensor

INIT_GRAPH = 800001

class InitGraphRequest(rpc.Request):
    """ Init graph on the backup servers.

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
            server_state.graph = _get_graph_from_shared_mem(self._graph_name)
        return InitGraphResponse(self._graph_name)

class InitGraphResponse(rpc.Response):
    """ Ack the init graph request
    """
    def __init__(self, graph_name):
        self._graph_name = graph_name

    def __getstate__(self):
        return self._graph_name

    def __setstate__(self, state):
        self._graph_name = state

def _copy_graph_to_shared_mem(g, graph_name):
    new_g = g.shared_memory(graph_name, formats='csc')
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    new_g.ndata['inner_node'] = _to_shared_mem(g.ndata['inner_node'],
                                               _get_ndata_path(graph_name, 'inner_node'))
    new_g.ndata[NID] = _to_shared_mem(g.ndata[NID], _get_ndata_path(graph_name, NID))

    new_g.edata['inner_edge'] = _to_shared_mem(g.edata['inner_edge'],
                                               _get_edata_path(graph_name, 'inner_edge'))
    new_g.edata[EID] = _to_shared_mem(g.edata[EID], _get_edata_path(graph_name, EID))
    return new_g

FIELD_DICT = {'inner_node': F.int32,    # A flag indicates whether the node is inside a partition.
              'inner_edge': F.int32,    # A flag indicates whether the edge is inside a partition.
              NID: F.int64,
              EID: F.int64,
              NTYPE: F.int16,
              ETYPE: F.int16}

def _get_shared_mem_ndata(g, graph_name, name):
    ''' Get shared-memory node data from DistGraph server.

    This is called by the DistGraph client to access the node data in the DistGraph server
    with shared memory.
    '''
    shape = (g.number_of_nodes(),)
    dtype = FIELD_DICT[name]
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, name), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def _get_shared_mem_edata(g, graph_name, name):
    ''' Get shared-memory edge data from DistGraph server.

    This is called by the DistGraph client to access the edge data in the DistGraph server
    with shared memory.
    '''
    shape = (g.number_of_edges(),)
    dtype = FIELD_DICT[name]
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(_get_edata_path(graph_name, name), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def _get_graph_from_shared_mem(graph_name):
    ''' Get the graph from the DistGraph server.

    The DistGraph server puts the graph structure of the local partition in the shared memory.
    The client can access the graph structure and some metadata on nodes and edges directly
    through shared memory to reduce the overhead of data access.
    '''
    g, ntypes, etypes = heterograph_index.create_heterograph_from_shared_memory(graph_name)
    if g is None:
        return None
    g = DGLHeteroGraph(g, ntypes, etypes)

    g.ndata['inner_node'] = _get_shared_mem_ndata(g, graph_name, 'inner_node')
    g.ndata[NID] = _get_shared_mem_ndata(g, graph_name, NID)

    g.edata['inner_edge'] = _get_shared_mem_edata(g, graph_name, 'inner_edge')
    g.edata[EID] = _get_shared_mem_edata(g, graph_name, EID)
    return g

NodeSpace = namedtuple('NodeSpace', ['data'])
EdgeSpace = namedtuple('EdgeSpace', ['data'])

class HeteroNodeView(object):
    """A NodeView class to act as G.nodes for a DistGraph."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        assert isinstance(key, str)
        return NodeSpace(data=NodeDataView(self._graph, key))

class HeteroEdgeView(object):
    """A NodeView class to act as G.nodes for a DistGraph."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        assert isinstance(key, str)
        return EdgeSpace(data=EdgeDataView(self._graph, key))

class NodeDataView(MutableMapping):
    """The data view class when dist_graph.ndata[...].data is called.
    """
    __slots__ = ['_graph', '_data']

    def __init__(self, g, ntype=None):
        self._graph = g
        # When this is created, the server may already load node data. We need to
        # initialize the node data in advance.
        names = g._get_ndata_names(ntype)
        if ntype is None:
            self._data = g._ndata_store
        else:
            if ntype in g._ndata_store:
                self._data = g._ndata_store[ntype]
            else:
                self._data = {}
                g._ndata_store[ntype] = self._data
        for name in names:
            assert name.is_node()
            policy = PartitionPolicy(name.policy_str, g.get_partition_book())
            dtype, shape, _ = g._client.get_data_meta(str(name))
            # We create a wrapper on the existing tensor in the kvstore.
            self._data[name.get_name()] = DistTensor(shape, dtype, name.get_name(),
                                                     part_policy=policy)

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
            reprs[name] = 'DistTensor(shape={}, dtype={})'.format(str(shape), str(dtype))
        return repr(reprs)

class EdgeDataView(MutableMapping):
    """The data view class when G.edges[...].data is called.
    """
    __slots__ = ['_graph', '_data']

    def __init__(self, g, etype=None):
        self._graph = g
        # When this is created, the server may already load edge data. We need to
        # initialize the edge data in advance.
        names = g._get_edata_names(etype)
        if etype is None:
            self._data = g._edata_store
        else:
            if etype in g._edata_store:
                self._data = g._edata_store[etype]
            else:
                self._data = {}
                g._edata_store[etype] = self._data
        for name in names:
            assert name.is_edge()
            policy = PartitionPolicy(name.policy_str, g.get_partition_book())
            dtype, shape, _ = g._client.get_data_meta(str(name))
            # We create a wrapper on the existing tensor in the kvstore.
            self._data[name.get_name()] = DistTensor(shape, dtype, name.get_name(),
                                                     part_policy=policy)

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
            reprs[name] = 'DistTensor(shape={}, dtype={})'.format(str(shape), str(dtype))
        return repr(reprs)


class DistGraphServer(KVServer):
    ''' The DistGraph server.

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
    '''
    def __init__(self, server_id, ip_config, num_servers,
                 num_clients, part_config, disable_shared_mem=False):
        super(DistGraphServer, self).__init__(server_id=server_id,
                                              ip_config=ip_config,
                                              num_servers=num_servers,
                                              num_clients=num_clients)
        self.ip_config = ip_config
        self.num_servers = num_servers
        # Load graph partition data.
        if self.is_backup_server():
            # The backup server doesn't load the graph partition. It'll initialized afterwards.
            self.gpb, graph_name, ntypes, etypes = load_partition_book(part_config, self.part_id)
            self.client_g = None
        else:
            self.client_g, node_feats, edge_feats, self.gpb, graph_name, \
                    ntypes, etypes = load_partition(part_config, self.part_id)
            print('load ' + graph_name)
            if not disable_shared_mem:
                self.client_g = _copy_graph_to_shared_mem(self.client_g, graph_name)

        if not disable_shared_mem:
            self.gpb.shared_memory(graph_name)
        assert self.gpb.partid == self.part_id
        for ntype in ntypes:
            node_name = HeteroDataName(True, ntype, None)
            self.add_part_policy(PartitionPolicy(node_name.policy_str, self.gpb))
        for etype in etypes:
            edge_name = HeteroDataName(False, etype, None)
            self.add_part_policy(PartitionPolicy(edge_name.policy_str, self.gpb))

        if not self.is_backup_server():
            for name in node_feats:
                # The feature name has the following format: node_type + "/" + feature_name to avoid
                # feature name collision for different node types.
                ntype, feat_name = name.split('/')
                data_name = HeteroDataName(True, ntype, feat_name)
                self.init_data(name=str(data_name), policy_str=data_name.policy_str,
                               data_tensor=node_feats[name])
            for name in edge_feats:
                # The feature name has the following format: edge_type + "/" + feature_name to avoid
                # feature name collision for different edge types.
                etype, feat_name = name.split('/')
                data_name = HeteroDataName(False, etype, feat_name)
                self.init_data(name=str(data_name), policy_str=data_name.policy_str,
                               data_tensor=edge_feats[name])

    def start(self):
        """ Start graph store server.
        """
        # start server
        server_state = ServerState(kv_store=self, local_g=self.client_g, partition_book=self.gpb)
        print('start graph service on server {} for part {}'.format(self.server_id, self.part_id))
        start_server(server_id=self.server_id,
                     ip_config=self.ip_config,
                     num_servers=self.num_servers,
                     num_clients=self.num_clients, server_state=server_state)

class DistGraph:
    '''The class for accessing a distributed graph.

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
    ...                                 out_path='output/', reshuffle=True)
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
    ``DistGraph`` currently only supports graphs with only one node type and one edge type.
    For heterogeneous graphs, users need to convert them into DGL graphs with one node type and
    one edge type and store the actual node types and edge types as node data and edge data.

    Note
    ----
    DGL's distributed training by default runs server processes and trainer processes on the same
    set of machines. If users need to run them on different sets of machines, it requires
    manually setting up servers and trainers. The setup is not fully tested yet.
    '''
    def __init__(self, graph_name, gpb=None, part_config=None):
        self.graph_name = graph_name
        self._gpb_input = gpb
        if os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone':
            assert part_config is not None, \
                    'When running in the standalone model, the partition config file is required'
            self._client = get_kvstore()
            assert self._client is not None, \
                    'Distributed module is not initialized. Please call dgl.distributed.initialize.'
            # Load graph partition data.
            g, node_feats, edge_feats, self._gpb, _, _, _ = load_partition(part_config, 0)
            assert self._gpb.num_partitions() == 1, \
                    'The standalone mode can only work with the graph data with one partition'
            if self._gpb is None:
                self._gpb = gpb
            self._g = g
            for name in node_feats:
                # The feature name has the following format: node_type + "/" + feature_name.
                ntype, feat_name = name.split('/')
                self._client.add_data(str(HeteroDataName(True, ntype, feat_name)),
                                      node_feats[name],
                                      NodePartitionPolicy(self._gpb, ntype=ntype))
            for name in edge_feats:
                # The feature name has the following format: edge_type + "/" + feature_name.
                etype, feat_name = name.split('/')
                self._client.add_data(str(HeteroDataName(False, etype, feat_name)),
                                      edge_feats[name],
                                      EdgePartitionPolicy(self._gpb, etype=etype))
            self._client.map_shared_data(self._gpb)
            rpc.set_num_client(1)
        else:
            self._init()
            # Tell the backup servers to load the graph structure from shared memory.
            for server_id in range(self._client.num_servers):
                rpc.send_request(server_id, InitGraphRequest(graph_name))
            for server_id in range(self._client.num_servers):
                rpc.recv_response()
            self._client.barrier()

        self._ndata_store = {}
        self._edata_store = {}
        self._ndata = NodeDataView(self)
        self._edata = EdgeDataView(self)

        self._num_nodes = 0
        self._num_edges = 0
        for part_md in self._gpb.metadata():
            self._num_nodes += int(part_md['num_nodes'])
            self._num_edges += int(part_md['num_edges'])

        # When we store node/edge types in a list, they are stored in the order of type IDs.
        self._ntype_map = {ntype:i for i, ntype in enumerate(self.ntypes)}
        self._etype_map = {etype:i for i, etype in enumerate(self.etypes)}

    def _init(self):
        self._client = get_kvstore()
        assert self._client is not None, \
                'Distributed module is not initialized. Please call dgl.distributed.initialize.'
        self._g = _get_graph_from_shared_mem(self.graph_name)
        self._gpb = get_shared_mem_partition_book(self.graph_name, self._g)
        if self._gpb is None:
            self._gpb = self._gpb_input
        self._client.map_shared_data(self._gpb)

    def __getstate__(self):
        return self.graph_name, self._gpb

    def __setstate__(self, state):
        self.graph_name, self._gpb_input = state
        self._init()

        self._ndata_store = {}
        self._edata_store = {}
        self._ndata = NodeDataView(self)
        self._edata = EdgeDataView(self)
        self._num_nodes = 0
        self._num_edges = 0
        for part_md in self._gpb.metadata():
            self._num_nodes += int(part_md['num_nodes'])
            self._num_edges += int(part_md['num_edges'])

    @property
    def local_partition(self):
        ''' Return the local partition on the client

        DistGraph provides a global view of the distributed graph. Internally,
        it may contains a partition of the graph if it is co-located with
        the server. When servers and clients run on separate sets of machines,
        this returns None.

        Returns
        -------
        DGLGraph
            The local partition
        '''
        return self._g

    @property
    def nodes(self):
        '''Return a node view
        '''
        return HeteroNodeView(self)

    @property
    def edges(self):
        '''Return an edge view
        '''
        return HeteroEdgeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        Returns
        -------
        NodeDataView
            The data view in the distributed graph storage.
        """
        assert len(self.ntypes) == 1, "ndata only works for a graph with one node type."
        return self._ndata

    @property
    def edata(self):
        """Return the data view of all the edges.

        Returns
        -------
        EdgeDataView
            The data view in the distributed graph storage.
        """
        assert len(self.etypes) == 1, "edata only works for a graph with one edge type."
        return self._edata

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
        return self._g.idtype

    @property
    def device(self):
        """Get the device context of this graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> g = dgl.bipartite(([0, 1, 1, 2], [0, 0, 2, 1]), 'user', 'plays', 'game')
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
        return self._g.device

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
        # Currently, we only support a graph with one edge type.
        return self._gpb.etypes

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
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
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
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            return 0
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
                return sum([self._gpb._num_nodes(ntype) for ntype in self.ntypes])
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
            if len(self.etypes) == 1:
                return self._gpb._num_edges(self.etypes[0])
            else:
                return sum([self._gpb._num_edges(etype) for etype in self.etypes])
        return self._gpb._num_edges(etype)

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
        ''' The rank of the current DistGraph.

        This returns a unique number to identify the DistGraph object among all of
        the client processes.

        Returns
        -------
        int
            The rank of the current DistGraph.
        '''
        return role.get_global_rank()

    def find_edges(self, edges):
        """ Given an edge ID array, return the source
        and destination node ID array ``s`` and ``d``.  ``s[i]`` and ``d[i]``
        are source and destination node ID for edge ``eid[i]``.

        Parameters
        ----------
        edges : tensor
            The edge ID array.

        Returns
        -------
        tensor
            The source node ID array.
        tensor
            The destination node ID array.
        """
        assert len(self.etypes) == 1, 'find_edges does not support heterogeneous graph for now.'
        return dist_find_edges(self, edges)

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
        etype : str
            The edge type

        Returns
        -------
        PartitionPolicy
            The partition policy for the edge type.
        """
        return EdgePartitionPolicy(self.get_partition_book(), etype)

    def barrier(self):
        '''Barrier for all client nodes.

        This API blocks the current process untill all the clients invoke this API.
        Please use this API with caution.
        '''
        self._client.barrier()

    def _get_ndata_names(self, ntype=None):
        ''' Get the names of all node data.
        '''
        names = self._client.data_name_list()
        ndata_names = []
        for name in names:
            name = parse_hetero_data_name(name)
            right_type = (name.get_type() == ntype) if ntype is not None else True
            if name.is_node() and right_type:
                ndata_names.append(name)
        return ndata_names

    def _get_edata_names(self, etype=None):
        ''' Get the names of all edge data.
        '''
        names = self._client.data_name_list()
        edata_names = []
        for name in names:
            name = parse_hetero_data_name(name)
            right_type = (name.get_type() == etype) if etype is not None else True
            if name.is_edge() and right_type:
                edata_names.append(name)
        return edata_names

def _get_overlap(mask_arr, ids):
    """ Select the IDs given a boolean mask array.

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
    ''' Split the input element list with respect to data locality.
    '''
    num_clients = role.get_num_trainers()
    num_client_per_part = num_clients // partition_book.num_partitions()
    if rank is None:
        rank = role.get_trainer_rank()
    assert rank < num_clients, \
            'The input rank ({}) is incorrect. #Trainers: {}'.format(rank, num_clients)
    # all ranks of the clients in the same machine are in a contiguous range.
    client_id_in_part = rank  % num_client_per_part
    local_eles = _get_overlap(elements, local_eles)

    # get a subset for the local client.
    size = len(local_eles) // num_client_per_part
    # if this isn't the last client in the partition.
    if client_id_in_part + 1 < num_client_per_part:
        return local_eles[(size * client_id_in_part):(size * (client_id_in_part + 1))]
    else:
        return local_eles[(size * client_id_in_part):]

def _split_even(partition_book, rank, elements):
    ''' Split the input element list evenly.
    '''
    num_clients = role.get_num_trainers()
    num_client_per_part = num_clients // partition_book.num_partitions()
    # all ranks of the clients in the same machine are in a contiguous range.
    if rank is None:
        rank = role.get_trainer_rank()
    assert rank < num_clients, \
            'The input rank ({}) is incorrect. #Trainers: {}'.format(rank, num_clients)
    # This conversion of rank is to make the new rank aligned with partitioning.
    client_id_in_part = rank  % num_client_per_part
    rank = client_id_in_part + num_client_per_part * partition_book.partid

    if isinstance(elements, DistTensor):
        # Here we need to fetch all elements from the kvstore server.
        # I hope it's OK.
        eles = F.nonzero_1d(elements[0:len(elements)])
    else:
        eles = F.nonzero_1d(F.tensor(elements))

    # here we divide the element list as evenly as possible. If we use range partitioning,
    # the split results also respect the data locality. Range partitioning is the default
    # strategy.
    # TODO(zhengda) we need another way to divide the list for other partitioning strategy.

    # compute the offset of each split and ensure that the difference of each partition size
    # is 1.
    part_size = len(eles) // num_clients
    sizes = [part_size] * num_clients
    remain = len(eles) - part_size * num_clients
    if remain > 0:
        for i in range(num_clients):
            sizes[i] += 1
            remain -= 1
            if remain == 0:
                break
    offsets = np.cumsum(sizes)
    assert offsets[-1] == len(eles)

    if rank == 0:
        return eles[0:offsets[0]]
    else:
        return eles[offsets[rank-1]:offsets[rank]]


def node_split(nodes, partition_book=None, ntype='_N', rank=None, force_even=True):
    ''' Split nodes and return a subset for the local rank.

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

    Returns
    -------
    1D-tensor
        The vector of node IDs that belong to the rank.
    '''
    if not isinstance(nodes, DistTensor):
        assert partition_book is not None, 'Regular tensor requires a partition book.'
    elif partition_book is None:
        partition_book = nodes.part_policy.partition_book

    assert len(nodes) == partition_book._num_nodes(ntype), \
            'The length of boolean mask vector should be the number of nodes in the graph.'
    if force_even:
        return _split_even(partition_book, rank, nodes)
    else:
        # Get all nodes that belong to the rank.
        local_nids = partition_book.partid2nids(partition_book.partid)
        return _split_local(partition_book, rank, nodes, local_nids)

def edge_split(edges, partition_book=None, etype='_E', rank=None, force_even=True):
    ''' Split edges and return a subset for the local rank.

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
    etype : str, optional
        The edge type of the input edges.
    rank : int, optional
        The rank of a process. If not given, the rank of the current process is used.
    force_even : bool, optional
        Force the edges are split evenly.

    Returns
    -------
    1D-tensor
        The vector of edge IDs that belong to the rank.
    '''
    if not isinstance(edges, DistTensor):
        assert partition_book is not None, 'Regular tensor requires a partition book.'
    elif partition_book is None:
        partition_book = edges.part_policy.partition_book
    assert len(edges) == partition_book._num_edges(etype), \
            'The length of boolean mask vector should be the number of edges in the graph.'

    if force_even:
        return _split_even(partition_book, rank, edges)
    else:
        # Get all edges that belong to the rank.
        local_eids = partition_book.partid2eids(partition_book.partid)
        return _split_local(partition_book, rank, edges, local_eids)

rpc.register_service(INIT_GRAPH, InitGraphRequest, InitGraphResponse)
