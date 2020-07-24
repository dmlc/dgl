"""Define distributed graph."""

from collections.abc import MutableMapping
import os
import numpy as np

from ..graph import DGLGraph
from .. import backend as F
from ..base import NID, EID
from .kvstore import KVServer, KVClient
from .standalone_kvstore import KVClient as SA_KVClient
from ..graph_index import from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem
from ..frame import infer_scheme
from .partition import load_partition
from .graph_partition_book import PartitionPolicy, get_shared_mem_partition_book
from .graph_partition_book import NODE_PART_POLICY, EDGE_PART_POLICY
from .shared_mem_utils import _to_shared_mem, _get_ndata_path, _get_edata_path, DTYPE_DICT
from . import rpc
from .rpc_client import connect_to_server
from .server_state import ServerState
from .rpc_server import start_server
from .dist_tensor import DistTensor, _get_data_name
from ..transform import as_heterograph

def _get_graph_path(graph_name):
    return "/" + graph_name

def _copy_graph_to_shared_mem(g, graph_name):
    gidx = g._graph.copyto_shared_mem(_get_graph_path(graph_name))
    new_g = DGLGraph(gidx)
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    local_node_path = _get_ndata_path(graph_name, 'inner_node')
    new_g.ndata['inner_node'] = _to_shared_mem(g.ndata['inner_node'], local_node_path)
    local_edge_path = _get_edata_path(graph_name, 'inner_edge')
    new_g.edata['inner_edge'] = _to_shared_mem(g.edata['inner_edge'], local_edge_path)
    new_g.ndata[NID] = _to_shared_mem(g.ndata[NID], _get_ndata_path(graph_name, NID))
    new_g.edata[EID] = _to_shared_mem(g.edata[EID], _get_edata_path(graph_name, EID))
    return new_g

FIELD_DICT = {'inner_node': F.int64,
              'inner_edge': F.int64,
              NID: F.int64,
              EID: F.int64}

def _is_ndata_name(name):
    ''' Is this node data in the kvstore '''
    return name[:5] == NODE_PART_POLICY + ':'

def _is_edata_name(name):
    ''' Is this edge data in the kvstore '''
    return name[:5] == EDGE_PART_POLICY + ':'

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
    gidx = from_shared_mem_graph_index(_get_graph_path(graph_name))
    if gidx is None:
        return gidx

    g = DGLGraph(gidx)
    g.ndata['inner_node'] = _get_shared_mem_ndata(g, graph_name, 'inner_node')
    g.edata['inner_edge'] = _get_shared_mem_edata(g, graph_name, 'inner_edge')
    g.ndata[NID] = _get_shared_mem_ndata(g, graph_name, NID)
    g.edata[EID] = _get_shared_mem_edata(g, graph_name, EID)
    return g

class NodeDataView(MutableMapping):
    """The data view class when dist_graph.ndata[...].data is called.
    """
    __slots__ = ['_graph', '_data']

    def __init__(self, g):
        self._graph = g
        # When this is created, the server may already load node data. We need to
        # initialize the node data in advance.
        names = g._get_all_ndata_names()
        policy = PartitionPolicy(NODE_PART_POLICY, g.get_partition_book())
        self._data = {}
        for name in names:
            name1 = _get_data_name(name, policy.policy_str)
            dtype, shape, _ = g._client.get_data_meta(name1)
            # We create a wrapper on the existing tensor in the kvstore.
            self._data[name] = DistTensor(g, shape, dtype, name, part_policy=policy)

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

    def __init__(self, g):
        self._graph = g
        # When this is created, the server may already load edge data. We need to
        # initialize the edge data in advance.
        names = g._get_all_edata_names()
        policy = PartitionPolicy(EDGE_PART_POLICY, g.get_partition_book())
        self._data = {}
        for name in names:
            name1 = _get_data_name(name, policy.policy_str)
            dtype, shape, _ = g._client.get_data_meta(name1)
            # We create a wrapper on the existing tensor in the kvstore.
            self._data[name] = DistTensor(g, shape, dtype, name, part_policy=policy)

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

    This DistGraph server loads the graph data and sets up a service so that clients can read data
    of a graph partition (graph structure, node data and edge data) from remote machines.
    A server is responsible for one graph partition.

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
    num_clients : int
        Total number of client nodes.
    conf_file : string
        The path of the config file generated by the partition tool.
    disable_shared_mem : bool
        Disable shared memory.
    '''
    def __init__(self, server_id, ip_config, num_clients, conf_file, disable_shared_mem=False):
        super(DistGraphServer, self).__init__(server_id=server_id, ip_config=ip_config,
                                              num_clients=num_clients)
        self.ip_config = ip_config
        # Load graph partition data.
        self.client_g, node_feats, edge_feats, self.gpb, graph_name = load_partition(conf_file,
                                                                                     server_id)
        print('load ' + graph_name)
        if not disable_shared_mem:
            self.client_g = _copy_graph_to_shared_mem(self.client_g, graph_name)

        # Init kvstore.
        if not disable_shared_mem:
            self.gpb.shared_memory(graph_name)
        assert self.gpb.partid == server_id
        self.add_part_policy(PartitionPolicy(NODE_PART_POLICY, self.gpb))
        self.add_part_policy(PartitionPolicy(EDGE_PART_POLICY, self.gpb))

        if not self.is_backup_server():
            for name in node_feats:
                self.init_data(name=_get_data_name(name, NODE_PART_POLICY),
                               policy_str=NODE_PART_POLICY,
                               data_tensor=node_feats[name])
            for name in edge_feats:
                self.init_data(name=_get_data_name(name, EDGE_PART_POLICY),
                               policy_str=EDGE_PART_POLICY,
                               data_tensor=edge_feats[name])
        else:
            for name in node_feats:
                self.init_data(name=_get_data_name(name, NODE_PART_POLICY),
                               policy_str=NODE_PART_POLICY)
            for name in edge_feats:
                self.init_data(name=_get_data_name(name, EDGE_PART_POLICY),
                               policy_str=EDGE_PART_POLICY)

    def start(self):
        """ Start graph store server.
        """
        # start server
        server_state = ServerState(kv_store=self, local_g=self.client_g, partition_book=self.gpb)
        print('start graph service on server ' + str(self.server_id))
        start_server(server_id=self.server_id, ip_config=self.ip_config,
                     num_clients=self.num_clients, server_state=server_state)

class DistGraph:
    ''' The DistGraph client.

    This provides the graph interface to access the partitioned graph data for distributed GNN
    training. All data of partitions are loaded by the DistGraph server.

    DistGraph can run in two modes: the standalone mode and the distributed mode.

    * When a user runs the training script normally, DistGraph will be in the standalone mode.
    In this mode, the input graph has to be constructed with only one partition. This mode is
    used for testing and debugging purpose.
    * When a user runs the training script with the distributed launch script, DistGraph will
    be set into the distributed mode. This is used for actual distributed training.

    When running in the distributed mode, `DistGraph` uses shared-memory to access
    the partition data in the local machine.
    This gives the best performance for distributed training when we run `DistGraphServer`
    and `DistGraph` on the same machine. However, a user may want to run them in separate
    machines. In this case, a user may want to disable shared memory by passing
    `disable_shared_mem=False` when creating `DistGraphServer`. When shared-memory is disabled,
    a user has to pass a partition book.

    Parameters
    ----------
    ip_config : str
        Path of IP configuration file.
    graph_name : str
        The name of the graph. This name has to be the same as the one used in DistGraphServer.
    gpb : PartitionBook
        The partition book object
    conf_file : str
        The partition config file. It's used in the standalone mode.
    '''
    def __init__(self, ip_config, graph_name, gpb=None, conf_file=None):
        if os.environ.get('DGL_DIST_MODE', 'standalone') == 'standalone':
            assert conf_file is not None, \
                    'When running in the standalone model, the partition config file is required'
            self._client = SA_KVClient()
            # Load graph partition data.
            g, node_feats, edge_feats, self._gpb, _ = load_partition(conf_file, 0)
            assert self._gpb.num_partitions() == 1, \
                    'The standalone mode can only work with the graph data with one partition'
            if self._gpb is None:
                self._gpb = gpb
            self._g = as_heterograph(g)
            for name in node_feats:
                self._client.add_data(_get_data_name(name, NODE_PART_POLICY), node_feats[name])
            for name in edge_feats:
                self._client.add_data(_get_data_name(name, EDGE_PART_POLICY), edge_feats[name])
            rpc.set_num_client(1)
        else:
            connect_to_server(ip_config=ip_config)
            self._client = KVClient(ip_config)
            g = _get_graph_from_shared_mem(graph_name)
            if g is not None:
                self._g = as_heterograph(g)
            else:
                self._g = None
            self._gpb = get_shared_mem_partition_book(graph_name, self._g)
            if self._gpb is None:
                self._gpb = gpb
            self._client.barrier()
            self._client.map_shared_data(self._gpb)

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
        the server. If there is no co-location, this returns None.

        Returns
        -------
        DGLHeterograph
            The local partition
        '''
        return self._g

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        Returns
        -------
        NodeDataView
            The data view in the distributed graph storage.
        """
        return self._ndata

    @property
    def edata(self):
        """Return the data view of all the edges.

        Returns
        -------
        EdgeDataView
            The data view in the distributed graph storage.
        """
        return self._edata

    def number_of_nodes(self):
        """Return the number of nodes"""
        return self._num_nodes

    def number_of_edges(self):
        """Return the number of edges"""
        return self._num_edges

    def node_attr_schemes(self):
        """Return the node feature and embedding schemes."""
        schemes = {}
        for key in self.ndata:
            schemes[key] = infer_scheme(self.ndata[key])
        return schemes

    def edge_attr_schemes(self):
        """Return the edge feature and embedding schemes."""
        schemes = {}
        for key in self.edata:
            schemes[key] = infer_scheme(self.edata[key])
        return schemes

    def rank(self):
        ''' The rank of the distributed graph store.

        Returns
        -------
        int
            The rank of the current graph store.
        '''
        # If DistGraph doesn't have a local partition, it doesn't matter what rank
        # it returns. There is no data locality any way, as long as the returned rank
        # is unique in the system.
        if self._g is None:
            return rpc.get_rank()
        else:
            # If DistGraph has a local partition, we should be careful about the rank
            # we return. We need to return a rank that node_split or edge_split can split
            # the workload with respect to data locality.
            num_client = rpc.get_num_client()
            num_client_per_part = num_client // self._gpb.num_partitions()
            # all ranks of the clients in the same machine are in a contiguous range.
            client_id_in_part = rpc.get_rank() % num_client_per_part
            return int(self._gpb.partid * num_client_per_part + client_id_in_part)

    def get_partition_book(self):
        """Get the partition information.

        Returns
        -------
        GraphPartitionBook
            Object that stores all kinds of partition information.
        """
        return self._gpb

    def barrier(self):
        '''Barrier for all client nodes.

        This API will be blocked untill all the clients invoke this API.
        '''
        self._client.barrier()

    def _get_all_ndata_names(self):
        ''' Get the names of all node data.
        '''
        names = self._client.data_name_list()
        ndata_names = []
        for name in names:
            if _is_ndata_name(name):
                # Remove the prefix "node:"
                ndata_names.append(name[5:])
        return ndata_names

    def _get_all_edata_names(self):
        ''' Get the names of all edge data.
        '''
        names = self._client.data_name_list()
        edata_names = []
        for name in names:
            if _is_edata_name(name):
                # Remove the prefix "edge:"
                edata_names.append(name[5:])
        return edata_names

def _get_overlap(mask_arr, ids):
    """ Select the Ids given a boolean mask array.

    The boolean mask array indicates all of the Ids to be selected. We want to
    find the overlap between the Ids selected by the boolean mask array and
    the Id array.

    Parameters
    ----------
    mask_arr : 1D tensor
        A boolean mask array.
    ids : 1D tensor
        A vector with Ids.

    Returns
    -------
    1D tensor
        The selected Ids.
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
    num_clients = rpc.get_num_client()
    num_client_per_part = num_clients // partition_book.num_partitions()
    if rank is None:
        rank = rpc.get_rank()
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
    num_clients = rpc.get_num_client()
    num_client_per_part = num_clients // partition_book.num_partitions()
    if rank is None:
        rank = rpc.get_rank()
    # all ranks of the clients in the same machine are in a contiguous range.
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
    # TODO(zhegnda) we need another way to divide the list for other partitioning strategy.

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


def node_split(nodes, partition_book=None, rank=None, force_even=True):
    ''' Split nodes and return a subset for the local rank.

    This function splits the input nodes based on the partition book and
    returns a subset of nodes for the local rank. This method is used for
    dividing workloads for distributed training.

    The input nodes can be stored as a vector of masks. The length of the vector is
    the same as the number of nodes in a graph; 1 indicates that the vertex in
    the corresponding location exists.

    There are two strategies to split the nodes. By default, it splits the nodes
    in a way to maximize data locality. That is, all nodes that belong to a process
    are returned. If `force_even` is set to true, the nodes are split evenly so
    that each process gets almost the same number of nodes. The current implementation
    can still enable data locality when a graph is partitioned with range partitioning.
    In this case, majority of the nodes returned for a process are the ones that
    belong to the process. If range partitioning is not used, data locality isn't guaranteed.

    Parameters
    ----------
    nodes : 1D tensor or DistTensor
        A boolean mask vector that indicates input nodes.
    partition_book : GraphPartitionBook
        The graph partition book
    rank : int
        The rank of a process. If not given, the rank of the current process is used.
    force_even : bool
        Force the nodes are split evenly.

    Returns
    -------
    1D-tensor
        The vector of node Ids that belong to the rank.
    '''
    num_nodes = 0
    if not isinstance(nodes, DistTensor):
        assert partition_book is not None, 'Regular tensor requires a partition book.'
    elif partition_book is None:
        partition_book = nodes.part_policy.partition_book
    for part in partition_book.metadata():
        num_nodes += part['num_nodes']
    assert len(nodes) == num_nodes, \
            'The length of boolean mask vector should be the number of nodes in the graph.'
    if force_even:
        return _split_even(partition_book, rank, nodes)
    else:
        # Get all nodes that belong to the rank.
        local_nids = partition_book.partid2nids(partition_book.partid)
        return _split_local(partition_book, rank, nodes, local_nids)

def edge_split(edges, partition_book=None, rank=None, force_even=True):
    ''' Split edges and return a subset for the local rank.

    This function splits the input edges based on the partition book and
    returns a subset of edges for the local rank. This method is used for
    dividing workloads for distributed training.

    The input edges can be stored as a vector of masks. The length of the vector is
    the same as the number of edges in a graph; 1 indicates that the edge in
    the corresponding location exists.

    There are two strategies to split the edges. By default, it splits the edges
    in a way to maximize data locality. That is, all edges that belong to a process
    are returned. If `force_even` is set to true, the edges are split evenly so
    that each process gets almost the same number of edges. The current implementation
    can still enable data locality when a graph is partitioned with range partitioning.
    In this case, majority of the edges returned for a process are the ones that
    belong to the process. If range partitioning is not used, data locality isn't guaranteed.

    Parameters
    ----------
    edges : 1D tensor or DistTensor
        A boolean mask vector that indicates input edges.
    partition_book : GraphPartitionBook
        The graph partition book
    rank : int
        The rank of a process. If not given, the rank of the current process is used.
    force_even : bool
        Force the edges are split evenly.

    Returns
    -------
    1D-tensor
        The vector of edge Ids that belong to the rank.
    '''
    num_edges = 0
    if not isinstance(edges, DistTensor):
        assert partition_book is not None, 'Regular tensor requires a partition book.'
    elif partition_book is None:
        partition_book = edges.part_policy.partition_book
    for part in partition_book.metadata():
        num_edges += part['num_edges']
    assert len(edges) == num_edges, \
            'The length of boolean mask vector should be the number of edges in the graph.'

    if force_even:
        return _split_even(partition_book, rank, edges)
    else:
        # Get all edges that belong to the rank.
        local_eids = partition_book.partid2eids(partition_book.partid)
        return _split_local(partition_book, rank, edges, local_eids)
