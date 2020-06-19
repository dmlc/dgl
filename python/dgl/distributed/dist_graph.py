"""Define distributed graph."""

from collections.abc import MutableMapping

from ..graph import DGLGraph
from .. import backend as F
from ..base import NID, EID
from .kvstore import KVServer, KVClient
from ..graph_index import from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem
from ..frame import infer_scheme
from .partition import load_partition
from .graph_partition_book import PartitionPolicy, get_shared_mem_partition_book
from .. import utils
from .shared_mem_utils import _to_shared_mem, _get_ndata_path, _get_edata_path, DTYPE_DICT
from .rpc_client import connect_to_server
from .server_state import ServerState
from .rpc_server import start_server

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

def _get_ndata_name(name):
    ''' This is to get the name of node data in the kvstore.

    KVStore doesn't understand node data or edge data. We'll use a prefix to distinguish them.
    '''
    return 'node:' + name

def _get_edata_name(name):
    ''' This is to get the name of edge data in the kvstore.

    KVStore doesn't understand node data or edge data. We'll use a prefix to distinguish them.
    '''
    return 'edge:' + name

def _is_ndata_name(name):
    ''' Is this node data in the kvstore '''
    return name[:5] == 'node:'

def _is_edata_name(name):
    ''' Is this edge data in the kvstore '''
    return name[:5] == 'edge:'

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

class DistTensor:
    ''' Distributed tensor.

    This is a wrapper to access a tensor stored in multiple machines.
    This wrapper provides an interface similar to the local tensor.

    Parameters
    ----------
    kv : DistGraph
        The distributed graph object.
    name : string
        The name of the tensor.
    '''
    def __init__(self, g, name):
        self.kvstore = g._client
        self.name = name
        dtype, shape, _ = g._client.get_data_meta(name)
        self._shape = shape
        self._dtype = dtype

    def __getitem__(self, idx):
        return self.kvstore.pull(name=self.name, id_tensor=idx)

    def __setitem__(self, idx, val):
        # TODO(zhengda) how do we want to support broadcast (e.g., G.ndata['h'][idx] = 1).
        self.kvstore.push(name=self.name, id_tensor=idx, data_tensor=val)

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        ''' Return the shape of the distributed tensor. '''
        return self._shape

    @property
    def dtype(self):
        ''' Return the data type of the distributed tensor. '''
        return self._dtype


class NodeDataView(MutableMapping):
    """The data view class when dist_graph.ndata[...].data is called.
    """
    __slots__ = ['_graph', '_data']

    def __init__(self, g):
        self._graph = g
        # When this is created, the server may already load node data. We need to
        # initialize the node data in advance.
        names = g._get_all_ndata_names()
        self._data = {name: DistTensor(g, _get_ndata_name(name)) for name in names}

    def _get_names(self):
        return list(self._data.keys())

    def _add(self, name):
        self._data[name] = DistTensor(self._graph, _get_ndata_name(name))

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        raise DGLError("DGL doesn't support assignment. "
                       + "Please call init_ndata to initialize new node data.")

    def __delitem__(self, key):
        #TODO(zhengda) how to delete data in the kvstore.
        raise NotImplementedError("delete node data isn't supported yet")

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
        self._data = {name: DistTensor(g, _get_edata_name(name)) for name in names}

    def _get_names(self):
        return list(self._data.keys())

    def _add(self, name):
        self._data[name] = DistTensor(self._graph, _get_edata_name(name))

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, val):
        raise DGLError("DGL doesn't support assignment. "
                       + "Please call init_edata to initialize new edge data.")

    def __delitem__(self, key):
        #TODO(zhengda) how to delete data in the kvstore.
        raise NotImplementedError("delete edge data isn't supported yet")

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
    graph_name : string
        The name of the graph. The server and the client need to specify the same graph name.
    conf_file : string
        The path of the config file generated by the partition tool.
    disable_shared_mem : bool
        Disable shared memory.
    '''
    def __init__(self, server_id, ip_config, num_clients, graph_name, conf_file,
                 disable_shared_mem=False):
        super(DistGraphServer, self).__init__(server_id=server_id, ip_config=ip_config,
                                              num_clients=num_clients)
        self.ip_config = ip_config
        # Load graph partition data.
        self.client_g, node_feats, edge_feats, self.gpb = load_partition(conf_file, server_id)
        if not disable_shared_mem:
            self.client_g = _copy_graph_to_shared_mem(self.client_g, graph_name)

        # Init kvstore.
        if not disable_shared_mem:
            self.gpb.shared_memory(graph_name)
        self.add_part_policy(PartitionPolicy('node', server_id, self.gpb))
        self.add_part_policy(PartitionPolicy('edge', server_id, self.gpb))

        if not self.is_backup_server():
            for name in node_feats:
                self.init_data(name=_get_ndata_name(name), policy_str='node',
                               data_tensor=node_feats[name])
            for name in edge_feats:
                self.init_data(name=_get_edata_name(name), policy_str='edge',
                               data_tensor=edge_feats[name])
        else:
            for name in node_feats:
                self.init_data(name=_get_ndata_name(name), policy_str='node')
            for name in edge_feats:
                self.init_data(name=_get_edata_name(name), policy_str='edge')

    def start(self):
        """ Start graph store server.
        """
        # start server
        server_state = ServerState(kv_store=self, local_g=self.client_g, partition_book=self.gpb)
        start_server(server_id=self.server_id, ip_config=self.ip_config,
                     num_clients=self.num_clients, server_state=server_state)

def _default_init_data(shape, dtype):
    return F.zeros(shape, dtype, F.cpu())

class DistGraph:
    ''' The DistGraph client.

    This provides the graph interface to access the partitioned graph data for distributed GNN
    training. All data of partitions are loaded by the DistGraph server.

    By default, `DistGraph` uses shared-memory to access the partition data in the local machine.
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
    '''
    def __init__(self, ip_config, graph_name, gpb=None):
        connect_to_server(ip_config=ip_config)
        self._client = KVClient(ip_config)
        self._g = _get_graph_from_shared_mem(graph_name)
        self._gpb = get_shared_mem_partition_book(graph_name, self._g)
        if self._gpb is None:
            self._gpb = gpb
        self._client.barrier()
        self._client.map_shared_data(self._gpb)
        self._ndata = NodeDataView(self)
        self._edata = EdgeDataView(self)
        self._default_init_ndata = _default_init_data
        self._default_init_edata = _default_init_data

        self._num_nodes = 0
        self._num_edges = 0
        for part_md in self._gpb.metadata():
            self._num_nodes += int(part_md['num_nodes'])
            self._num_edges += int(part_md['num_edges'])


    def init_ndata(self, ndata_name, shape, dtype):
        '''Initialize node data

        This initializes the node data in the distributed graph storage.

        Parameters
        ----------
        name : string
            The name of the node data.
        shape : tuple
            The shape of the node data.
        dtype : dtype
            The data type of the node data.
        '''
        assert shape[0] == self.number_of_nodes()
        self._client.init_data(_get_ndata_name(ndata_name), shape, dtype, 'node', self._gpb,
                               self._default_init_ndata)
        self._ndata._add(ndata_name)

    def init_edata(self, edata_name, shape, dtype):
        '''Initialize edge data

        This initializes the edge data in the distributed graph storage.

        Parameters
        ----------
        name : string
            The name of the edge data.
        shape : tuple
            The shape of the edge data.
        dtype : dtype
            The data type of the edge data.
        '''
        assert shape[0] == self.number_of_edges()
        self._client.init_data(_get_edata_name(edata_name), shape, dtype, 'edge', self._gpb,
                               self._default_init_edata)
        self._edata._add(edata_name)

    def init_node_emb(self, name, shape, dtype, initializer):
        ''' Initialize node embeddings.

        This initializes the node embeddings in the distributed graph storage.

        Parameters
        ----------
        name : string
            The name of the node embeddings.
        shape : tuple
            The shape of the node embeddings.
        dtype : string
            The data type of the node embeddings.
        initializer : callable
            The initializer.
        '''
        # TODO(zhengda)
        raise NotImplementedError("init_node_emb isn't supported yet")

    def get_node_embeddings(self):
        ''' Return node embeddings

        Returns
        -------
        a dict of SparseEmbedding
            All node embeddings in the graph store.
        '''
        # TODO(zhengda)
        raise NotImplementedError("get_node_embeddings isn't supported yet")

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
        return self._client.client_id

    def get_partition_book(self):
        """Get the partition information.

        Returns
        -------
        GraphPartitionBook
            Object that stores all kinds of partition information.
        """
        return self._gpb

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
        mask_arr = utils.toindex(mask_arr)
        masks = F.gather_row(mask_arr.tousertensor(), ids)
        return F.boolean_mask(ids, masks)

def node_split(nodes, partition_book, rank):
    ''' Split nodes and return a subset for the local rank.

    This function splits the input nodes based on the partition book and
    returns a subset of nodes for the local rank. This method is used for
    dividing workloads for distributed training.

    The input nodes can be stored as a vector of masks. The length of the vector is
    the same as the number of nodes in a graph; 1 indicates that the vertex in
    the corresponding location exists.

    Parameters
    ----------
    nodes : 1D tensor or DistTensor
        A boolean mask vector that indicates input nodes.
    partition_book : GraphPartitionBook
        The graph partition book
    rank : int
        The rank of the current process

    Returns
    -------
    1D-tensor
        The vector of node Ids that belong to the rank.
    '''
    num_nodes = 0
    for part in partition_book.metadata():
        num_nodes += part['num_nodes']
    assert len(nodes) == num_nodes, \
            'The length of boolean mask vector should be the number of nodes in the graph.'
    # Get all nodes that belong to the rank.
    local_nids = partition_book.partid2nids(rank)
    return _get_overlap(nodes, local_nids)

def edge_split(edges, partition_book, rank):
    ''' Split edges and return a subset for the local rank.

    This function splits the input edges based on the partition book and
    returns a subset of edges for the local rank. This method is used for
    dividing workloads for distributed training.

    The input edges can be stored as a vector of masks. The length of the vector is
    the same as the number of edges in a graph; 1 indicates that the edge in
    the corresponding location exists.

    Parameters
    ----------
    edges : 1D tensor or DistTensor
        A boolean mask vector that indicates input nodes.
    partition_book : GraphPartitionBook
        The graph partition book
    rank : int
        The rank of the current process

    Returns
    -------
    1D-tensor
        The vector of edge Ids that belong to the rank.
    '''
    num_edges = 0
    for part in partition_book.metadata():
        num_edges += part['num_edges']
    assert len(edges) == num_edges, \
            'The length of boolean mask vector should be the number of edges in the graph.'
    # Get all edges that belong to the rank.
    local_eids = partition_book.partid2eids(rank)
    return _get_overlap(edges, local_eids)
