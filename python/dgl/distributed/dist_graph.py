"""Define distributed graph."""

import socket
from collections.abc import MutableMapping
import numpy as np

from ..graph import DGLGraph
from .. import backend as F
from ..base import NID, EID
from ..contrib.dis_kvstore import KVServer, KVClient
from ..graph_index import from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem
from ..frame import infer_scheme
from .partition import load_partition
from .. import ndarray as nd

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _get_graph_path(graph_name):
    return "/" + graph_name

DTYPE_DICT = F.data_type_dict
DTYPE_DICT = {DTYPE_DICT[key]:key for key in DTYPE_DICT}

def _move_data_to_shared_mem_array(arr, name):
    dlpack = F.zerocopy_to_dlpack(arr)
    dgl_tensor = nd.from_dlpack(dlpack)
    new_arr = empty_shared_mem(name, True, F.shape(arr), DTYPE_DICT[F.dtype(arr)])
    dgl_tensor.copyto(new_arr)
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def _copy_graph_to_shared_mem(g, graph_name):
    gidx = g._graph.copyto_shared_mem(_get_graph_path(graph_name))
    new_g = DGLGraph(gidx)
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    local_node_path = _get_ndata_path(graph_name, 'local_node')
    new_g.ndata['local_node'] = _move_data_to_shared_mem_array(g.ndata['local_node'],
                                                               local_node_path)
    local_edge_path = _get_edata_path(graph_name, 'local_edge')
    new_g.edata['local_edge'] = _move_data_to_shared_mem_array(g.edata['local_edge'],
                                                               local_edge_path)
    new_g.ndata[NID] = _move_data_to_shared_mem_array(g.ndata[NID],
                                                      _get_ndata_path(graph_name, NID))
    new_g.edata[EID] = _move_data_to_shared_mem_array(g.edata[EID],
                                                      _get_edata_path(graph_name, EID))
    return new_g

FIELD_DICT = {'local_node': F.int64,
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
    g.ndata['local_node'] = _get_shared_mem_ndata(g, graph_name, 'local_node')
    g.ndata[NID] = _get_shared_mem_ndata(g, graph_name, NID)
    g.edata[EID] = _get_shared_mem_edata(g, graph_name, EID)
    return g

def _get_shared_mem_metadata(graph_name):
    ''' Get the metadata of the graph through shared memory.

    The metadata includes the number of nodes and the number of edges. In the future,
    we can add more information, especially for heterograph.
    '''
    shape = (2,)
    dtype = F.int64
    dtype = DTYPE_DICT[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, 'meta'), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

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

    In addition, the partition data is also shared with the DistGraph clients that run on
    the same machine.

    Parameters
    ----------
    server_id : int
        The server ID (start from 0).
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID
        (start from 0) and value is the server's machine_id, IP address and port, e.g.,

          {0:'[0, 172.31.40.143, 30050],
           1:'[0, 172.31.40.143, 30051],
           2:'[1, 172.31.36.140, 30050],
           3:'[1, 172.31.36.140, 30051],
           4:'[2, 172.31.47.147, 30050],
           5:'[2, 172.31.47.147, 30051],
           6:'[3, 172.31.30.180, 30050],
           7:'[3, 172.31.30.180, 30051]}
    num_client : int
        Total number of client nodes.
    graph_name : string
        The name of the graph. The server and the client need to specify the same graph name.
    conf_file : string
        The path of the config file generated by the partition tool.
    '''
    def __init__(self, server_id, server_namebook, num_client, graph_name, conf_file):
        super(DistGraphServer, self).__init__(server_id=server_id, server_namebook=server_namebook,
                                              num_client=num_client)

        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        self.client_g, node_feats, edge_feats, self.meta = load_partition(conf_file, server_id)
        num_nodes, num_edges, node_map, edge_map = self.meta
        self.client_g = _copy_graph_to_shared_mem(self.client_g, graph_name)
        self.meta = _move_data_to_shared_mem_array(F.tensor([num_nodes, num_edges]),
                                                   _get_ndata_path(graph_name, 'meta'))

        # Create node global2local map.
        node_g2l = F.zeros((num_nodes), dtype=F.int64, ctx=F.cpu()) - 1
        # The nodes that belong to this partition.
        local_nids = F.nonzero_1d(self.client_g.ndata['local_node'])
        nids = self.client_g.ndata[NID][local_nids]
        assert np.all(node_map[nids] == server_id), 'Load a wrong partition'
        F.scatter_row_inplace(node_g2l, nids, F.arange(0, len(nids)))

        # Create edge global2local map.
        if len(edge_feats) > 0:
            edge_g2l = F.zeros((num_edges), dtype=F.int64, ctx=F.cpu()) - 1
            local_eids = F.nonzero_1d(self.client_g.edata['local_edge'])
            eids = self.client_g.edata[EID][local_eids]
            assert np.all(edge_map[eids] == server_id), 'Load a wrong partition'
            F.scatter_row_inplace(edge_g2l, eids, F.arange(0, len(eids)))

        node_map = F.zerocopy_from_numpy(node_map)
        edge_map = F.zerocopy_from_numpy(edge_map)
        if self.get_id() % self.get_group_count() == 0: # master server
            for name in node_feats:
                self.set_global2local(name=_get_ndata_name(name), global2local=node_g2l)
                self.init_data(name=_get_ndata_name(name), data_tensor=node_feats[name])
                self.set_partition_book(name=_get_ndata_name(name), partition_book=node_map)
            for name in edge_feats:
                self.set_global2local(name=_get_edata_name(name), global2local=edge_g2l)
                self.init_data(name=_get_edata_name(name), data_tensor=edge_feats[name])
                self.set_partition_book(name=_get_edata_name(name), partition_book=edge_map)
        else:
            for name in node_feats:
                self.set_global2local(name=_get_ndata_name(name))
                self.init_data(name=_get_ndata_name(name))
                self.set_partition_book(name=_get_ndata_name(name), partition_book=node_map)
            for name in edge_feats:
                self.set_global2local(name=_get_edata_name(name))
                self.init_data(name=_get_edata_name(name))
                self.set_partition_book(name=_get_edata_name(name), partition_book=edge_map)

class DistGraph:
    ''' The DistGraph client.

    This provides the graph interface to access the partitioned graph data for distributed GNN
    training. All data of partitions are loaded by the DistGraph server. The client doesn't need
    to load any data.

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID
        (start from 0) and value is the server's machine_id, IP address and port,
        and group_count, e.g.,

          {0:'[0, 172.31.40.143, 30050, 2],
           1:'[0, 172.31.40.143, 30051, 2],
           2:'[1, 172.31.36.140, 30050, 2],
           3:'[1, 172.31.36.140, 30051, 2],
           4:'[2, 172.31.47.147, 30050, 2],
           5:'[2, 172.31.47.147, 30051, 2],
           6:'[3, 172.31.30.180, 30050, 2],
           7:'[3, 172.31.30.180, 30051, 2]}
    graph_name : str
        The name of the graph. This name has to be the same as the one used in DistGraphServer.
    '''
    def __init__(self, server_namebook, graph_name):
        self._client = KVClient(server_namebook=server_namebook)
        self._client.connect()

        self.g = _get_graph_from_shared_mem(graph_name)
        self.graph_name = graph_name
        self.meta = F.asnumpy(_get_shared_mem_metadata(graph_name))

        self._client.barrier()

        if self.g is not None:
            self._local_nids = F.nonzero_1d(self.g.ndata['local_node'])
            self._local_gnid = self.g.ndata[NID][self._local_nids]
        else:
            self._local_nids = None
            self._local_gnid = None

        self._ndata = NodeDataView(self)
        self._edata = EdgeDataView(self)


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
        names = self._ndata._get_names()
        # TODO we need to fix this. We should be able to init ndata even when there is no node data.
        assert len(names) > 0
        self._client.init_data(_get_ndata_name(ndata_name), shape, dtype, _get_ndata_name(names[0]))
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
        names = self._edata._get_names()
        # TODO we need to fix this. We should be able to init ndata even when there is no edge data.
        assert len(names) > 0
        self._client.init_data(_get_edata_name(edata_name), shape, dtype, _get_edata_name(names[0]))
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
        return self.meta[0]

    def number_of_edges(self):
        """Return the number of edges"""
        return self.meta[1]

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
        return self._client.get_id()

    def shut_down(self):
        """Shut down all KVServer nodes.

        We usually invoke this API by just one client (e.g., client_0).
        """
        # We have to remove them. Otherwise, kvstore cannot shut down correctly.
        self._ndata = None
        self._edata = None
        self._client.shut_down()

    def _get_all_ndata_names(self):
        ''' Get the names of all node data.
        '''
        names = self._client.get_data_name_list()
        ndata_names = []
        for name in names:
            if _is_ndata_name(name):
                # Remove the prefix "node:"
                ndata_names.append(name[5:])
        return ndata_names

    def _get_all_edata_names(self):
        ''' Get the names of all edge data.
        '''
        names = self._client.get_data_name_list()
        edata_names = []
        for name in names:
            if _is_edata_name(name):
                # Remove the prefix "edge:"
                edata_names.append(name[5:])
        return edata_names
