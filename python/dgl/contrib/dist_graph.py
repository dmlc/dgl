from ..graph import DGLGraph
from .. import backend as F
from ..base import ALL, NID, EID
from ..data.utils import load_graphs
from .graph_store import _move_data_to_shared_mem_array
from .graph_store import _get_ndata_path, _get_edata_path, _get_graph_path, dtype_dict
from .dis_kvstore import KVServer, KVClient
from ..graph_index import from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem
from ..frame import infer_scheme

import socket
import numpy as np
from collections.abc import MutableMapping
import pickle

def copy_graph_to_shared_mem(g, graph_name):
    gidx = g._graph.copyto_shared_mem(_get_graph_path(graph_name))
    new_g = DGLGraph(gidx)
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
    new_g.ndata['part_id'] = _move_data_to_shared_mem_array(g.ndata['part_id'],
                                                            _get_ndata_path(graph_name, 'part_id'))
    new_g.ndata['local_node'] = _move_data_to_shared_mem_array(g.ndata['local_node'],
                                                               _get_ndata_path(graph_name, 'local_node'))
    new_g.ndata[NID] = _move_data_to_shared_mem_array(g.ndata[NID],
                                                      _get_ndata_path(graph_name, NID))
    new_g.edata[EID] = _move_data_to_shared_mem_array(g.edata[EID],
                                                      _get_edata_path(graph_name, EID))
    return new_g

field_dict = {'part_id': F.int64,
              'local_node': F.int32,
              NID: F.int64,
              EID: F.int64}

def get_ndata_name(name):
    return 'node:' + name

def get_edata_name(name):
    return 'edge:' + name

def is_ndata_name(name):
    return name[:5] == 'node:'

def is_edata_name(name):
    return name[:5] == 'edge:'

def get_shared_mem_ndata(g, graph_name, name):
    shape = (g.number_of_nodes(),)
    dtype = field_dict[name]
    dtype = dtype_dict[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, name), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def get_shared_mem_edata(g, graph_name, name):
    shape = (g.number_of_edges(),)
    dtype = field_dict[name]
    dtype = dtype_dict[dtype]
    data = empty_shared_mem(_get_edata_path(graph_name, name), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def get_graph_from_shared_mem(graph_name):
    gidx = from_shared_mem_graph_index(_get_graph_path(graph_name))
    g = DGLGraph(gidx)
    g.ndata['part_id'] = get_shared_mem_ndata(g, graph_name, 'part_id')
    g.ndata['local_node'] = get_shared_mem_ndata(g, graph_name, 'local_node')
    g.ndata[NID] = get_shared_mem_ndata(g, graph_name, NID)
    g.edata[EID] = get_shared_mem_edata(g, graph_name, EID)
    return g

def get_shared_mem_metadata(graph_name):
    shape = (2,)
    dtype = F.int64
    dtype = dtype_dict[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, 'meta'), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

class KVStoreTensorView:
    def __init__(self, kv, name):
        self.kv = kv
        self.name = name
        dtype, shape, _ = self.kv.get_data_meta(name)
        self._shape = shape
        self._dtype = dtype

    def __getitem__(self, idx):
        return self.kv.pull(name=self.name, id_tensor=idx)

    def __setitem__(self, idx, val):
        # TODO
        pass

    def __len__(self):
        return self._shape[0]

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype


class NodeDataView(MutableMapping):
    """The data view class when G.nodes[...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_graph_name']

    def __init__(self, g, graph_name):
        self._graph = g
        self._graph_name = graph_name

    def __getitem__(self, key):
        return KVStoreTensorView(self._graph._client, get_ndata_name(key))

    def __setitem__(self, key, val):
        #TODO how to set data to the kvstore.
        pass

    def __delitem__(self, key):
        #TODO how to delete data in the kvstore.
        pass

    def __len__(self):
        return self._graph.number_of_nodes()

    def __iter__(self):
        return iter(self._graph._get_all_ndata_names())

    def __repr__(self):
        name_list = self._graph._get_all_ndata_names()
        reprs = {}
        for name in name_list:
            dtype, shape, _ = self._graph._client.get_data_meta(get_ndata_name(name))
            reprs[name] = 'KVStoreTensorView(shape={}, dtype={})'.format(str(shape), str(dtype))
        return repr(reprs)

class EdgeDataView(MutableMapping):
    """The data view class when G.edges[...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_graph_name']

    def __init__(self, graph, graph_name):
        self._graph = graph
        self._graph_name = graph_name

    def __getitem__(self, key):
        return KVStoreTensorView(self._graph._client, get_edata_name(key))

    def __setitem__(self, key, val):
        #TODO
        pass

    def __delitem__(self, key):
        #TODO
        pass

    def __len__(self):
        return self._graph.number_of_edges()

    def __iter__(self):
        return iter(self._graph._get_all_edata_names())

    def __repr__(self):
        name_list = self._graph._get_all_edata_names()
        reprs = {}
        for name in name_list:
            dtype, shape, _ = self._graph._client.get_data_meta(get_edata_name(name))
            reprs[name] = 'KVStoreTensorView(shape={}, dtype={})'.format(str(shape), str(dtype))
        return repr(reprs)

def load_data(data_path, graph_name, part_id):
    server_data = '{}/{}-server-{}.dgl'.format(data_path, graph_name, part_id)
    client_data = '{}/{}-client-{}.dgl'.format(data_path, graph_name, part_id)
    part_g = load_graphs(server_data)[0][0]
    client_g = load_graphs(client_data)[0][0]
    meta = pickle.load(open('{}/{}-meta.pkl'.format(data_path, graph_name), 'rb'))
    return part_g, client_g, meta

class DistGraphServer(KVServer):
    def __init__(self, server_namebook, server_id, graph_name, data_path, num_client):
        super(DistGraphServer, self).__init__(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        self.part_g, self.client_g, self.meta = load_data(data_path, graph_name, server_id)
        self.client_g.ndata['local_node'] = F.astype(self.client_g.ndata['part_id'] == server_id, F.int32)
        self.client_g = copy_graph_to_shared_mem(self.client_g, graph_name)
        self.meta = _move_data_to_shared_mem_array(F.tensor(self.meta), _get_ndata_path(graph_name, 'meta'))

        num_nodes = self.meta[0]
        self.g2l = F.zeros((num_nodes), dtype=F.int64, ctx=F.cpu())
        self.g2l[:] = -1
        self.g2l[self.part_g.ndata[NID]] = F.arange(0, self.part_g.number_of_nodes())

        # TODO this works if the HALO nodes can cover all nodes used by mini-batches.
        partition = F.zeros(shape=(num_nodes,), dtype=F.int64, ctx=F.cpu())
        partition[self.client_g.ndata[NID]] = self.client_g.ndata['part_id']

        if self.get_id() % self.get_group_count() == 0: # master server
            for name in self.part_g.ndata.keys():
                self.set_global2local(name=get_ndata_name(name), global2local=self.g2l)
                self.init_data(name=get_ndata_name(name), data_tensor=self.part_g.ndata[name])
                self.set_partition_book(name=get_ndata_name(name), partition_book=partition)
        else:
            for name in self.part_g.ndata.keys():
                self.set_global2local(name=get_ndata_name(name))
                self.init_data(name=get_ndata_name(name))
                self.set_partition_book(name=get_ndata_name(name), partition_book=partition)
        # TODO Do I need synchronization?

class DistGraph:
    def __init__(self, server_namebook, graph_name):
        self._client = KVClient(server_namebook=server_namebook)
        self._client.connect()

        self.g = get_graph_from_shared_mem(graph_name)
        self.graph_name = graph_name
        self.meta = F.asnumpy(get_shared_mem_metadata(graph_name))

        self._client.barrier()

        self._local_nids = F.nonzero_1d(self.g.ndata['local_node'])
        self._local_gnid = self.g.ndata[NID][self._local_nids]

    def _get_all_ndata_names(self):
        names = self._client.get_data_name_list()
        ndata_names = []
        for name in names:
            if is_ndata_name(name):
                # Remove the prefix "node:"
                ndata_names.append(name[5:])
        return ndata_names

    def _get_all_edata_names(self):
        names = self._client.get_data_name_list()
        edata_names = []
        for name in names:
            if is_edata_name(name):
                # Remove the prefix "edge:"
                edata_names.append(name[5:])
        return edata_names


    def init_ndata(self, ndata_name, shape, dtype):
        '''Initialize node data

        This initializes the node data in the distributed KVStore.

        Parameters
        ----------
        name : string
            The name of the node data.
        shape : tuple
            The shape of the node data.
        dtype : string
            The data type of the node data.
        '''
        assert shape[0] == self.number_of_nodes()
        self._client.init_data(get_ndata_name(ndata_name), shape, dtype)

    def init_edata(self, edata_name, shape, dtype):
        '''Initialize edge data

        This initializes the edge data in the distributed KVStore.

        Parameters
        ----------
        name : string
            The name of the edge data.
        shape : tuple
            The shape of the edge data.
        dtype : string
            The data type of the edge data.
        '''
        assert shape[1] == self.number_of_edges()
        self._client.init_data(get_edata_name(edata_name), shape, dtype)

    def init_node_emb(self, name, shape, dtype, initializer):
        ''' Initialize node embeddings.

        This initializes the node embeddings in the distributed KVStore.

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
        # TODO
        pass

    def get_node_embeddings(self):
        ''' Return node embeddings

        Returns
        -------
        a dict of SparseEmbedding
            All node embeddings in the graph store.
        '''
        # TODO
        pass

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        Returns
        -------
        NodeDataView
            The data view in the distributed KVStore.
        """
        return NodeDataView(self, self.graph_name)

    @property
    def edata(self):
        """Return the data view of all the edges.

        Returns
        -------
        EdgeDataView
            The data view in the distributed KVStore.
        """
        return EdgeDataView(self, self.graph_name)

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

    @property
    def local_nids(self):
        return self._local_nids

    @property
    def local_gnids(self):
        ''' The Ids of the nodes owned by the local graph store.

        The node Ids are global node Ids.

        Returns
        -------
        tensor
            The node Ids.
        '''
        return self._local_gnid

    def rank(self):
        ''' The rank of the distributed graph store.

        Returns
        -------
        int
            The rank of the current graph store.
        '''
        return self._client.get_id()

    def is_local(self, nids):
        ''' Test if the node Ids are owned by the local graph store.

        Returns
        -------
        Boolean tensor
            Indicate whether the node Ids are owned by the local graph store.
        '''
        #TODO we need to implement it more carefully.
        return self.g.ndata['local_node'][nids]

    def barrier(self):
        ''' The barrier for all processes.

        When a process enters a barrier, it can only exit from
        the barrier when all other processes enter the barrier.
        '''
        self._client.barrier()
