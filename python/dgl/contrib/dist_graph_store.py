from ..graph import DGLGraph
from .. import backend as F
from ..base import ALL, NID, EID
from ..data.utils import load_graphs
from .graph_store import _move_data_to_shared_mem_array
from .graph_store import _get_ndata_path, _get_edata_path, _get_graph_path, dtype_dict
from .dis_kvstore import KVServer, KVClient
from ..graph_index import from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem

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

    def __getitem__(self, idx):
        return self.kv.fast_pull(name=self.name, id_tensor=idx)


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
        return KVStoreTensorView(self._graph._client, key)

    def __setitem__(self, key, val):
        #TODO how to set data to the kvstore.
        pass

    def __delitem__(self, key):
        #TODO how to delete data in the kvstore.
        pass

    def __len__(self):
        return self._graph.number_of_nodes()

    def __iter__(self):
        # TODO
        pass

    def __repr__(self):
        # TODO
        pass

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
        return KVStoreTensorView(self._graph._client, key)

    def __setitem__(self, key, val):
        #TODO
        pass

    def __delitem__(self, key):
        #TODO
        pass

    def __len__(self):
        return self._graph.number_of_edges()

    def __iter__(self):
        #TODO
        pass

    def __repr__(self):
        #TODO
        pass

def load_data(data_path, graph_name, part_id):
    server_data = '{}/{}-server-{}.dgl'.format(data_path, graph_name, part_id)
    client_data = '{}/{}-client-{}.dgl'.format(data_path, graph_name, part_id)
    part_g = load_graphs(server_data)[0][0]
    client_g = load_graphs(client_data)[0][0]
    meta = pickle.load(open('{}/{}-meta.pkl'.format(data_path, graph_name), 'rb'))
    return part_g, client_g, meta

class DistGraphStoreServer(KVServer):
    def __init__(self, server_namebook, server_id, graph_name, data_path, num_client):
        super(DistGraphStoreServer, self).__init__(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

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
        if self.get_id() % self.get_group_count() == 0: # master server
            for ndata_name in self.part_g.ndata.keys():
                print(ndata_name)
                self.set_global2local(name=ndata_name, global2local=self.g2l)
                self.init_data(name=ndata_name, data_tensor=self.part_g.ndata[ndata_name])
        else:
            for ndata_name in self.part_g.ndata.keys():
                self.set_global2local(name=ndata_name)
                self.init_data(name=ndata_name)
        # TODO Do I need synchronization?

class DistGraphStore:
    def __init__(self, server_namebook, graph_name):
        self._client = KVClient(server_namebook=server_namebook)
        self._client.connect()

        self.g = get_graph_from_shared_mem(graph_name)
        self.graph_name = graph_name
        self.meta = F.asnumpy(get_shared_mem_metadata(graph_name))
        num_nodes = self.number_of_nodes()

        partition = F.zeros(shape=(num_nodes,), dtype=F.int64, ctx=F.cpu())
        partition[self.g.ndata[NID]] = self.g.ndata['part_id']
        # TODO what is the node data name?
        self._client.set_partition_book(name='features', partition_book=partition)
        self._client.set_partition_book(name='labels', partition_book=partition)
        self._client.set_partition_book(name='test_mask', partition_book=partition)
        self._client.set_partition_book(name='val_mask', partition_book=partition)
        self._client.set_partition_book(name='train_mask', partition_book=partition)

        self._client.barrier()

        self._local_nids = F.nonzero_1d(self.g.ndata['local_node'])
        self._local_gnid = self.g.ndata[NID][self._local_nids]

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data

        See Also
        --------
        dgl.DGLGraph.nodes
        """
        return NodeDataView(self, self.graph_name)

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
        return EdgeDataView(self, self.graph_name)

    def number_of_nodes(self):
        return self.meta[0]

    def number_of_local_nodes(self):
        return len(self._local_nids)

    def number_of_edges(self):
        return self.meta[1]

    @property
    def local_nids(self):
        return self._local_nids

    def get_id(self):
        return self._client.get_id()

    def is_local(self, nids):
        #TODO we need to implement it more carefully.
        return self.g.ndata['local_node'][nids]
