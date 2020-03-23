from ..graph import DGLGraph
from .. import backend as F
from ..base import ALL, NID, EID
from ..data.utils import load_graphs
from .graph_store import _move_data_to_shared_mem_array
from .graph_store import _get_ndata_path, _get_edata_path, _get_graph_path
from .dis_kvstore import KVServer, KVClient

import socket
import numpy as np

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

dtype_dict = {'part_id': F.int64,
              'local_node': F.int32,
              NID: F.int64,
              EID: F.int64}

def get_shared_mem_ndata(g, graph_name, ndata_name):
    shape = (g.number_of_nodes(),)
    dtype = dtype_dict[ndata_name]
    data = empty_shared_mem(_get_ndata_path(graph_name, ndata_name), False, shape, dtype)
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

class DistGraphStoreServer(KVServer):
    def __init__(self, server_namebook, server_id, graph_name, server_data, client_data, num_client):
        super(DistGraphStoreServer, self).__init__(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        self.part_g = load_graphs(server_data)[0][0]
        client_g = load_graphs(client_data)[0][0]
        client_g.ndata['local_node'] = F.astype(client_g.ndata['part_id'] == server_id, F.int32)
        self.client_g = copy_graph_to_shared_mem(client_g, graph_name)

        num_nodes = F.as_scalar(F.max(self.part_g.ndata[NID], 0)) + 1
        self.g2l = F.zeros((num_nodes), dtype=F.int64, ctx=F.cpu())
        self.g2l[:] = -1
        self.g2l[self.part_g.ndata[NID]] = F.arange(self.part_g.number_of_nodes())
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

    def _pull_handler(self, name, lID, target):
        #lID = self.g2l[gID].asnumpy()
        #gID = gID.asnumpy()
        #print(gID[lID == -1])
        assert np.sum(lID == -1) == 0
        return target[name][lID]

class DistGraphStore:
    def __init__(self, server_namebook, graph_name):
        self._client = KVClient(server_namebook=server_namebook)
        self._client.connect()

        self.g = get_graph_from_shared_mem(graph_name)
        # TODO If we don't have HALO nodes, how do we set partition?
        num_nodes = F.as_scalar(F.max(self.g.ndata[NID], 0)) + 1
        partition = F.zeros(shape=(num_nodes,), dtype=F.int64, ctx=F.cpu())
        partition[self.g.ndata[NID]] = self.g.ndata['part_id']
        # TODO what is the node data name?
        self._client.set_partition_book(name='features', partition_book=partition)
        self._client.set_partition_book(name='labels', partition_book=partition)
        self._client.set_partition_book(name='test_mask', partition_book=partition)
        self._client.set_partition_book(name='val_mask', partition_book=partition)
        self._client.set_partition_book(name='train_mask', partition_book=partition)

        self._client.barrier()

        self.local_nids = np.nonzero(self.g.ndata['local_node'].asnumpy())[0]
        self.local_gnid = self.g.ndata[NID][self.local_nids]

    def number_of_nodes(self):
        return len(self.local_gnid)

    def get_local_nids(self):
        return self.local_nids

    def get_id(self):
        return self._client.get_id()

    def get_ndata(self, name, nids=None):
        if nids is None:
            gnid = self.local_gnid
        else:
            gnid = self.g.ndata[NID][nids]
        return self._client.pull(name=name, id_tensor=gnid)
