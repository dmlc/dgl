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
from ..transform import metis_partition_assignment, partition_graph_with_halo
from ..data.utils import save_graphs

import json
import socket
import numpy as np
from collections.abc import MutableMapping
import pickle

def _copy_graph_to_shared_mem(g, graph_name):
    gidx = g._graph.copyto_shared_mem(_get_graph_path(graph_name))
    new_g = DGLGraph(gidx)
    # We should share the node/edge data to the client explicitly instead of putting them
    # in the KVStore because some of the node/edge data may be duplicated.
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
    dtype = field_dict[name]
    dtype = dtype_dict[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, name), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

def _get_shared_mem_edata(g, graph_name, name):
    ''' Get shared-memory edge data from DistGraph server.

    This is called by the DistGraph client to access the edge data in the DistGraph server
    with shared memory.
    '''
    shape = (g.number_of_edges(),)
    dtype = field_dict[name]
    dtype = dtype_dict[dtype]
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
    dtype = dtype_dict[dtype]
    data = empty_shared_mem(_get_ndata_path(graph_name, 'meta'), False, shape, dtype)
    dlpack = data.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

class DistTensor:
    ''' Distributed tensor.

    This is a wrapper on top of KVStore to access a tensor in the KVStore.
    This wrapper provides an interface similar to the local tensor.

    Parameters
    ----------
    kv : KVClient
        The KVStore client.
    name : string
        The name of the tensor in the KVStore.
    '''
    def __init__(self, g, name):
        self.kv = g._client
        self.name = name
        dtype, shape, _ = g._client.get_data_meta(name)
        # We need to ensure that the first dim is the number of nodes in a graph.
        shape = [s for s in shape]
        shape[0] = g.number_of_nodes()
        self._shape = tuple(shape)
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
        return DistTensor(self._graph, _get_ndata_name(key))

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
            dtype, shape, _ = self._graph._client.get_data_meta(_get_ndata_name(name))
            reprs[name] = 'DistTensor(shape={}, dtype={})'.format(str(shape), str(dtype))
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
        return DistTensor(self._graph, _get_edata_name(key))

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
            dtype, shape, _ = self._graph._client.get_data_meta(_get_edata_name(name))
            reprs[name] = 'DistTensor(shape={}, dtype={})'.format(str(shape), str(dtype))
        return repr(reprs)

def load_partition(conf_file, part_id):
    ''' Load data of a partition from the data path in the DistGraph server.

    Here we load data through the normal filesystem interface. In the future, we need to support
    loading data from other storage such as S3 and HDFS.
    '''
    with open(conf_file) as f:
        part_metadata = json.load(f)
    graph_name = part_metadata['graph_name']
    part_files = part_metadata['part-{}'.format(part_id)]
    node_feats = pickle.load(open(part_files['node_feats'], 'rb'))
    edge_feats = pickle.load(open(part_files['edge_feats'], 'rb'))
    client_g = load_graphs(part_files['part_graph'])[0][0]
    node_map = pickle.load(open(part_metadata['node_map'], 'rb'))
    meta = (part_metadata['num_nodes'], part_metadata['num_edges'], node_map)
    return client_g, node_feats, edge_feats, meta

def partition_graph(g, graph_name, num_parts, num_hops, part_method, out_path):
    ''' Partition a graph for distributed training and store the partitions on files.

    The partitioning occurs in three steps: 1) run a partition algorithm (e.g., Metis) to
    assign nodes to partitions; 2) construct partition graph structure based on
    the node assignment; 3) split the node features and edge features based on
    the partition result.

    The partitioned data is stored into multiple files.

    First, the metadata of the original graph and the partitioning is stored in a JSON file
    named after `graph_name`. This JSON file contains the information of the originla graph
    as well as the file names that store each partition.

    The node assignment is stored in a separate file if we don't reshuffle node Ids to ensure
    that all nodes in a partition fall into a contiguous Id range. The node assignment is stored
    in a pickle file.

    All node features in a partition are stored in a pickle file. The node features are stored
    in a dictionary, in which the key is the node data name and the value is a tensor.

    All edge features in a partition are stored in a pickle file. The edge features are stored
    in a dictionary, in which the key is the edge data name and the value is a tensor.

    The graph structure of a partition is stored in a file with the DGLGraph format. The DGLGraph
    contains the mapping of node/edge Ids to the Ids in the original graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph to partition
    graph_name : str
        The name of the graph.
    num_parts : int
        The number of partitions
    num_hops : int
        The number of hops of HALO nodes we construct on a partition graph structure.
    part_method : str
        The partition method. It supports "random" and "metis".
    out_path : str
        The path to store the files for all partitioned data.
    '''
    if num_parts == 1:
        client_parts = {0: g}
        node_parts = F.zeros((g.number_of_nodes()), F.int64, F.cpu())
        g.ndata[NID] = F.arange(0, g.number_of_nodes())
        g.edata[EID] = F.arange(0, g.number_of_edges())
    elif part_method == 'metis':
        node_parts = metis_partition_assignment(g, num_parts)
        client_parts = partition_graph_with_halo(g, node_parts, num_hops)
    elif part_method == 'random':
        node_parts = np.random.choice(num_parts, g.number_of_nodes())
        client_parts = partition_graph_with_halo(g, node_parts, num_hops)
    else:
        raise Exception('unknown partitioning method: ' + part_method)

    tot_num_inner_edges = 0
    node_part_file = '{}/{}-node_part.pkl'.format(out_path, graph_name)
    pickle.dump(node_parts, open(node_part_file, 'wb'))
    part_metadata = {'graph_name': graph_name,
                     'num_nodes': g.number_of_nodes(),
                     'num_edges': g.number_of_edges(),
                     'part_method': part_method,
                     'num_parts': num_parts,
                     'halo_hops': num_hops,
                     'node_split': 'original',
                     'node_map': node_part_file}
    for part_id in range(num_parts):
        part = client_parts[part_id]

        # Get the node Ids that belong to this partition.
        part_ids = node_parts[part.ndata[NID]]
        local_nids = F.asnumpy(part.ndata[NID])[F.asnumpy(part_ids) == part_id]

        # Get the node/edge features of each partition.
        node_feats = {}
        edge_feats = {}
        if num_parts > 1:
            local_nodes = F.asnumpy(part.ndata[NID])[F.asnumpy(part.ndata['inner_node']) == 1]
            local_edges = F.asnumpy(part.edata[EID])[F.asnumpy(part.edata['inner_edge']) == 1]
            print('part {} has {} nodes and {} edges. {} nodes and {} edges are inside the partition'.format(
                part_id, part.number_of_nodes(), part.number_of_edges(),
                len(local_nodes), len(local_edges)))
            tot_num_inner_edges += len(local_edges)
            for name in g.ndata:
                node_feats[name] = g.ndata[name][local_nodes]
            for name in g.edata:
                edge_feats[name] = g.edata[name][local_edges]
        else:
            for name in g.ndata:
                node_feats[name] = g.ndata[name]
            for name in g.edata:
                edge_feats[name] = g.edata[name]

        node_feat_file = '{}/{}-node_feat-{}.pkl'.format(out_path, graph_name, part_id)
        edge_feat_file = '{}/{}-edge_feat-{}.pkl'.format(out_path, graph_name, part_id)
        part_graph_file = '{}/{}-part_graph-{}.dgl'.format(out_path, graph_name, part_id)
        part_metadata['part-{}'.format(part_id)] = {'node_feats': node_feat_file,
                                                    'edge_feats': edge_feat_file,
                                                    'part_graph': part_graph_file}
        pickle.dump(node_feats, open(node_feat_file, 'wb'))
        pickle.dump(edge_feats, open(edge_feat_file, 'wb'))
        save_graphs(part_graph_file, [part])

    with open('{}/{}.json'.format(out_path, graph_name), 'w') as outfile:
        json.dump(part_metadata, outfile, sort_keys=True, indent=4)

    num_cuts = g.number_of_edges() - tot_num_inner_edges
    if num_parts == 1:
        num_cuts = 0
    print('there are {} edges in the graph and {} edge cuts for {} partitions.'.format(
        g.number_of_edges(), num_cuts, num_parts))

class DistGraphServer(KVServer):
    ''' The DistGraph server.

    This DistGraph server extends KVStore server. Its main function is to load the graph data and share
    the data in its own partition with shared memory so that the DistGraph client can access the data
    in the local partition with shared memory.

    Parameters
    ----------
    server_id : int
        KVServer's ID (start from 0).
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
        The name of the graph.
    conf_file : string
        The path of the config file generated by the partition tool.
    '''
    def __init__(self, server_id, server_namebook, num_client, graph_name, conf_file):
        super(DistGraphServer, self).__init__(server_id=server_id, server_namebook=server_namebook, num_client=num_client)

        host_name = socket.gethostname()
        host_ip = socket.gethostbyname(host_name)
        print('Server {}: host name: {}, ip: {}'.format(server_id, host_name, host_ip))

        self.client_g, node_feats, edge_feats, self.meta = load_partition(conf_file, server_id)
        full_part_ids = self.meta[2]
        part_ids = full_part_ids[self.client_g.ndata[NID]]
        # TODO we need to fix this. DGL backend doesn't support boolean or byte.
        # int64 is unnecessary.
        self.client_g.ndata['local_node'] = F.astype(part_ids == server_id, F.int64)
        self.client_g = _copy_graph_to_shared_mem(self.client_g, graph_name)
        self.meta = _move_data_to_shared_mem_array(F.tensor([self.meta[0], self.meta[1]]),
                                                   _get_ndata_path(graph_name, 'meta'))

        num_nodes = self.meta[0]
        self.g2l = F.zeros((num_nodes), dtype=F.int64, ctx=F.cpu())
        self.g2l[:] = -1
        nids = F.boolean_mask(self.client_g.ndata[NID], self.client_g.ndata['local_node'])
        self.g2l[nids] = F.arange(0, len(nids))

        if self.get_id() % self.get_group_count() == 0: # master server
            for name in node_feats.keys():
                self.set_global2local(name=_get_ndata_name(name), global2local=self.g2l)
                self.init_data(name=_get_ndata_name(name), data_tensor=node_feats[name])
                self.set_partition_book(name=_get_ndata_name(name), partition_book=full_part_ids)
        else:
            for name in node_feats.keys():
                self.set_global2local(name=_get_ndata_name(name))
                self.init_data(name=_get_ndata_name(name))
                self.set_partition_book(name=_get_ndata_name(name), partition_book=full_part_ids)
        # TODO Do I need synchronization?

class DistGraph:
    ''' The DistGraph client.

    This provides the graph interface to access the partitioned graph data for distributed GNN training.
    All data of partitions are loaded by the DistGraph server. The client doesn't need to load any data.

    Parameters
    ----------
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's machine_id, IP address and port, and group_count, e.g.,

          {0:'[0, 172.31.40.143, 30050, 2],
           1:'[0, 172.31.40.143, 30051, 2],
           2:'[1, 172.31.36.140, 30050, 2],
           3:'[1, 172.31.36.140, 30051, 2],
           4:'[2, 172.31.47.147, 30050, 2],
           5:'[2, 172.31.47.147, 30051, 2],
           6:'[3, 172.31.30.180, 30050, 2],
           7:'[3, 172.31.30.180, 30051, 2]}
    graph_name : str
        The name of the graph.
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

    def _get_all_ndata_names(self):
        names = self._client.get_data_name_list()
        ndata_names = []
        for name in names:
            if _is_ndata_name(name):
                # Remove the prefix "node:"
                ndata_names.append(name[5:])
        return ndata_names

    def _get_all_edata_names(self):
        names = self._client.get_data_name_list()
        edata_names = []
        for name in names:
            if _is_edata_name(name):
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
        self._client.init_data(_get_ndata_name(ndata_name), shape, dtype)

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
        self._client.init_data(_get_edata_name(edata_name), shape, dtype)

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

    def shut_down(self):
        """Shut down all KVServer nodes.

        We usually invoke this API by just one client (e.g., client_0).
        """
        self._client.shut_down()
