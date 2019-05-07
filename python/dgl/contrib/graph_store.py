import os
import time
import scipy
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
import numpy as np

from collections.abc import MutableMapping

from ..base import ALL, is_all, DGLError
from .. import backend as F
from ..graph import DGLGraph
from .. import utils
from ..graph_index import GraphIndex, create_graph_index
from .._ffi.ndarray import empty_shared_mem
from .._ffi.function import _init_api
from .. import ndarray as nd

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _get_graph_path(graph_name):
    return "/" + graph_name

def _move_data_to_shared_mem_array(arr, name):
    dlpack = F.zerocopy_to_dlpack(arr)
    dgl_tensor = nd.from_dlpack(dlpack)
    new_arr = empty_shared_mem(name, True, F.shape(arr), np.dtype(F.dtype(arr)).name)
    dgl_tensor.copyto(new_arr)
    dlpack = new_arr.to_dlpack()
    return F.zerocopy_from_dlpack(dlpack)

class NodeDataView(MutableMapping):
    """The data view class when G.nodes[...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_nodes', '_graph_name']

    def __init__(self, graph, nodes, graph_name):
        self._graph = graph
        self._nodes = nodes
        self._graph_name = graph_name

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._nodes)[key]

    def __setitem__(self, key, val):
        # Move the data in val to shared memory.
        val = _move_data_to_shared_mem_array(val, _get_ndata_path(self._graph_name, key))
        self._graph.set_n_repr({key : val}, self._nodes)

    def __delitem__(self, key):
        if not is_all(self._nodes):
            raise DGLError('Delete feature data is not supported on only a subset'
                           ' of nodes. Please use `del G.ndata[key]` instead.')
        self._graph.pop_n_repr(key)

    def __len__(self):
        return len(self._graph._node_frame)

    def __iter__(self):
        return iter(self._graph._node_frame)

    def __repr__(self):
        data = self._graph.get_n_repr(self._nodes)
        return repr({key : data[key] for key in self._graph._node_frame})

class EdgeDataView(MutableMapping):
    """The data view class when G.edges[...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_edges', '_graph_name']

    def __init__(self, graph, edges, graph_name):
        self._graph = graph
        self._edges = edges
        self._graph_name = graph_name

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._edges)[key]

    def __setitem__(self, key, val):
        # Move the data in val to shared memory.
        val = _move_data_to_shared_mem_array(val, _get_edata_path(self._graph_name, key))
        self._graph.set_e_repr({key : val}, self._edges)

    def __delitem__(self, key):
        if not is_all(self._edges):
            raise DGLError('Delete feature data is not supported on only a subset'
                           ' of nodes. Please use `del G.edata[key]` instead.')
        self._graph.pop_e_repr(key)

    def __len__(self):
        return len(self._graph._edge_frame)

    def __iter__(self):
        return iter(self._graph._edge_frame)

    def __repr__(self):
        data = self._graph.get_e_repr(self._edges)
        return repr({key : data[key] for key in self._graph._edge_frame})

def _to_csr(graph_data, edge_dir, multigraph):
    try:
        indptr = graph_data.indptr
        indices = graph_data.indices
        return indptr, indices
    except:
        if isinstance(graph_data, scipy.sparse.spmatrix):
            csr = graph_data.tocsr()
            return csr.indptr, csr.indices
        else:
            idx = create_graph_index(graph_data=graph_data, multigraph=multigraph, readonly=True)
            transpose = (edge_dir != 'in')
            csr = idx.adjacency_matrix_scipy(transpose, 'csr')
            return csr.indptr, csr.indices

class Barrier(object):
    """ A barrier in the KVStore server used for one synchronization.

    All workers have to enter the barrier before any of them can proceed
    with any further computation.

    Parameters
    ----------
    num_workers: int
        The number of workers will enter the barrier.
    """
    def __init__(self, num_workers):
        self.num_enters = 0
        self.num_leaves = 0
        self.num_workers = num_workers

    def enter(self):
        """ A worker enters the barrier.
        """
        self.num_enters += 1

    def leave(self):
        """ A worker notifies the server that it's going to leave the barrier.
        """
        self.num_leaves += 1

    def all_enter(self):
        """ Indicate that all workers have entered the barrier.
        """
        return self.num_enters == self.num_workers

    def all_leave(self):
        """ Indicate that all workers have left the barrier.
        """
        return self.num_leaves == self.num_workers

class BarrierManager(object):
    """ The manager of barriers

    When a worker wants to enter a barrier, it creates the barrier if it doesn't
    exist. Otherwise, the worker will enter an existing barrier.

    The manager needs to know the number of workers in advance so that it can
    keep track of barriers and workers.

    Parameters
    ----------
    num_workers: int
        The number of workers that need to synchronize with barriers.
    """
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.barrier_ids = [0] * num_workers
        self.barriers = {}

    def enter(self, worker_id):
        """ A worker enters a barrier.

        Parameters
        ----------
        worker_id : int
            The worker that wants to enter a barrier.
        """
        bid = self.barrier_ids[worker_id]
        self.barrier_ids[worker_id] += 1
        if bid in self.barriers:
            self.barriers[bid].enter()
        else:
            self.barriers.update({bid : Barrier(self.num_workers)})
            self.barriers[bid].enter()
        return bid

    def all_enter(self, worker_id, barrier_id):
        """ Indicate whether all workers have entered a specified barrier.
        """
        return self.barriers[barrier_id].all_enter()

    def leave(self, worker_id, barrier_id):
        """ A worker leaves a barrier.

        This is useful for garbage collection of used barriers.
        """
        self.barriers[barrier_id].leave()
        if self.barriers[barrier_id].all_leave():
            del self.barriers[barrier_id]

class SharedMemoryStoreServer(object):
    """The graph store server.

    The server loads graph structure and node embeddings and edge embeddings
    and store them in shared memory. The loaded graph can be identified by
    the graph name in the input argument.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    edge_dir : string
        the edge direction for the graph structure ("in" or "out")
    graph_name : string
        Define the name of the graph, so the client can use the name to access the graph.
    multigraph : bool, optional
        Whether the graph would be a multigraph (default: False)
    num_workers : int
        The number of workers that will connect to the server.
    port : int
        The port that the server listens to.
    """
    def __init__(self, graph_data, edge_dir, graph_name, multigraph, num_workers, port):
        graph_idx = GraphIndex(multigraph=multigraph, readonly=True)
        indptr, indices = _to_csr(graph_data, edge_dir, multigraph)
        graph_idx.from_csr_matrix(indptr, indices, edge_dir, _get_graph_path(graph_name))

        self._graph = DGLGraph(graph_idx, multigraph=multigraph, readonly=True)
        self._num_workers = num_workers
        self._graph_name = graph_name
        self._edge_dir = edge_dir
        self._registered_nworkers = 0

        self._barrier = BarrierManager(num_workers)

        # RPC command: register a graph to the graph store server.
        def register(graph_name):
            if graph_name != self._graph_name:
                print("graph store has %s, but the worker wants %s"
                      % (self._graph_name, graph_name))
                return (-1, -1)
            worker_id = self._registered_nworkers
            self._registered_nworkers += 1
            return worker_id, self._num_workers

        # RPC command: get the graph information from the graph store server.
        def get_graph_info(graph_name):
            assert graph_name == self._graph_name
            return self._graph.number_of_nodes(), self._graph.number_of_edges(), \
                    self._graph.is_multigraph, edge_dir

        # RPC command: initialize node embedding in the server.
        def init_ndata(ndata_name, shape, dtype):
            if ndata_name in self._graph.ndata:
                ndata = self._graph.ndata[ndata_name]
                assert np.all(ndata.shape == tuple(shape))
                return 0

            assert self._graph.number_of_nodes() == shape[0]
            data = empty_shared_mem(_get_ndata_path(graph_name, ndata_name), True, shape, dtype)
            dlpack = data.to_dlpack()
            self._graph.ndata[ndata_name] = F.zerocopy_from_dlpack(dlpack)
            return 0

        # RPC command: initialize edge embedding in the server.
        def init_edata(edata_name, shape, dtype):
            if edata_name in self._graph.edata:
                edata = self._graph.edata[edata_name]
                assert np.all(edata.shape == tuple(shape))
                return 0

            assert self._graph.number_of_edges() == shape[0]
            data = empty_shared_mem(_get_edata_path(graph_name, edata_name), True, shape, dtype)
            dlpack = data.to_dlpack()
            self._graph.edata[edata_name] = F.zerocopy_from_dlpack(dlpack)
            return 0

        # RPC command: get the names of all node embeddings.
        def list_ndata():
            ndata = self._graph.ndata
            return [[key, F.shape(ndata[key]), np.dtype(F.dtype(ndata[key])).name] for key in ndata]

        # RPC command: get the names of all edge embeddings.
        def list_edata():
            edata = self._graph.edata
            return [[key, F.shape(edata[key]), np.dtype(F.dtype(edata[key])).name] for key in edata]

        # RPC command: notify the server of the termination of the client.
        def terminate():
            self._num_workers -= 1
            return 0

        # RPC command: a worker enters a barrier.
        def enter_barrier(worker_id):
            return self._barrier.enter(worker_id)

        # RPC command: a worker leaves a barrier.
        def leave_barrier(worker_id, barrier_id):
            self._barrier.leave(worker_id, barrier_id)
            return 0

        # RPC command: test if all workers have left a barrier.
        def all_enter(worker_id, barrier_id):
            return self._barrier.all_enter(worker_id, barrier_id)

        self.server = SimpleXMLRPCServer(("localhost", port))
        self.server.register_function(register, "register")
        self.server.register_function(get_graph_info, "get_graph_info")
        self.server.register_function(init_ndata, "init_ndata")
        self.server.register_function(init_edata, "init_edata")
        self.server.register_function(terminate, "terminate")
        self.server.register_function(list_ndata, "list_ndata")
        self.server.register_function(list_edata, "list_edata")
        self.server.register_function(enter_barrier, "enter_barrier")
        self.server.register_function(leave_barrier, "leave_barrier")
        self.server.register_function(all_enter, "all_enter")

    def __del__(self):
        self._graph = None

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data

        See Also
        --------
        dgl.DGLGraph.nodes
        """
        return NodeDataView(self._graph, ALL, self._graph_name)

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
        return EdgeDataView(self._graph, ALL, self._graph_name)

    def run(self):
        """Run the graph store server.

        The server runs to process RPC requests from clients.
        """
        while self._num_workers > 0:
            self.server.handle_request()
        self._graph = None

class SharedMemoryDGLGraph(DGLGraph):
    """Shared-memory DGLGraph.

    This is a client to access data in the shared-memory graph store that has loads
    the graph structure and node embeddings and edge embeddings to shared memory.
    It provides the DGLGraph interface.

    Parameters
    ----------
    graph_name : string
        Define the name of the graph.
    port : int
        The port that the server listens to.
    """
    def __init__(self, graph_name, port):
        self._graph_name = graph_name
        self._pid = os.getpid()
        self.proxy = xmlrpc.client.ServerProxy("http://localhost:" + str(port) + "/")
        self._worker_id, self._num_workers = self.proxy.register(graph_name)
        if self._worker_id < 0:
            raise Exception('fail to get graph ' + graph_name + ' from the graph store')
        num_nodes, num_edges, multigraph, edge_dir = self.proxy.get_graph_info(graph_name)

        graph_idx = GraphIndex(multigraph=multigraph, readonly=True)
        graph_idx.from_shared_mem_csr_matrix(_get_graph_path(graph_name), num_nodes, num_edges, edge_dir)
        super(SharedMemoryDGLGraph, self).__init__(graph_idx, multigraph=multigraph, readonly=True)

        # map all ndata and edata from the server.
        ndata_infos = self.proxy.list_ndata()
        for name, shape, dtype in ndata_infos:
            self._init_ndata(name, shape, dtype)

        edata_infos = self.proxy.list_edata()
        for name, shape, dtype in edata_infos:
            self._init_edata(name, shape, dtype)

        # Set the ndata and edata initializers.
        # so that when a new node/edge embedding is created, it'll be created on the server as well.
        def node_initializer(name, arr):
            shape = F.shape(arr)
            dtype = np.dtype(F.dtype(arr)).name
            self.proxy.init_ndata(name, shape, dtype)
            data = empty_shared_mem(_get_ndata_path(self._graph_name, name),
                                    False, shape, dtype)
            dlpack = data.to_dlpack()
            arr1 = F.zerocopy_from_dlpack(dlpack)
            arr1[:] = arr
            return arr1
        def edge_initializer(name, arr):
            shape = F.shape(arr)
            dtype = np.dtype(F.dtype(arr)).name
            self.proxy.init_edata(name, shape, dtype)
            data = empty_shared_mem(_get_edata_path(self._graph_name, name),
                                    False, shape, dtype)
            dlpack = data.to_dlpack()
            arr1 = F.zerocopy_from_dlpack(dlpack)
            arr1[:] = arr
            return arr1
        self._node_frame.set_remote_initializer(node_initializer)
        self._edge_frame.set_remote_initializer(edge_initializer)
        self._msg_frame.set_remote_initializer(edge_initializer)

    def __del__(self):
        if self.proxy is not None:
            self.proxy.terminate()

    def _init_ndata(self, ndata_name, shape, dtype):
        assert self.number_of_nodes() == shape[0]
        data = empty_shared_mem(_get_ndata_path(self._graph_name, ndata_name), False, shape, dtype)
        dlpack = data.to_dlpack()
        self.ndata[ndata_name] = F.zerocopy_from_dlpack(dlpack)

    def _init_edata(self, edata_name, shape, dtype):
        assert self.number_of_edges() == shape[0]
        data = empty_shared_mem(_get_edata_path(self._graph_name, edata_name), False, shape, dtype)
        dlpack = data.to_dlpack()
        self.edata[edata_name] = F.zerocopy_from_dlpack(dlpack)

    @property
    def num_workers(self):
        """ The number of workers using the graph store.
        """
        return self._num_workers

    @property
    def worker_id(self):
        """ The id of the current worker using the graph store.

        When a worker connects to a graph store, it is assigned with a worker id.
        This is useful for the graph store server to identify who is sending
        requests.

        The worker id is a unique number between 0 and num_workers.
        This is also useful for user's code. For example, user's code can
        use this number to decide how to assign GPUs to workers in multi-processing
        training.
        """
        return self._worker_id

    def _sync_barrier(self):
        # Here I manually implement multi-processing barrier with RPC.
        # It uses busy wait with RPC. Whenever, all_enter is called, there is
        # a context switch, so it doesn't burn CPUs so badly.
        bid = self.proxy.enter_barrier(self._worker_id)
        while not self.proxy.all_enter(self._worker_id, bid):
            continue
        self.proxy.leave_barrier(self._worker_id, bid)

    def init_ndata(self, ndata_name, shape, dtype):
        """Create node embedding.

        It first creates the node embedding in the server and maps it to the current process
        with shared memory.

        Parameters
        ----------
        ndata_name : string
            The name of node embedding
        shape : tuple
            The shape of the node embedding
        dtype : string
            The data type of the node embedding. The currently supported data types
            are "float32" and "int32".
        """
        self.proxy.init_ndata(ndata_name, shape, dtype)
        self._init_ndata(ndata_name, shape, dtype)

    def init_edata(self, edata_name, shape, dtype):
        """Create edge embedding.

        It first creates the edge embedding in the server and maps it to the current process
        with shared memory.

        Parameters
        ----------
        edata_name : string
            The name of edge embedding
        shape : tuple
            The shape of the edge embedding
        dtype : string
            The data type of the edge embedding. The currently supported data types
            are "float32" and "int32".
        """
        self.proxy.init_edata(edata_name, shape, dtype)
        self._init_edata(edata_name, shape, dtype)


    def dist_update_all(self, message_func="default",
                        reduce_func="default",
                        apply_node_func="default"):
        """ Distribute the computation in update_all among all pre-defined workers.

        dist_update_all requires that all workers invoke this method and will
        return only when all workers finish their own portion of computation.
        The number of workers are pre-defined. If one of them doesn't invoke the method,
        it won't return because some portion of computation isn't finished.

        Parameters
        ----------
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        """
        num_worker_nodes = int(self.number_of_nodes() / self.num_workers) + 1
        start_node = self.worker_id * num_worker_nodes
        end_node = min((self.worker_id + 1) * num_worker_nodes, self.number_of_nodes())
        worker_nodes = np.arange(start_node, end_node, dtype=np.int64)
        self.pull(worker_nodes, message_func, reduce_func, apply_node_func, inplace=True)
        self._sync_barrier()


    def destroy(self):
        """Destroy the graph store.

        This notifies the server that this client has terminated.
        """
        if self.proxy is not None:
            self.proxy.terminate()
        self.proxy = None


def create_graph_store_server(graph_data, graph_name, store_type, num_workers,
                              multigraph=False, edge_dir='in', port=8000):
    """Create the graph store server.

    The server loads graph structure and node embeddings and edge embeddings.

    Currently, only shared-memory graph store server is supported, so `store_type`
    can only be "shared_mem".

    After the server runs, the graph store clients can access the graph data
    with the specified graph name.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    graph_name : string
        Define the name of the graph.
    store_type : string
        The type of the graph store. The current option is "shared_mem".
    num_workers : int
        The number of workers that will connect to the server.
    multigraph : bool, optional
        Whether the graph would be a multigraph (default: False)
    edge_dir : string
        the edge direction for the graph structure. The supported option is
        "in" and "out".
    port : int
        The port that the server listens to.

    Returns
    -------
    SharedMemoryStoreServer
        The graph store server
    """
    return SharedMemoryStoreServer(graph_data, edge_dir, graph_name, multigraph,
                                   num_workers, port)

def create_graph_from_store(graph_name, store_type, port=8000):
    """Create a client from the graph store.

    The client constructs the graph structure and node embeddings and edge embeddings
    that has been loaded by the graph store server.

    Currently, only shared-memory graph store server is supported, so `store_type`
    can only be "shared_memory".

    Parameters
    ----------
    graph_name : string
        Define the name of the graph.
    store_type : string
        The type of the graph store. The current option is "shared_mem".
    port : int
        The port that the server listens to.

    Returns
    -------
    SharedMemoryDGLGraph
        The shared-memory DGLGraph
    """
    return SharedMemoryDGLGraph(graph_name, port)


_init_api("dgl.contrib.graph_store")
