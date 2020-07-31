import os
import sys
import time
import scipy
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
import numpy as np
from functools import partial

from collections.abc import MutableMapping

from ..base import ALL, is_all, DGLError, dgl_warning
from .. import backend as F
from .._deprecate.graph import DGLGraph
from .. import utils
from ..graph_index import GraphIndex, create_graph_index, from_shared_mem_graph_index
from .._ffi.ndarray import empty_shared_mem
from .._ffi.function import _init_api
from .. import ndarray as nd
from ..init import zero_initializer

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _get_graph_path(graph_name):
    return "/" + graph_name

dtype_dict = F.data_type_dict
dtype_dict = {dtype_dict[key]:key for key in dtype_dict}

def _move_data_to_shared_mem_array(arr, name):
    dlpack = F.zerocopy_to_dlpack(arr)
    dgl_tensor = nd.from_dlpack(dlpack)
    new_arr = empty_shared_mem(name, True, F.shape(arr), dtype_dict[F.dtype(arr)])
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

def shared_mem_zero_initializer(shape, dtype, name):  # pylint: disable=unused-argument
    """Zero feature initializer in shared memory
    """
    data = empty_shared_mem(name, True, shape, dtype)
    dlpack = data.to_dlpack()
    arr = F.zerocopy_from_dlpack(dlpack)
    arr[:] = 0
    return arr

class InitializerManager(object):
    """Manage initializer.

    We need to convert built-in frame initializer to strings
    and send them to the graph store server through RPC.
    Through the conversion, we need to convert local built-in initializer
    to shared-memory initializer.
    """

    # Map the built-in initializer functions to strings.
    _fun2str = {
        zero_initializer: 'zero',
    }

    # Map the strings to built-in initializer functions.
    _str2fun = {
        'zero': shared_mem_zero_initializer,
    }

    def serialize(self, init):
        """Convert the initializer function to string.

        Parameters
        ----------
        init : callable
            the initializer function.

        Returns
        ------
        string
            The name of the built-in initializer function.
        """
        if init in self._fun2str:
            return self._fun2str[init]
        else:
            raise Exception("Shared-memory graph store doesn't support user's initializer")

    def deserialize(self, init):
        """Convert the string to the initializer function.

        Parameters
        ----------
        init : string
            the name of the initializer function

        Returns
        -------
        callable
            The shared-memory initializer function.
        """
        if init in self._str2fun:
            return self._str2fun[init]
        else:
            raise Exception("Shared-memory graph store doesn't support initializer "
                            + str(init))


class SharedMemoryStoreServer(object):
    """The graph store server.

    The server loads graph structure and node embeddings and edge embeddings
    and store them in shared memory. The loaded graph can be identified by
    the graph name in the input argument.

    DGL graph accepts graph data of multiple formats:

    * NetworkX graph,
    * scipy matrix,
    * DGLGraph.

    If the input graph data is DGLGraph, the constructed DGLGraph only contains
    its graph index.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph.
    graph_name : string
        Define the name of the graph, so the client can use the name to access the graph.
    multigraph : bool, optional
        Deprecated (Will be deleted in the future).
        Whether the graph would be a multigraph (default: True)
    num_workers : int
        The number of workers that will connect to the server.
    port : int
        The port that the server listens to.
    """
    def __init__(self, graph_data, graph_name, multigraph, num_workers, port):
        self.server = None
        if multigraph is not None:
            dgl_warning("multigraph will be deprecated." \
                        "DGL will treat all graphs as multigraph in the future.")

        if isinstance(graph_data, GraphIndex):
            graph_data = graph_data.copyto_shared_mem(_get_graph_path(graph_name))
        elif isinstance(graph_data, DGLGraph):
            graph_data = graph_data._graph.copyto_shared_mem(_get_graph_path(graph_name))
        else:
            graph_data = create_graph_index(graph_data, readonly=True)
            graph_data = graph_data.copyto_shared_mem(_get_graph_path(graph_name))
        self._graph = DGLGraph(graph_data, readonly=True)

        self._num_workers = num_workers
        self._graph_name = graph_name
        self._registered_nworkers = 0

        self._barrier = BarrierManager(num_workers)
        self._init_manager = InitializerManager()

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
            # if the integers are larger than 2^31, xmlrpc can't handle them.
            # we convert them to strings to send them to clients.
            return str(self._graph.number_of_nodes()), str(self._graph.number_of_edges())

        # RPC command: initialize node embedding in the server.
        def init_ndata(init, ndata_name, shape, dtype):
            if ndata_name in self._graph.ndata:
                ndata = self._graph.ndata[ndata_name]
                assert np.all(tuple(F.shape(ndata)) == tuple(shape))
                return 0

            assert self._graph.number_of_nodes() == shape[0]
            init = self._init_manager.deserialize(init)
            data = init(shape, dtype, _get_ndata_path(graph_name, ndata_name))
            self._graph.ndata[ndata_name] = data
            F.sync()
            return 0

        # RPC command: initialize edge embedding in the server.
        def init_edata(init, edata_name, shape, dtype):
            if edata_name in self._graph.edata:
                edata = self._graph.edata[edata_name]
                assert np.all(tuple(F.shape(edata)) == tuple(shape))
                return 0

            assert self._graph.number_of_edges() == shape[0]
            init = self._init_manager.deserialize(init)
            data = init(shape, dtype, _get_edata_path(graph_name, edata_name))
            F.sync()
            self._graph.edata[edata_name] = data
            return 0

        # RPC command: get the names of all node embeddings.
        def list_ndata():
            ndata = self._graph.ndata
            return [[key, tuple(F.shape(ndata[key])), dtype_dict[F.dtype(ndata[key])]] for key in ndata]

        # RPC command: get the names of all edge embeddings.
        def list_edata():
            edata = self._graph.edata
            return [[key, tuple(F.shape(edata[key])), dtype_dict[F.dtype(edata[key])]] for key in edata]

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

        self.server = SimpleXMLRPCServer(("127.0.0.1", port), logRequests=False)
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
        if self.server is not None:
            self.server.server_close()
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


class BaseGraphStore(DGLGraph):
    """The base class of the graph store.

    Shared-memory graph store and distributed graph store will be inherited from
    this base class. The graph stores only support large read-only graphs. Thus, many of
    DGLGraph APIs aren't supported.

    Specially, the graph store doesn't support the following methods:
        - ndata
        - edata
        - incidence_matrix
        - line_graph
        - reverse
    """
    def __init__(self,
                 graph_data=None,
                 multigraph=None):
        super(BaseGraphStore, self).__init__(graph_data, multigraph=multigraph, readonly=True)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data
        """
        raise Exception("Graph store doesn't support access data of all nodes.")

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
        raise Exception("Graph store doesn't support access data of all edges.")

    def incidence_matrix(self, typestr, ctx=F.cpu()):
        """Return the incidence matrix representation of this graph.

        Parameters
        ----------
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        raise Exception("Graph store doesn't support creating an incidence matrix.")

    def line_graph(self, backtracking=True, shared=False):
        """Return the line graph of this graph.

        See :func:`~dgl.transform.line_graph`.
        """
        raise Exception("Graph store doesn't support creating an line matrix.")

    def reverse(self, share_ndata=False, share_edata=False):
        """Return the reverse of this graph.

        See :func:`~dgl.transform.reverse`.
        """
        raise Exception("Graph store doesn't support reversing a matrix.")


class SharedMemoryDGLGraph(BaseGraphStore):
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
        self.proxy = xmlrpc.client.ServerProxy("http://127.0.0.1:" + str(port) + "/")
        self._worker_id, self._num_workers = self.proxy.register(graph_name)
        if self._worker_id < 0:
            raise Exception('fail to get graph ' + graph_name + ' from the graph store')
        num_nodes, num_edges = self.proxy.get_graph_info(graph_name)
        num_nodes, num_edges = int(num_nodes), int(num_edges)

        graph_idx = from_shared_mem_graph_index(_get_graph_path(graph_name))
        super(SharedMemoryDGLGraph, self).__init__(graph_idx)
        self._init_manager = InitializerManager()

        # map all ndata and edata from the server.
        ndata_infos = self.proxy.list_ndata()
        for name, shape, dtype in ndata_infos:
            self._init_ndata(name, shape, dtype)

        edata_infos = self.proxy.list_edata()
        for name, shape, dtype in edata_infos:
            self._init_edata(name, shape, dtype)

        # Set the ndata and edata initializers.
        # so that when a new node/edge embedding is created, it'll be created on the server as well.

        # These two functions create initialized tensors on the server.
        def node_initializer(init, name, shape, dtype, ctx):
            init = self._init_manager.serialize(init)
            dtype = dtype_dict[dtype]
            self.proxy.init_ndata(init, name, tuple(shape), dtype)
            data = empty_shared_mem(_get_ndata_path(self._graph_name, name),
                                    False, shape, dtype)
            dlpack = data.to_dlpack()
            return F.zerocopy_from_dlpack(dlpack)
        def edge_initializer(init, name, shape, dtype, ctx):
            init = self._init_manager.serialize(init)
            dtype = dtype_dict[dtype]
            self.proxy.init_edata(init, name, tuple(shape), dtype)
            data = empty_shared_mem(_get_edata_path(self._graph_name, name),
                                    False, shape, dtype)
            dlpack = data.to_dlpack()
            return F.zerocopy_from_dlpack(dlpack)

        self._node_frame.set_remote_init_builder(lambda init, name: partial(node_initializer, init, name))
        self._edge_frame.set_remote_init_builder(lambda init, name: partial(edge_initializer, init, name))
        self._msg_frame.set_remote_init_builder(lambda init, name: partial(edge_initializer, init, name))

    def __del__(self):
        if self.proxy is not None:
            self.proxy.terminate()

    def _init_ndata(self, ndata_name, shape, dtype):
        assert self.number_of_nodes() == shape[0]
        data = empty_shared_mem(_get_ndata_path(self._graph_name, ndata_name), False, shape, dtype)
        dlpack = data.to_dlpack()
        self.set_n_repr({ndata_name: F.zerocopy_from_dlpack(dlpack)})

    def _init_edata(self, edata_name, shape, dtype):
        assert self.number_of_edges() == shape[0]
        data = empty_shared_mem(_get_edata_path(self._graph_name, edata_name), False, shape, dtype)
        dlpack = data.to_dlpack()
        self.set_e_repr({edata_name: F.zerocopy_from_dlpack(dlpack)})

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

    def _sync_barrier(self, timeout=None):
        """This is a sync barrier among all workers.

        Parameters
        ----------
        timeout: int
            time out in seconds.
        """
        # Before entering the barrier, we need to make sure all computation in the local
        # process has completed.
        F.sync()

        # Here I manually implement multi-processing barrier with RPC.
        # It uses busy wait with RPC. Whenever, all_enter is called, there is
        # a context switch, so it doesn't burn CPUs so badly.

        # if timeout isn't specified, we wait forever.
        if timeout is None:
            timeout = sys.maxsize

        bid = self.proxy.enter_barrier(self._worker_id)
        start = time.time()
        while not self.proxy.all_enter(self._worker_id, bid) and time.time() - start < timeout:
            continue
        self.proxy.leave_barrier(self._worker_id, bid)
        if time.time() - start >= timeout and not self.proxy.all_enter(self._worker_id, bid):
            raise TimeoutError("leave the sync barrier because of timeout.")

    def init_ndata(self, ndata_name, shape, dtype, ctx=F.cpu()):
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
        ctx : DGLContext
            The column context.
        """
        if ctx != F.cpu():
            raise Exception("graph store only supports CPU context for node data")
        init = self._node_frame.get_initializer(ndata_name)
        if init is None:
            self._node_frame._frame._set_zero_default_initializer()
        init = self._node_frame.get_initializer(ndata_name)
        init = self._init_manager.serialize(init)
        self.proxy.init_ndata(init, ndata_name, tuple(shape), dtype)
        self._init_ndata(ndata_name, shape, dtype)

    def init_edata(self, edata_name, shape, dtype, ctx=F.cpu()):
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
        ctx : DGLContext
            The column context.
        """
        if ctx != F.cpu():
            raise Exception("graph store only supports CPU context for edge data")
        init = self._edge_frame.get_initializer(edata_name)
        if init is None:
            self._edge_frame._frame._set_zero_default_initializer()
        init = self._edge_frame.get_initializer(edata_name)
        init = self._init_manager.serialize(init)
        self.proxy.init_edata(init, edata_name, tuple(shape), dtype)
        self._init_edata(edata_name, shape, dtype)

    def get_n_repr(self, u=ALL):
        """Get node(s) representation.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        if len(self.node_attr_schemes()) == 0:
            return dict()
        if is_all(u):
            dgl_warning("It may not be safe to access node data of all nodes."
                        "It's recommended to node data of a subset of nodes directly.")
            return dict(self._node_frame)
        else:
            u = utils.toindex(u)
            return self._node_frame.select_rows(u)

    def get_e_repr(self, edges=ALL):
        """Get edge(s) representation.

        Parameters
        ----------
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        if is_all(edges):
            dgl_warning("It may not be safe to access edge data of all edges."
                        "It's recommended to edge data of a subset of edges directly.")
        return super(SharedMemoryDGLGraph, self).get_e_repr(edges)


    def set_n_repr(self, data, u=ALL, inplace=True):
        """Set node(s) representation.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        data : dict of tensor
            Node representation.
        u : node, container or tensor
            The node(s).
        inplace : bool
            The value is always True.
        """
        super(BaseGraphStore, self).set_n_repr(data, u, inplace=True)

    def set_e_repr(self, data, edges=ALL, inplace=True):
        """Set edge(s) representation.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        data : tensor or dict of tensor
            Edge representation.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        inplace : bool
            The value is always True.
        """
        super(BaseGraphStore, self).set_e_repr(data, edges, inplace=True)

    def apply_nodes(self, func="default", v=ALL, inplace=True):
        """Apply the function on the nodes to update their features.

        If None is provided for ``func``, nothing will happen.

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        func : callable or None, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : int, iterable of int, tensor, optional
            The node (ids) on which to apply ``func``. The default
            value is all the nodes.
        inplace : bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).apply_nodes(func, v, inplace=True)

    def apply_edges(self, func="default", edges=ALL, inplace=True):
        """Apply the function on the edges to update their features.

        If None is provided for ``func``, nothing will happen.

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        func : callable, optional
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : valid edges type, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).apply_edges(func, edges, inplace=True)

    def group_apply_edges(self, group_by, func, edges=ALL, inplace=True):
        """Group the edges by nodes and apply the function on the grouped edges to
         update their features.

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        group_by : str
            Specify how to group edges. Expected to be either 'src' or 'dst'
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.
        edges : valid edges type, optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).group_apply_edges(group_by, func, edges, inplace=True)

    def recv(self,
             v=ALL,
             reduce_func="default",
             apply_node_func="default",
             inplace=True):
        """Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        Optionally, apply a function to update the node features after receive.

        In the graph store, all updates are written inplace.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        The node features will be updated by the result of the ``reduce_func``.

        Messages are consumed once received.

        The provided UDF maybe called multiple times so it is recommended to provide
        function with no side effect.

        Parameters
        ----------
        v : node, container or tensor, optional
            The node to be updated. Default is receiving all the nodes.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).recv(v, reduce_func, apply_node_func, inplace=True)

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
                      inplace=True):
        """Send messages along edges and let destinations receive them.

        Optionally, apply a function to update the node features after receive.

        In the graph store, all updates are written inplace.

        This is a convenient combination for performing
        ``send(self, self.edges, message_func)`` and
        ``recv(self, dst, reduce_func, apply_node_func)``, where ``dst``
        are the destinations of the ``edges``.

        Parameters
        ----------
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).send_and_recv(edges, message_func, reduce_func,
                                                  apply_node_func, inplace=True)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=True):
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

        In the graph store, all updates are written inplace.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        Parameters
        ----------
        v : int, iterable of int, or tensor
            The node(s) to be updated.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).pull(v, message_func, reduce_func,
                                         apply_node_func, inplace=True)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=True):
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

        In the graph store, all updates are written inplace.

        Parameters
        ----------
        u : int, iterable of int, or tensor
            The node(s) to push messages out.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        inplace: bool, optional
            The value is always True.
        """
        super(BaseGraphStore, self).push(u, message_func, reduce_func,
                                         apply_node_func, inplace=True)


    def update_all(self, message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """ Distribute the computation in update_all among all pre-defined workers.

        update_all requires that all workers invoke this method and will
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
                              multigraph=None, port=8000):
    """Create the graph store server.

    The server loads graph structure and node embeddings and edge embeddings.

    Currently, only shared-memory graph store server is supported, so `store_type`
    can only be "shared_mem".

    After the server runs, the graph store clients can access the graph data
    with the specified graph name.

    DGL graph accepts graph data of multiple formats:

    * NetworkX graph,
    * scipy matrix,
    * DGLGraph.

    If the input graph data is DGLGraph, the constructed DGLGraph only contains
    its graph index.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph.
    graph_name : string
        Define the name of the graph.
    store_type : string
        The type of the graph store. The current option is "shared_mem".
    num_workers : int
        The number of workers that will connect to the server.
    multigraph : bool, optional
        Deprecated (Will be deleted in the future).
        Whether the graph would be a multigraph (default: True)
    port : int
        The port that the server listens to.

    Returns
    -------
    SharedMemoryStoreServer
        The graph store server
    """
    if multigraph is not None:
        dgl_warning("multigraph is deprecated." \
                    "DGL treat all graphs as multigraph by default.")
    return SharedMemoryStoreServer(graph_data, graph_name, None,
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
