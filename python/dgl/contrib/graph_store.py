import os
import time
import posix_ipc as ipc

from collections.abc import MutableMapping

from ..base import ALL, is_all, DGLError
from .. import backend as F
from ..graph import DGLGraph
from .._ffi.function import _init_api
from .. import ndarray as nd

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, ndata_name):
    return "/" + graph_name + "_edge_" + ndata_name

class InitNData:
    def __init__(self, serv, args):
        self.graph = serv._graph
        self.graph_name = serv._graph_name
        args = args.split(',')
        self.ndata_name = args[1]
        self.num_feats = int(args[2])
        self.dtype = int(args[3])

    def run(self):
        if self.ndata_name in self.graph.ndata:
            ndata = self.graph.ndata[self.ndata_name]
            assert ndata.shape[1] == self.num_feats
            return

        data = _CAPI_DGLCreateSharedMem(_get_ndata_path(self.graph_name, self.ndata_name),
                                        self.graph.number_of_nodes(),
                                        self.num_feats, self.dtype, "zero", True)
        dlpack = data.to_dlpack()
        self.graph.ndata[self.ndata_name] = F.zerocopy_from_dlpack(dlpack)

class Init:
    def __init__(self, serv, args):
        args = args.split(',')
        self.pid = int(args[0])
        self.pipe_path = args[1]
        self.serv = serv

    def run(self):
        self.serv._reply_pipes[self.pid] = os.open(self.pipe_path, os.O_WRONLY)

class Terminate:
    def __init__(self, serv, args):
        self.pid = int(args)
        self.serv = serv

    def run(self):
        self.serv._num_workers -= 1

_commands = {'init_ndata': lambda serv, args: InitNData(serv, args),
             'init':       lambda serv, args: Init(serv, args),
             'terminate':  lambda serv, args: Terminate(serv, args)}

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
        dlpack = F.zerocopy_to_dlpack(val)
        dgl_tensor = nd.from_dlpack(dlpack)
        val = _CAPI_DGLCreateSharedMemWithData(_get_ndata_path(self._graph_name, key),
                                                dgl_tensor)
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
        dlpack = F.zerocopy_to_dlpack(val)
        dgl_tensor = nd.from_dlpack(dlpack)
        val = _CAPI_DGLCreateSharedMemWithData(_get_edata_path(self._graph_name, key),
                                                dgl_tensor)
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

class SharedMemoryStoreServer:
    def __init__(self, graph_data, graph_name, multigraph,
                 num_workers):
        self._graph = DGLGraph(graph_data, multigraph=multigraph, readonly=False)
        self._num_workers = num_workers
        self._graph_name = graph_name
        self._reply_pipes = {}
        self._msg_queue = ipc.MessageQueue("/" + graph_name, flags=ipc.O_CREAT)

    def __del__(self):
        for key in self._reply_pipes:
            os.close(self._reply_pipes[key])
        if self._msg_queue is not None:
            self._msg_queue.close()
            self._msg_queue.unlink()

    def _decode_command(self, line):
        parts = line[0].decode('utf-8').split(':')
        return _commands[parts[0]](self, parts[1])

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
        while self._num_workers > 0:
            line = self._msg_queue.receive()
            comm = self._decode_command(line)
            comm.run()
        self._msg_queue.close()
        self._msg_queue.unlink()
        self._msg_queue = None

class SharedMemoryGraphStore:
    def __init__(self, graph_data, graph_name, multigraph=False):
        self._graph = DGLGraph(graph_data, multigraph=multigraph, readonly=False)
        self._graph_name = graph_name
        self._msg_queue = ipc.MessageQueue("/" + graph_name)

        self._pid = os.getpid()
        #self._recv_path = "/" + graph_name + "-" + str(self._pid)
        #self._comm_pipe.write(self._encode_comm("init:", [self._recv_path]))
        #self._comm_pipe.flush()
        #os.mkfifo(self._recv_path)
        #self._recv_pipe = os.open(self._recv_path, os.O_RDONLY)

    def __del__(self):
        self._msg_queue.close()
        #os.close(self._recv_pipe)
        #os.unlink(self._recv_path)

    def _encode_comm(self, comm, args):
        comm = comm + str(self._pid)
        for arg in args:
            comm = comm + "," + arg
        comm = comm + '\n'
        return comm

    def init_ndata(self, ndata_name, num_feats, dtype):
        self._msg_queue.send(self._encode_comm("init_ndata:", [ndata_name, str(num_feats), str(dtype)]))
        #TODO remove sleep later
        time.sleep(1)
        data = _CAPI_DGLCreateSharedMem(_get_ndata_path(self._graph_name, ndata_name),
                                        self._graph.number_of_nodes(),
                                        num_feats, dtype, "zero", False)
        dlpack = data.to_dlpack()
        self._graph.ndata[ndata_name] = F.zerocopy_from_dlpack(dlpack)

    def destroy(self):
        self._msg_queue.send(self._encode_comm("terminate:", []))

    def register_message_func(self, func):
        """Register global message function.

        Once registered, ``func`` will be used as the default
        message function in message passing operations, including
        :func:`send`, :func:`send_and_recv`, :func:`pull`,
        :func:`push`, :func:`update_all`.

        Parameters
        ----------
        func : callable
            Message function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        See Also
        --------
        send
        send_and_recv
        pull
        push
        update_all
        """
        self._graph.register_message_func(func)

    def register_reduce_func(self, func):
        """Register global message reduce function.

        Once registered, ``func`` will be used as the default
        message reduce function in message passing operations, including
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        recv
        send_and_recv
        push
        pull
        update_all
        """
        self._graph.register_reduce_func(func)

    def register_apply_node_func(self, func):
        """Register global node apply function.

        Once registered, ``func`` will be used as the default apply
        node function. Related operations include :func:`apply_nodes`,
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        apply_nodes
        register_apply_edge_func
        """
        self._graph.register_apply_node_func(func)

    def register_apply_edge_func(self, func):
        """Register global edge apply function.

        Once registered, ``func`` will be used as the default apply
        edge function in :func:`apply_edges`.

        Parameters
        ----------
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        See Also
        --------
        apply_edges
        register_apply_node_func
        """
        self._graph.register_apply_edge_func(func)

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data

        See Also
        --------
        dgl.DGLGraph.nodes
        """
        return self._graph.ndata

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
        return self._graph.edata

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the nodes to update their features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable or None, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : int, iterable of int, tensor, optional
            The node (ids) on which to apply ``func``. The default
            value is all the nodes.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        See Also
        --------
        register_apply_node_func
        apply_edges
        """
        self._graph.apply_nodes(func, v, inplace)

    def apply_edges(self, func="default", edges=ALL, inplace=False):
        """Apply the function on the edges to update their features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, optional
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : valid edges type, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then all the edges
        between :math:`u` and :math:`v` will be updated.

        See Also
        --------
        apply_nodes
        """
        self._graph.apply_edges(func, edges, inplace)

    def group_apply_edges(self, group_by, func, edges=ALL, inplace=False):
        """Group the edges by nodes and apply the function on the grouped edges to
         update their features.

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
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then all the edges
        between :math:`u` and :math:`v` will be updated.

        See Also
        --------
        apply_edges
        """
        self._graph.group_apply_edges(group_by, func, edges, inplace)

    def send(self, edges=ALL, message_func="default"):
        """Send messages along the given edges.

        ``edges`` can be any of the following types:

        * ``int`` : Specify one edge using its edge id.
        * ``pair of int`` : Specify one edge using its endpoints.
        * ``int iterable`` / ``tensor`` : Specify multiple edges using their edge ids.
        * ``pair of int iterable`` / ``pair of tensors`` :
          Specify multiple edges using their endpoints.

        The UDF returns messages on the edges and can be later fetched in
        the destination node's ``mailbox``. Receiving will consume the messages.
        See :func:`recv` for example.

        If multiple ``send`` are triggered on the same edge without ``recv``. Messages
        generated by the later ``send`` will overwrite previous messages.

        Parameters
        ----------
        edges : valid edges type, optional
            Edges on which to apply ``message_func``. Default is sending along all
            the edges.
        message_func : callable
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then the messages will be sent
        along all edges between :math:`u` and :math:`v`.

        Examples
        --------
        See the *message passing* example in :class:`DGLGraph` or :func:`recv`.
        """
        self._graph.send(edges, message_func)

    def recv(self,
             v=ALL,
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        Optionally, apply a function to update the node features after receive.

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
            If True, update will be done in place, but autograd will break.

        """
        self._graph.recv(v, reduce_func, apply_node_func, inplace)

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
                      inplace=False):
        """Send messages along edges and let destinations receive them.

        Optionally, apply a function to update the node features after receive.

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
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        and received along all edges between u and v.

        See Also
        --------
        send
        recv
        """
        self._graph.send_and_recv(edges, message_func, reduce_func, apply_node_func, inplace)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

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
            If True, update will be done in place, but autograd will break.

        See Also
        --------
        push
        """
        self._graph.pull(v, message_func, reduce_func, apply_node_func, inplace)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

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
            If True, update will be done in place, but autograd will break.

        The feature of node :math:`2` changes but the feature of node :math:`1`
        remains the same as we did not :func:`push` for node :math:`0`.

        See Also
        --------
        pull
        """
        self._graph.push(u, message_func, reduce_func, apply_node_func, inplace)

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Send messages through all edges and update all nodes.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges(), message_func)`` and
        ``recv(self, self.nodes(), reduce_func, apply_node_func)``.

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

        See Also
        --------
        send
        recv
        """
        self._graph.update_all(message_func, reduce_func, apply_node_func)

    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering
        :func:`pull()` on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        node_generators : iterable, each element is a list or a tensor of node ids
            The generator of node frontiers. It specifies which nodes perform
            :func:`pull` at each timestep.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        prop_edges
        """
        self._graph.prop_nodes(nodes_generator, message_func, reduce_func, apply_node_func)

    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering
        :func:`send_and_recv()` on edges.

        The traversal order is specified by the ``edges_generator``. It generates
        edge frontiers. The edge frontiers should be of *valid edges type*.
        See :func:`send` for more details.

        Edges in the same frontier will be triggered together, while edges in
        different frontiers will be triggered according to the generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

        See Also
        --------
        prop_nodes
        """
        self._graph.prop_edges(edges_generator, message_func, reduce_func, apply_node_func)


def create_graph_store_server(graph_data, graph_name, store_type,
                           multigraph, num_workers):
    return SharedMemoryStoreServer(graph_data, graph_name, multigraph, num_workers)

def create_graph_store_client(graph_data, graph_name, store_type, multigraph):
    return SharedMemoryGraphStore(graph_data, graph_name, multigraph)


_init_api("dgl.contrib.graph_store")
