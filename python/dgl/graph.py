"""Base graph class specialized for neural networks on graphs.
"""
from __future__ import absolute_import

import networkx as nx
import numpy as np

import dgl
from .base import ALL, is_all, DGLError, dgl_warning
from . import backend as F
from .frame import FrameRef, Frame, merge_frames
from .function.message import BundledMessageFunction
from .function.reducer import BundledReduceFunction
from .graph_index import GraphIndex, create_graph_index
from . import scheduler
from .udf import NodeBatch, EdgeBatch
from . import utils
from .view import NodeView, EdgeView

__all__ = ['DGLGraph']

class DGLGraph(object):
    """Base graph class specialized for neural networks on graphs.

    TODO(minjie): document of batching semantics

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    node_frame : FrameRef
        Node feature storage.
    edge_frame : FrameRef
        Edge feature storage.
    multigraph : bool, optional
        Whether the graph would be a multigraph (default: False)
    readonly : bool, optional
        Whether the graph structure is read-only (default: False).
    """
    def __init__(self,
                 graph_data=None,
                 node_frame=None,
                 edge_frame=None,
                 multigraph=False,
                 readonly=False):
        # graph
        self._readonly=readonly
        self._graph = create_graph_index(graph_data, multigraph, readonly)
        # frame
        self._node_frame = node_frame if node_frame is not None else FrameRef()
        self._edge_frame = edge_frame if edge_frame is not None else FrameRef()
        # msg graph & frame
        self._msg_graph = create_graph_index(multigraph=multigraph)
        self._msg_frame = FrameRef()
        self._msg_edges = []
        self.reset_messages()
        # registered functions
        self._message_func = None
        self._reduce_func = None
        self._edge_func = None
        self._apply_node_func = None

    def add_nodes(self, num, reprs=None):
        """Add nodes.

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        reprs : dict
            Optional node representations.
        """
        self._graph.add_nodes(num)
        self._msg_graph.add_nodes(num)
        #TODO(minjie): change frames
        assert reprs is None

        # Initialize feature placeholders if there are features existing
        if self._node_frame.num_columns > 0 and self._node_frame.num_rows > 0:
            self._node_frame.add_rows(num)

    def add_edge(self, u, v, reprs=None):
        """Add one edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        reprs : dict
            Optional edge representation.
        """
        self._graph.add_edge(u, v)
        #TODO(minjie): change frames
        assert reprs is None

        # Initialize feature placeholders if there are features existing
        if self._edge_frame.num_columns > 0 and self._edge_frame.num_rows > 0:
            self._edge_frame.add_rows(1)

    def add_edges(self, u, v, reprs=None):
        """Add many edges.

        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.
        reprs : dict
            Optional node representations.
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        self._graph.add_edges(u, v)
        #TODO(minjie): change frames
        assert reprs is None

        # Initialize feature placeholders if there are features existing
        if self._edge_frame.num_columns > 0 and self._edge_frame.num_rows > 0:
            self._edge_frame.add_rows(len(u))

    def clear(self):
        """Clear the graph and its storage."""
        self._graph.clear()
        self._node_frame.clear()
        self._edge_frame.clear()
        self._msg_graph.clear()
        self._msg_frame.clear()
        self._msg_edges.clear()

    def reset_messages(self):
        """Clear all messages."""
        self._msg_graph.clear()
        self._msg_frame.clear()
        self._msg_edges.clear()
        self._msg_graph.add_nodes(self.number_of_nodes())

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return self._graph.number_of_nodes()

    def __len__(self):
        """Return the number of nodes."""
        return self.number_of_nodes()

    @property
    def is_multigraph(self):
        """Whether the graph is a multigraph.
        """
        return self._graph.is_multigraph()

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return self._graph.number_of_edges()

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists
        """
        return self.has_node(vid)

    def __contains__(self, vid):
        """Same as has_node."""
        return self.has_node(vid)

    def has_nodes(self, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        vid : list, tensor
            The nodes

        Returns
        -------
        tensor
            0-1 array indicating existence
        """
        vids = utils.toindex(vids)
        rst = self._graph.has_nodes(vids)
        return rst.tousertensor()

    def has_edge_between(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists
        """
        return self._graph.has_edge_between(u, v)

    def has_edges_between(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.

        Returns
        -------
        tensor
            0-1 array indicating existence
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        rst = self._graph.has_edges_between(u, v)
        return rst.tousertensor()

    def predecessors(self, v, radius=1):
        """Return the predecessors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        tensor
            Array of predecessors
        """
        return self._graph.predecessors(v).tousertensor()

    def successors(self, v, radius=1):
        """Return the successors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        tensor
            Array of successors
        """
        return self._graph.successors(v).tousertensor()

    def edge_id(self, u, v, force_multi=False):
        """Return the id of the edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        force_multi : bool
            If False, will return a single edge ID if the graph is a simple graph.
            If True, will always return an array.

        Returns
        -------
        int or tensor
            The edge id if force_multi == True and the graph is a simple graph.
            The edge id array otherwise.
        """
        idx = self._graph.edge_id(u, v)
        return idx.tousertensor() if force_multi or self.is_multigraph else idx[0]

    def edge_ids(self, u, v, force_multi=False):
        """Return the edge ids.

        Parameters
        ----------
        u : list, tensor
            The src nodes.
        v : list, tensor
            The dst nodes.
        force_multi : bool
            If False, will return a single edge ID array if the graph is a simple graph.
            If True, will always return 3 arrays (src nodes, dst nodes, edge ids).

        Returns
        -------
        tensor, or (tensor, tensor, tensor)
        If force_multi is True or the graph is multigraph, return (src nodes, dst nodes, edge ids)
        Otherwise, return a single tensor of edge ids.
        """
        u = utils.toindex(u)
        v = utils.toindex(v)
        src, dst, eid = self._graph.edge_ids(u, v)
        if force_multi or self.is_multigraph:
            return src.tousertensor(), dst.tousertensor(), eid.tousertensor()
        else:
            return eid.tousertensor()

    def find_edges(self, eid):
        """Given the edge ids, return their source and destination node ids.

        Parameters
        ----------
        eid : list, tensor
            The edge ids.

        Returns
        -------
        tensor, tensor
        The source and destination node IDs.
        """
        eid = utils.toindex(u)
        src, dst, _ = self._graph.find_edges(eid)
        return src.tousertensor(), dst.tousertensor()

    def in_edges(self, v, form='uv'):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor

        Returns
        -------
        A tuple of Tensors (u, v, eid) if form == 'all'
        A pair of Tensors (u, v) if form == 'uv'
        One Tensor if form == 'eid'
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.in_edges(v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def out_edges(self, v, form='uv'):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : int, list, tensor
            The node(s).
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor

        Returns
        -------
        A tuple of Tensors (u, v, eid) if form == 'all'
        A pair of Tensors (u, v) if form == 'uv'
        One Tensor if form == 'eid'
        """
        v = utils.toindex(v)
        src, dst, eid = self._graph.out_edges(v)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def all_edges(self, form='uv', sorted=False):
        """Return all the edges.

        Parameters
        ----------
        form : str, optional
            The return form. Currently support:
            - 'all' : a tuple (u, v, eid)
            - 'uv'  : a pair (u, v), default
            - 'eid' : one eid tensor
        sorted : bool
            True if the returned edges are sorted by their src and dst ids.

        Returns
        -------
        A tuple of Tensors (u, v, eid) if form == 'all'
        A pair of Tensors (u, v) if form == 'uv'
        One Tensor if form == 'eid'
        """
        src, dst, eid = self._graph.edges(sorted)
        if form == 'all':
            return (src.tousertensor(), dst.tousertensor(), eid.tousertensor())
        elif form == 'uv':
            return (src.tousertensor(), dst.tousertensor())
        elif form == 'eid':
            return eid.tousertensor()
        else:
            raise DGLError('Invalid form:', form)

    def in_degree(self, v):
        """Return the in degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return self._graph.in_degree(v)

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : list, tensor
            The nodes.

        Returns
        -------
        tensor
            The in degree array.
        """
        v = utils.toindex(v)
        return self._graph.in_degrees(v).tousertensor()

    def out_degree(self, v):
        """Return the out degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return self._graph.out_degree(v)

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : list, tensor
            The nodes.

        Returns
        -------
        tensor
            The out degree array.
        """
        v = utils.toindex(v)
        return self._graph.out_degrees(v).tousertensor()

    def to_networkx(self, node_attrs=None, edge_attrs=None):
        """Convert to networkx graph.

        The edge id will be saved as the 'id' edge attribute.

        Parameters
        ----------
        node_attrs : iterable of str, optional
            The node attributes to be copied.
        edge_attrs : iterable of str, optional
            The edge attributes to be copied.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        nx_graph = self._graph.to_networkx()
        #TODO(minjie): attributes
        dgl_warning('to_networkx currently does not support converting'
                    ' node/edge features automatically.')
        return nx_graph

    def from_networkx(self, nx_graph, node_attrs=None, edge_attrs=None):
        """Convert from networkx graph.

        If 'id' edge attribute exists, the edge will be added follows
        the edge id order. Otherwise, order is undefined.

        Parameters
        ----------
        nx_graph : networkx.DiGraph
            The nx graph
        node_attrs : iterable of str, optional
            The node attributes needs to be copied.
        edge_attrs : iterable of str, optional
            The edge attributes needs to be copied.
        """
        self.clear()
        self._graph.from_networkx(nx_graph)
        self._msg_graph.add_nodes(self._graph.number_of_nodes())
        # copy attributes
        def _batcher(lst):
            if F.is_tensor(lst[0]):
                return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
            else:
                return F.tensor(lst)
        if node_attrs is not None:
            attr_dict = {attr : [] for attr in node_attrs}
            for nid in range(self.number_of_nodes()):
                for attr in node_attrs:
                    attr_dict[attr].append(nx_graph.nodes[nid][attr])
            for attr in node_attrs:
                self._node_frame[attr] = _batcher(attr_dict[attr])
        if edge_attrs is not None:
            attr_dict = {attr : [] for attr in edge_attrs}
            src, dst, _ = self._graph.edges()
            for u, v in zip(src.tolist(), dst.tolist()):
                for attr in edge_attrs:
                    attr_dict[attr].append(nx_graph.edges[u, v][attr])
            for attr in edge_attrs:
                self._edge_frame[attr] = _batcher(attr_dict[attr])

    def from_scipy_sparse_matrix(self, a):
        """ Convert from scipy sparse matrix.

        Parameters
        ----------
        a : scipy sparse matrix
            The graph's adjacency matrix
        """
        self.clear()
        self._graph.from_scipy_sparse_matrix(a)
        self._msg_graph.add_nodes(self._graph.number_of_nodes())

    def node_attr_schemes(self):
        """Return the node feature schemes.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        return self._node_frame.schemes

    def edge_attr_schemes(self):
        """Return the edge feature schemes.

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.
        """
        return self._edge_frame.schemes

    def set_n_initializer(self, initializer):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape and data type.

        Parameters
        ----------
        initializer : callable
            The initializer.
        """
        self._node_frame.set_initializer(initializer)

    def set_e_initializer(self, initializer):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape and data type.

        Parameters
        ----------
        initializer : callable
            The initializer.
        """
        self._edge_frame.set_initializer(initializer)

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data."""
        return NodeView(self)

    @property
    def ndata(self):
        """Return the data view of all the nodes."""
        return self.nodes[:].data

    @property
    def edges(self):
        """Return a edges view that can used to set/get feature data."""
        return EdgeView(self)

    @property
    def edata(self):
        """Return the data view of all the edges."""
        return self.edges[:].data

    def set_n_repr(self, hu, u=ALL, inplace=False):
        """Set node(s) representation.

        `hu` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All update will be done out-placely to work with autograd unless the inplace
        flag is true.

        Parameters
        ----------
        hu : dict of tensor
            Node representation.
        u : node, container or tensor
            The node(s).
        inplace : bool
            True if the update is done inplacely
        """
        # sanity check
        if not utils.is_dict_like(hu):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(hu))
        if is_all(u):
            num_nodes = self.number_of_nodes()
        else:
            u = utils.toindex(u)
            num_nodes = len(u)
        for key, val in hu.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))
        # set
        if is_all(u):
            for key, val in hu.items():
                self._node_frame[key] = val
        else:
            self._node_frame.update_rows(u, hu, inplace=inplace)

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
            return dict(self._node_frame)
        else:
            u = utils.toindex(u)
            return self._node_frame.select_rows(u)

    def pop_n_repr(self, key):
        """Get and remove the specified node repr.

        Parameters
        ----------
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        return self._node_frame.pop(key)

    def set_e_repr(self, he, edges=ALL, inplace=False):
        """Set edge(s) representation.

        `he` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out-placely to work with autograd unless the inplace
        flag is true.

        Parameters
        ----------
        he : tensor or dict of tensor
            Edge representation.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        inplace : bool
            True if the update is done inplacely
        """
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        # sanity check
        if not utils.is_dict_like(he):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(he))

        if is_all(eid):
            num_edges = self.number_of_edges()
        else:
            eid = utils.toindex(eid)
            num_edges = len(eid)
        for key, val in he.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_edges:
                raise DGLError('Expect number of features to match number of edges.'
                               ' Got %d and %d instead.' % (nfeats, num_edges))
        # set
        if is_all(eid):
            # update column
            for key, val in he.items():
                self._edge_frame[key] = val
        else:
            # update row
            self._edge_frame.update_rows(eid, he, inplace=inplace)

    def get_e_repr(self, edges=ALL):
        """Get node(s) representation.

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
        if len(self.edge_attr_schemes()) == 0:
            return dict()
        # parse argument
        if is_all(edges):
            eid = ALL
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            _, _, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)

        if is_all(eid):
            return dict(self._edge_frame)
        else:
            eid = utils.toindex(eid)
            return self._edge_frame.select_rows(eid)
        
    def pop_e_repr(self, key):
        """Get and remove the specified edge repr.

        Parameters
        ----------
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        return self._edge_frame.pop(key)

    def register_edge_func(self, edge_func):
        """Register global edge update function.

        Parameters
        ----------
        edge_func : callable
          Message function on the edge.
        """
        self._edge_func = edge_func

    def register_message_func(self, message_func):
        """Register global message function.

        Parameters
        ----------
        message_func : callable
          Message function on the edge.
        """
        self._message_func = message_func

    def register_reduce_func(self, reduce_func):
        """Register global message reduce function.

        Parameters
        ----------
        reduce_func : str or callable
          Reduce function on incoming edges.
        """
        self._reduce_func = reduce_func

    def register_apply_node_func(self, apply_node_func):
        """Register global node apply function.

        Parameters
        ----------
        apply_node_func : callable
          Apply function on the node.
        """
        self._apply_node_func = apply_node_func

    def apply_nodes(self, v=ALL, apply_node_func="default"):
        """Apply the function on node representations.

        Applying a None function will be ignored.

        Parameters
        ----------
        v : int, iterable of int, tensor, optional
          The node id(s).
        apply_node_func : callable
          The apply node function.
        """
        self._apply_nodes(v, apply_node_func)

    def _apply_nodes(self, v, apply_node_func="default", reduce_accum=None):
        """Internal apply nodes

        Parameters
        ----------
        reduce_accum: dict-like
          The output of reduce func
        """
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        if not apply_node_func:
            # Skip none function call.
            if reduce_accum is not None:
                # write reduce result back
                self.set_n_repr(reduce_accum, v)
            return
        # take out current node repr
        curr_repr = self.get_n_repr(v)
        if reduce_accum is not None:
            # merge current node_repr with reduce output
            curr_repr = utils.HybridDict(reduce_accum, curr_repr)
        nb = NodeBatch(self, v, curr_repr)
        new_repr = apply_node_func(nb)
        if reduce_accum is not None:
            # merge new node_repr with reduce output
            reduce_accum.update(new_repr)
            new_repr = reduce_accum
        self.set_n_repr(new_repr, v)

    def send(self, edges=ALL, message_func="default"):
        """Send messages along the given edges.

        Parameters
        ----------
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        message_func : callable
            The message function.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        along all edges between u and v.
        """
        if message_func == "default":
            message_func = self._message_func
        assert message_func is not None
        if isinstance(message_func, (tuple, list)):
            message_func = BundledMessageFunction(message_func)

        if is_all(edges):
            eid = ALL
            u, v, _ = self._graph.edges()
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        src_data = self.get_n_repr(u)
        edge_data = self.get_e_repr(eid)
        dst_data = self.get_n_repr(v)
        eb = EdgeBatch(self, (u, v, eid),
                src_data, edge_data, dst_data)
        msgs = message_func(eb)
        self._msg_graph.add_edges(u, v)
        self._msg_frame.append(msgs)

    def update_edges(self, edges=ALL, edge_func="default"):
        """Update features on the given edges.

        Parameters
        ----------
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        edge_func : callable
            The update function.

        Notes
        -----
        On multigraphs, if u and v are specified, then all the edges
        between u and v will be updated.
        """
        if edge_func == "default":
            edge_func = self._edge_func
        assert edge_func is not None

        if is_all(edges):
            eid = ALL
            u, v, _ = self._graph.edges()
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        src_data = self.get_n_repr(u)
        edge_data = self.get_e_repr(eid)
        dst_data = self.get_n_repr(v)
        eb = EdgeBatch(self, (u, v, eid),
                src_data, edge_data, dst_data)
        self.set_e_repr(edge_func(eb), eid)

    def recv(self,
             u,
             reduce_func="default",
             apply_node_func="default"):
        """Receive and reduce in-coming messages and update representation on node u.

        TODO(minjie): document on zero-in-degree case
        TODO(minjie): document on how returned new features are merged with the old features
        TODO(minjie): document on how many times UDFs will be called

        Parameters
        ----------
        u : node, container or tensor
          The node to be updated.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        """
        if reduce_func == "default":
            reduce_func = self._reduce_func
        assert reduce_func is not None
        if isinstance(reduce_func, (list, tuple)):
            reduce_func = BundledReduceFunction(reduce_func)
        self._batch_recv(u, reduce_func)
        # optional apply nodes
        self.apply_nodes(u, apply_node_func)

    def _batch_recv(self, v, reduce_func):
        if self._msg_frame.num_rows == 0:
            # no message has ever been sent
            return

        v_is_all = is_all(v)
        if v_is_all:
            v = F.arange(0, self.number_of_nodes())
        elif isinstance(v, int):
            v = [v]
        v = utils.toindex(v)
        if len(v) == 0:
            # no vertex to be triggered.
            return

        # degree bucketing
        degrees, v_buckets = scheduler.degree_bucketing(self._msg_graph, v)
        if degrees == [0]:
            # no message has been sent to the specified node
            return

        reordered_v = []
        new_reprs = []
        has_zero_degree = False
        for deg, v_bkt in zip(degrees, v_buckets):
            if deg == 0:
                # no need to trigger reduce func for zero-degree nodes
                has_zero_degree = True
                continue
            bkt_len = len(v_bkt)
            v_data = self.get_n_repr(v_bkt)
            uu, vv, in_msg_ids = self._msg_graph.in_edges(v_bkt)
            in_msgs = self._msg_frame.select_rows(in_msg_ids)
            # Reshape the column tensor to (B, Deg, ...).
            def _reshape_fn(msg):
                msg_shape = F.shape(msg)
                new_shape = (bkt_len, deg) + msg_shape[1:]
                return F.reshape(msg, new_shape)
            reshaped_in_msgs = utils.LazyDict(
                    lambda key: _reshape_fn(in_msgs[key]), self._msg_frame.schemes)
            reordered_v.append(v_bkt.tousertensor())
            nb = NodeBatch(self, v_bkt, v_data, reshaped_in_msgs)
            new_reprs.append(reduce_func(nb))

        # TODO(minjie): clear partial messages
        self.reset_messages()

        # Pack all reducer results together
        reordered_v = F.cat(reordered_v, dim=0)
        keys = new_reprs[0].keys()
        new_reprs = {key : F.cat([repr[key] for repr in new_reprs], dim=0)
                     for key in keys}

        if v_is_all and not has_zero_degree:
            # First do reorder and then replace the whole column.
            _, indices = F.sort_1d(reordered_v)
            indices = utils.toindex(indices)
            new_reprs = utils.reorder(new_reprs, indices)
            self.set_n_repr(new_reprs)
        else:
            # Use setter to do reorder.
            self.set_n_repr(new_reprs, reordered_v)

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default"):
        """Send messages along edges and receive them on the targets.

        Parameters
        ----------
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        message_func : callable, optional
            The message function. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            The reduce function. Registered function will be used if not
            specified.
        apply_node_func : callable, optional
            The update function. Registered function will be used if not
            specified.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        and received along all edges between u and v.
        """
        if message_func == "default":
            message_func = self._message_func
        elif isinstance(message_func, (tuple, list)):
            message_func = BundledMessageFunction(message_func)
        if reduce_func == "default":
            reduce_func = self._reduce_func
        elif isinstance(reduce_func, (list, tuple)):
            reduce_func = BundledReduceFunction(reduce_func)
        assert message_func is not None
        assert reduce_func is not None

        if isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        if len(u) == 0:
            # no edges to be triggered
            return

        if not self.is_multigraph:
            executor = scheduler.get_executor(
                    'send_and_recv', self, src=u, dst=v,
                    message_func=message_func, reduce_func=reduce_func)
        else:
            executor = None

        if executor:
            accum = executor.run()
            unique_v = executor.recv_nodes
        else:
            # message func
            src_data = self.get_n_repr(u)
            edge_data = self.get_e_repr(eid)
            dst_data = self.get_n_repr(v)
            eb = EdgeBatch(self, (u, v, eid),
                    src_data, edge_data, dst_data)
            msgs = message_func(eb)
            msg_frame = FrameRef(Frame(msgs))
            # recv with degree bucketing
            executor = scheduler.get_recv_executor(graph=self,
                                                   reduce_func=reduce_func,
                                                   message_frame=msg_frame,
                                                   edges=(u, v))
            assert executor is not None
            accum = executor.run()
            unique_v = executor.recv_nodes

        self._apply_nodes(unique_v, apply_node_func, reduce_accum=accum)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default"):
        """Pull messages from the node's predecessors and then update it.

        Parameters
        ----------
        v : node, container or tensor
          The node to be updated.
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        """
        v = utils.toindex(v)
        if len(v) == 0:
            return
        uu, vv, _ = self._graph.in_edges(v)
        self.send_and_recv((uu, vv), message_func, reduce_func, apply_node_func=None)
        unique_v = F.unique(v.tousertensor())
        self.apply_nodes(unique_v, apply_node_func)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default"):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : node, container or tensor
          The node that sends out messages.
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable
          The update function.
        """
        u = utils.toindex(u)
        if len(u) == 0:
            return
        uu, vv, _ = self._graph.out_edges(u)
        self.send_and_recv((uu, vv), message_func,
                reduce_func, apply_node_func)

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Send messages through all the edges and update all nodes.

        Parameters
        ----------
        message_func : callable
          The message function.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        assert message_func is not None
        assert reduce_func is not None

        executor = scheduler.get_executor(
                "update_all", self, message_func=message_func, reduce_func=reduce_func)
        if executor:
            new_reprs = executor.run()
            self._apply_nodes(ALL, apply_node_func, reduce_accum=new_reprs)
        else:
            self.send(ALL, message_func)
            self.recv(ALL, reduce_func, apply_node_func)

    def propagate(self,
                  traverser='topo',
                  message_func="default",
                  reduce_func="default",
                  apply_node_func="default",
                  **kwargs):
        """Propagate messages and update nodes using graph traversal.

        A convenient function for passing messages and updating
        nodes according to the traverser. The traverser can be
        any of the pre-defined traverser (e.g. 'topo'). User can also provide custom
        traverser that generates the edges and nodes.

        Parameters
        ----------
        traverser : str or generator of edges.
          The traverser of the graph.
        message_func : str or callable
          The message function.
        reduce_func : str or callable
          The reduce function.
        apply_node_func : str or callable
          The update function.
        kwargs : keyword arguments, optional
            Arguments for pre-defined iterators.
        """
        if isinstance(traverser, str):
            # TODO(minjie): Call pre-defined routine to unroll the computation.
            raise RuntimeError('Not implemented.')
        else:
            # NOTE: the iteration can return multiple edges at each step.
            for u, v in traverser:
                self.send_and_recv((u, v),
                        message_func, reduce_func, apply_node_func)

    def subgraph(self, nodes):
        """Generate the subgraph among the given nodes.

        Parameters
        ----------
        nodes : list, or iterable
            A container of the nodes to construct subgraph.

        Returns
        -------
        G : DGLSubGraph
            The subgraph.
        """
        induced_nodes = utils.toindex(nodes)
        sgi = self._graph.node_subgraph(induced_nodes)
        return dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges,
                sgi, readonly=self._readonly)

    def subgraphs(self, nodes):
        """Generate the subgraphs among the given nodes.

        Parameters
        ----------
        nodes : a list of lists or iterable
            A list of the nodes to construct subgraph.

        Returns
        -------
        G : A list of DGLSubGraph
            The subgraphs.
        """
        induced_nodes = [utils.toindex(n) for n in nodes]
        sgis = self._graph.node_subgraphs(induced_nodes)
        return [dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges,
            sgi, readonly=self._readonly) for sgi in sgis]

    def edge_subgraph(self, edges):
        """Generate the subgraph among the given edges.

        Parameters
        ----------
        edges : list, or iterable
            A container of the edges to construct subgraph.

        Returns
        -------
        G : DGLSubGraph
            The subgraph.
        """
        induced_edges = utils.toindex(edges)
        sgi = self._graph.edge_subgraph(induced_edges)
        return dgl.DGLSubGraph(self, sgi.induced_nodes, sgi.induced_edges, sgi)

    def merge(self, subgraphs, reduce_func='sum'):
        """Merge subgraph features back to this parent graph.

        Parameters
        ----------
        subgraphs : iterator of DGLSubGraph
            The subgraphs to be merged.
        reduce_func : str
            The reduce function (only 'sum' is supported currently)
        """
        # sanity check: all the subgraphs and the parent graph
        # should have the same node/edge feature schemes.
        # merge node features
        to_merge = []
        for sg in subgraphs:
            if len(sg.node_attr_schemes()) == 0:
                continue
            if sg.node_attr_schemes() != self.node_attr_schemes():
                raise RuntimeError('Subgraph and parent graph do not '
                                   'have the same node attribute schemes.')
            to_merge.append(sg)
        self._node_frame = merge_frames(
                [sg._node_frame for sg in to_merge],
                [sg._parent_nid for sg in to_merge],
                self._node_frame.num_rows,
                reduce_func)

        # merge edge features
        to_merge.clear()
        for sg in subgraphs:
            if len(sg.edge_attr_schemes()) == 0:
                continue
            if sg.edge_attr_schemes() != self.edge_attr_schemes():
                raise RuntimeError('Subgraph and parent graph do not '
                                   'have the same edge attribute schemes.')
            to_merge.append(sg)
        self._edge_frame = merge_frames(
                [sg._edge_frame for sg in to_merge],
                [sg._parent_eid for sg in to_merge],
                self._edge_frame.num_rows,
                reduce_func)

    def adjacency_matrix(self, ctx=None):
        """Return the adjacency matrix representation of this graph.

        Parameters
        ----------
        ctx : optional
            The context of returned adjacency matrix.

        Returns
        -------
        sparse_tensor
            The adjacency matrix.
        """
        return self._graph.adjacency_matrix().get(ctx)

    def incidence_matrix(self, oriented=False, ctx=None):
        """Return the incidence matrix representation of this graph.

        Parameters
        ----------
        oriented : bool, optional
            Whether the returned incidence matrix is oriented.

        ctx : optional
            The context of returned incidence matrix.

        Returns
        -------
        sparse_tensor
            The incidence matrix.
        """
        return self._graph.incidence_matrix(oriented).get(ctx)

    def line_graph(self, backtracking=True, shared=False):
        """Return the line graph of this graph.

        Parameters
        ----------
        backtracking : bool, optional
            Whether the returned line graph is backtracking.

        shared : bool, optional
            Whether the returned line graph shares representations with `self`.

        Returns
        -------
        DGLGraph
            The line graph of this graph.
        """
        graph_data = self._graph.line_graph(backtracking)
        node_frame = self._edge_frame if shared else None
        return DGLGraph(graph_data, node_frame)

    def filter_nodes(self, predicate, nodes=ALL):
        """Return a tensor of node IDs that satisfy the given predicate.

        Parameters
        ----------
        predicate : callable
            The predicate should take in a dict of tensors whose values
            are concatenation of node representations by node ID (same as
            get_n_repr()), and return a boolean tensor with N elements
            indicating which node satisfy the predicate.
        nodes : container or tensor
            The nodes to filter on

        Returns
        -------
        tensor
            The filtered nodes
        """
        n_repr = self.get_n_repr(nodes)
        n_mask = predicate(n_repr)

        if is_all(nodes):
            return F.nonzero_1d(n_mask)
        else:
            nodes = F.tensor(nodes)
            return nodes[n_mask]

    def filter_edges(self, predicate, edges=ALL):
        """Return a tensor of edge IDs that satisfy the given predicate.

        Parameters
        ----------
        predicate : callable
            The predicate should take in a dict of tensors whose values
            are concatenation of edge representations by edge ID,
            and return a boolean tensor with N elements indicating which
            node satisfy the predicate.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        tensor
            The filtered edges
        """
        e_repr = self.get_e_repr(edges)
        e_mask = predicate(e_repr)

        if is_all(edges):
            return F.nonzero_1d(e_mask)
        else:
            edges = F.tensor(edges)
            return edges[e_mask]
