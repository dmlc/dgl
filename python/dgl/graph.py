"""Base graph class specialized for neural networks on graphs."""
from __future__ import absolute_import

import networkx as nx
import numpy as np

import dgl
from .base import ALL, is_all, DGLError, dgl_warning
from . import backend as F
from .frame import FrameRef, Frame, merge_frames
from .graph_index import GraphIndex, create_graph_index
from . import utils
from .view import NodeView, EdgeView
from .runtime import scheduler, Runtime

__all__ = ['DGLGraph']

class DGLGraph(object):
    """Base graph class.

    The graph stores nodes, edges and also their features.

    DGL graph is always directional. Undirected graph can be represented using
    two bi-directional edges.

    Nodes are identified by consecutive integers starting from zero.

    Edges can be specified by two end points (u, v) or the integer id assigned
    when the edges are added.

    Node and edge features are stored as a dictionary from the feature name
    to the feature data (in tensor).

    Parameters
    ----------
    graph_data : graph data, optional
        Data to initialize graph. Same as networkx's semantics.
    node_frame : FrameRef, optional
        Node feature storage.
    edge_frame : FrameRef, optional
        Edge feature storage.
    multigraph : bool, optional
        Whether the graph would be a multigraph (default: False)
    readonly : bool, optional
        Whether the graph structure is read-only (default: False).

    Examples
    --------
    Create an empty graph with no nodes and edges.

    >>> G = dgl.DGLGraph()

    G can be grown in several ways.

    **Nodes:**

    Add N nodes:

    >>> G.add_nodes(10)  # 10 isolated nodes are added

    **Edges:**

    Add one edge at a time,

    >>> G.add_edge(0, 1)

    or multiple edges,

    >>> G.add_edges([1, 2, 3], [3, 4, 5])  # three edges: 1->3, 2->4, 3->5

    or multiple edges starting from the same node,

    >>> G.add_edges(4, [7, 8, 9])  # three edges: 4->7, 4->8, 4->9

    or multiple edges pointing to the same node,

    >>> G.add_edges([2, 6, 8], 5)  # three edges: 2->5, 6->5, 8->5

    or multiple edges using tensor type (demo in pytorch syntax).

    >>> import torch as th
    >>> G.add_edges(th.tensor([3, 4, 5]), 1)  # three edges: 3->1, 4->1, 5->1

    NOTE: Removing nodes and edges is not supported by DGLGraph.

    **Features:**

    Both nodes and edges can have feature data. Features are stored as
    key/value pair. The key must be hashable while the value must be tensor
    type. Features are batched on the first dimension.

    Use G.ndata to get/set features for all nodes.

    >>> G = dgl.DGLGraph()
    >>> G.add_nodes(3)
    >>> G.ndata['x'] = th.zeros((3, 5))  # init 3 nodes with zero vector(len=5)
    >>> G.ndata
    {'x' : tensor([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])}

    Use G.nodes to get/set features for some nodes.

    >>> G.nodes[[0, 2]].data['x'] = th.ones((2, 5))
    >>> G.ndata
    {'x' : tensor([[1., 1., 1., 1., 1.],
                   [0., 0., 0., 0., 0.],
                   [1., 1., 1., 1., 1.]])}

    Similarly, use G.edata and G.edges to get/set features for edges.

    >>> G.add_edges([0, 1], 2)  # 0->2, 1->2
    >>> G.edata['y'] = th.zeros((2, 4))  # init 2 edges with zero vector(len=4)
    >>> G.edata
    {'y' : tensor([[0., 0., 0., 0.],
                   [0., 0., 0., 0.]])}
    >>> G.edges[1, 2].data['y'] = th.ones((1, 4))
    >>> G.edata
    {'y' : tensor([[0., 0., 0., 0.],
                   [1., 1., 1., 1.]])}

    Note that each edge is assigned a unique id equal to its adding
    order. So edge 1->2 has id=1. DGL supports directly use edge id
    to access edge features.

    >>> G.edges[0].data['y'] += 2.
    >>> G.edata
    {'y' : tensor([[2., 2., 2., 2.],
                   [1., 1., 1., 1.]])}
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
        self.reset_messages()
        # registered functions
        self._message_func = None
        self._reduce_func = None
        self._apply_node_func = None
        self._apply_edge_func = None

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

        See Also
        --------
        add_edges
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

        See Also
        --------
        add_edge
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

    def reset_messages(self):
        """Clear all messages."""
        self._msg_graph.clear()
        self._msg_frame.clear()
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

        See Also
        --------
        has_nodes
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

        See Also
        --------
        has_node
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

        See Also
        --------
        has_edges_between
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

        See Also
        --------
        has_edge_between
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

        See Also
        --------
        edge_ids
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

        See Also
        --------
        edge_id
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
        tensor
            The source nodes.
        tensor
            The destination nodes.
        """
        eid = utils.toindex(eid)
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

        See Also
        --------
        in_degrees
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

        See Also
        --------
        in_degree
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

        See Also
        --------
        out_degrees
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

        See Also
        --------
        out_degree
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

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        Parameters
        ----------
        initializer : callable
            The initializer.
        """
        self._node_frame.set_initializer(initializer)

    def set_e_initializer(self, initializer):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

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

    def register_message_func(self, func):
        """Register global message function.

        Parameters
        ----------
        func : callable
          Message function on the edge.
        """
        self._message_func = func

    def register_reduce_func(self, func):
        """Register global message reduce function.

        Parameters
        ----------
        func : str or callable
          Reduce function on incoming edges.
        """
        self._reduce_func = func

    def register_apply_node_func(self, func):
        """Register global node apply function.

        Parameters
        ----------
        func : callable
            Apply function on the node.
        """
        self._apply_node_func = func

    def register_apply_edge_func(self, func):
        """Register global edge apply function.

        Parameters
        ----------
        edge_func : callable
            Apply function on the edge.
        """
        self._apply_edge_func = func

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the node features.

        Applying a None function will be ignored.

        Parameters
        ----------
        func : callable, optional
            The UDF applied on the node features.
        v : int, iterable of int, tensor, optional
            The node id(s).
        """
        if func == "default":
            func = self._apply_node_func
        execs = scheduler.get_apply_nodes_schedule(graph=self,
                                                   v=v,
                                                   apply_func=func)
        Runtime.run(execs)

    def apply_edges(self, func="default", edges=ALL):
        """Apply the function on the edge features.

        Parameters
        ----------
        func : callable, optional
            The UDF applied on the edge features.
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Notes
        -----
        On multigraphs, if u and v are specified, then all the edges
        between u and v will be updated.
        """
        if func == "default":
            func = self._apply_edge_func
        assert func is not None

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

        execs = scheduler.get_apply_edges_schedule(graph=self,
                                                   u=u,
                                                   v=v,
                                                   eid=eid,
                                                   apply_func=func)
        Runtime.run(execs)

    def send(self, edges, message_func="default"):
        """Send messages along the given edges.

        Parameters
        ----------
        edges : edges, optional
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids.
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

        execs = scheduler.get_send_schedule(graph=self,
                                            u=u,
                                            v=v,
                                            eid=eid,
                                            message_func=message_func)
        Runtime.run(execs)

        # update message graph and frame
        self._msg_graph.add_edges(u, v)

    def recv(self,
             v,
             reduce_func="default",
             apply_node_func="default"):
        """Receive and reduce in-coming messages and update representation on node v.

        TODO(minjie): document on zero-in-degree case
        TODO(minjie): document on how returned new features are merged with the old features
        TODO(minjie): document on how many times UDFs will be called

        Parameters
        ----------
        v : node, container or tensor
          The node to be updated.
        reduce_func : callable
          The reduce function.
        apply_node_func : callable, optional
          The update function.
        """
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert reduce_func is not None

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

        execs = scheduler.get_recv_schedule(graph=self,
                                            v=v,
                                            reduce_func=reduce_func,
                                            apply_func=apply_node_func)
        Runtime.run(execs)

        # clear message
        self.reset_messages()

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
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

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

        prog = scheduler.get_snr_schedule(graph=self,
                                           u=u,
                                           v=v,
                                           eid=eid,
                                           message_func=message_func,
                                           reduce_func=reduce_func,
                                           apply_func=apply_node_func)
        prog.pprint()
        #Runtime.run(execs)

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
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        v = utils.toindex(v)
        if len(v) == 0:
            return
        execs = scheduler.get_pull_schedule(graph=self,
                                            v = v,
                                            message_func=message_func,
                                            reduce_func=reduce_func,
                                            apply_func=apply_node_func)
        Runtime.run(execs)

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
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        u = utils.toindex(u)
        if len(u) == 0:
            return
        execs = scheduler.get_push_schedule(graph=self,
                                            u = u,
                                            message_func=message_func,
                                            reduce_func=reduce_func,
                                            apply_func=apply_node_func)
        Runtime.run(execs)

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
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert message_func is not None
        assert reduce_func is not None

        execs = scheduler.get_update_all_schedule(graph=self,
                                                  message_func=message_func,
                                                  reduce_func=reduce_func,
                                                  apply_func=apply_node_func)
        Runtime.run(execs)

    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering pull() on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        node_generators : generator
            The generator of node frontiers.
        message_func : callable, optional
            The message function.
        reduce_func : callable, optional
            The reduce function.
        apply_node_func : callable, optional
            The update function.
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier,
                    message_func, reduce_func, apply_node_func)

    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering send_and_recv() on edges.

        The traversal order is specified by the ``edges_generator``. It
        generates edge frontiers, which is a list or a tensor of edge ids or
        end points.  The edges in the same frontier will be triggered together,
        while edges in different frontiers will be triggered according to the
        generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable, optional
            The message function.
        reduce_func : callable, optional
            The reduce function.
        apply_node_func : callable, optional
            The update function.
        """
        for edge_frontier in edges_generator:
            self.send_and_recv(edge_frontier,
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

        See Also
        --------
        subgraphs
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

        See Also
        --------
        subgraph
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

    def adjacency_matrix(self, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        transpose : bool, optional (default=False)
            A flag to tranpose the returned adjacency matrix.
        ctx : context, optional (default=cpu)
            The context of returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        """
        return self._graph.adjacency_matrix(transpose, ctx)

    def incidence_matrix(self, type, ctx=F.cpu()):
        """Return the incidence matrix representation of this graph.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix `I`:
        * "in":
          - I[v, e] = 1 if e is the in-edge of v (or v is the dst node of e);
          - I[v, e] = 0 otherwise.
        * "out":
          - I[v, e] = 1 if e is the out-edge of v (or v is the src node of e);
          - I[v, e] = 0 otherwise.
        * "both":
          - I[v, e] = 1 if e is the in-edge of v;
          - I[v, e] = -1 if e is the out-edge of v;
          - I[v, e] = 0 otherwise (including self-loop).

        Parameters
        ----------
        type : str
            Can be either "in", "out" or "both"
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        return self._graph.incidence_matrix(type, ctx)

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

