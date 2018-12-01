"""Base graph class specialized for neural networks on graphs."""
from __future__ import absolute_import

import networkx as nx
import numpy as np

import dgl
from .base import ALL, is_all, DGLError, dgl_warning
from . import backend as F
from .frame import FrameRef, Frame, merge_frames
from .graph_index import GraphIndex, create_graph_index
from .runtime import ir, scheduler, Runtime
from . import utils
from .view import NodeView, EdgeView
from .udf import NodeBatch, EdgeBatch

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

    **Message Passing:**

    One common operation for updating node features is message passing,
    where the source nodes send messages through edges to the destinations.
    With :class:`DGLGraph`, we can do this with :func:`send` and :func:`recv`.

    In the example below, the source nodes add 1 to their node features as
    the messages and send the messages to the destinations.

    >>> # Define the function for sending node features as messages.
    >>> def send_source(edges): return {'m': edges.src['x'] + 1}
    >>> # Set the function defined to be the default message function.
    >>> G.register_message_func(send_source)
    >>> # Send messages through all edges.
    >>> G.send(G.edges())

    Just like you need to go to your mailbox for retrieving mails, the destination
    nodes also need to receive the messages and potentially update their features.

    >>> # Define a function for summing messages received and replacing the original feature.
    >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
    >>> # Set the function defined to be the default message reduce function.
    >>> G.registe r_reduce_func(simple_reduce)
    >>> # All existing edges have node 2 as the destination.
    >>> # Receive the messages for node 2 and update its feature.
    >>> G.recv(v=2)
    >>> G.ndata
    {'x': tensor([[1., 1., 1., 1., 1.],
                  [0., 0., 0., 0., 0.],
                  [3., 3., 3., 3., 3.]])} # 3 = (1 + 1) + (0 + 1)

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
        if node_frame is None:
            self._node_frame = FrameRef(Frame(num_rows=self.number_of_nodes()))
        else:
            self._node_frame = node_frame
        if edge_frame is None:
            self._edge_frame = FrameRef(Frame(num_rows=self.number_of_edges()))
        else:
            self._edge_frame = edge_frame
        # msg graph & frame
        self._msg_graph = create_graph_index(multigraph=multigraph)
        self._msg_frame = FrameRef()
        self.reset_messages()
        # registered functions
        self._message_func = None
        self._reduce_func = None
        self._apply_node_func = None
        self._apply_edge_func = None

    def add_nodes(self, num, data=None):
        """Add nodes.

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        data : dict
            Optional node feature data.
        """
        self._graph.add_nodes(num)
        self._msg_graph.add_nodes(num)
        if data is None:
            # Initialize feature placeholders if there are features existing
            self._node_frame.add_rows(num)
        else:
            self._node_frame.append(data)

    def add_edge(self, u, v, data=None):
        """Add one edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        data : dict
            Optional node feature data.

        See Also
        --------
        add_edges
        """
        self._graph.add_edge(u, v)
        if data is None:
            # Initialize feature placeholders if there are features existing
            self._edge_frame.add_rows(1)
        else:
            self._edge_frame.append(data)

    def add_edges(self, u, v, data=None):
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
        if data is None:
            # Initialize feature placeholders if there are features existing
            # NOTE: use max due to edge broadcasting syntax
            self._edge_frame.add_rows(max(len(u), len(v)))
        else:
            self._edge_frame.append(data)

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
        return self._graph.has_node(vid)

    def __contains__(self, vid):
        """Same as has_node."""
        return self._graph.has_node(vid)

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
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
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
        self._node_frame.add_rows(self.number_of_nodes())
        self._edge_frame.add_rows(self.number_of_edges())
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

    def set_n_initializer(self, initializer, field=None):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        See Also
        --------
        dgl.init.base_initializer
        """
        self._node_frame.set_initializer(initializer, field)

    def set_e_initializer(self, initializer, field=None):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        Parameters
        ----------
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.

        See Also
        --------
        dgl.init.base_initializer
        """
        self._edge_frame.set_initializer(initializer, field)

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

        Once registered, :attr:`func` will be used as the default
        message function in message passing operations, including
        :func:`send`, :func:`send_and_recv`, :func:`pull`,
        :func:`push`, :func:`update_all`.

        Parameters
        ----------
        func : callable
            Message function on the edge. ``func(edges) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`.
        """
        self._message_func = func

    def register_reduce_func(self, func):
        """Register global message reduce function.

        Once registered, :attr:`func` will be used as the default
        message reduce function in message passing operations, including
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : str or callable
            Reduce function on incoming edges. ``func(nodes) -> dict`` ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`.
        """
        self._reduce_func = func

    def register_apply_node_func(self, func):
        """Register global node apply function.

        Once registered, :attr:`func` will be used as the default apply
        node function. Related operations include :func:`apply_nodes`,
        :func:`recv`, :func:`send_and_recv`, :func:`push`, :func:`pull`,
        :func:`update_all`.

        Parameters
        ----------
        func : callable
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`.

        See Also
        --------
        register_apply_edge_func
        """
        self._apply_node_func = func

    def register_apply_edge_func(self, func):
        """Register global edge apply function.

        Once registered, :attr:`func` will be used as the default apply
        edge function in :func:`apply_edges`.

        Parameters
        ----------
        func : callable
            Apply function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`.

        See Also
        --------
        register_apply_node_func
        """
        self._apply_edge_func = func

    def apply_nodes(self, func="default", v=ALL, inplace=False):
        """Apply the function on the node features.

        Parameters
        ----------
        func : callable or None, optional
            The user defined applied function on the node features. If
            None, nothing will happen. A callable function takes the form
            ``func(nodes) -> dict``. ``dict`` has ``str`` keys and ``tensor``
            values. nodes are :class:`NodeBatch` objects as in :mod:`~dgl.udf`.
        v : int, iterable of int, tensor, optional
            The node (ids) on which to apply :attr:`func`. The default
            value is all the nodes.

        Examples
        --------

        >>> import dgl
        >>> import torch as th

        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.ones(3, 1)

        >>> # Increment the node feature by 1.
        >>> def increment_feature(nodes): return {'x': nodes.data['x'] + 1}
        >>> g.apply_nodes(func=increment_feature, v=[0, 2]) # Apply func to nodes 0, 2
        >>> g.ndata
        {'x': tensor([[2.],
                      [1.],
                      [2.]])}

        See Also
        --------
        apply_edges
        """
        if func == "default":
            func = self._apply_node_func
        if is_all(v):
            v = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(v)
        with ir.prog() as prog:
            scheduler.schedule_apply_nodes(graph=self, v=v, apply_func=func)
            Runtime.run(prog)

    def apply_edges(self, func="default", edges=ALL):
        """Apply the function on the edge features.

        Parameters
        ----------
        func : callable, optional
            The user defined applied function on the edge features. A
            callable function takes the form ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`.
        edges : tuple of 2 tensors, tuple of 2 iterable of int, int, iterable of int, or tensor, optional
            Edges on which to apply :attr:`func`. The default value is all the
            edges. :attr:`edges` can be pair(s) of endpoint nodes :math:`(u, v)`
            represented as a ``tuple of 2 tensors`` or a
            ``tuple of 2 iterable of int``. It can also be specified with edge ids,
            using a ``tensor`` of edge ids, an ``int``, an ``iterable of int``.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then all the edges
        between :math:`u` and :math:`v` will be updated.

        Examples
        --------

        >>> import dgl
        >>> import torch as th

        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.add_edges([0, 1], [1, 2])   # 0 -> 1, 1 -> 2
        >>> g.edata['y'] = th.ones(2, 1)

        >>> # Doubles the edge feature.
        >>> def double_feature(edges): return {'y': edges.data['y'] * 2}
        >>> g.apply_edges(func=double_feature, edges=0) # Apply func to the first edge.
        >>> g.edata
        {'y': tensor([[2.],   # 2 * 1
                      [1.]])}

        See Also
        --------
        apply_nodes
        """
        if func == "default":
            func = self._apply_edge_func
        assert func is not None

        if is_all(edges):
            u, v, eid = self._graph.edges()
        elif isinstance(edges, tuple):
            u, v = edges
            u = utils.toindex(u)
            v = utils.toindex(v)
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(u, v)
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)

        with ir.prog() as prog:
            scheduler.schedule_apply_edges(graph=self, u=u, v=v,
                    eid=eid, apply_func=func)
            Runtime.run(prog)

    def send(self, edges, message_func="default"):
        """Send messages along the given edges.

        Parameters
        ----------
        edges : tuple of 2 tensors, tuple of 2 iterable of int, int, iterable of int, or tensor
            Edges on which to apply :attr:`message_func`. :attr:`edges` can be pair(s) of
            endpoint nodes :math:`(u, v)` represented as a ``tuple of 2 tensors``
            or a ``tuple of 2 iterable of int``. It can also be specified with
            edge ids, using a ``tensor`` of edge ids, an ``int``, an ``iterable of int``.
        message_func : callable
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then the messages will be sent
        along all edges between :math:`u` and :math:`v`.

        Examples
        --------

        See the *message passing* example in :class:`DGLGraph` or :func:`recv`.
        """
        if message_func == "default":
            message_func = self._message_func

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

        with ir.prog() as prog:
            scheduler.schedule_send(graph=self, u=u, v=v, eid=eid,
                                    message_func=message_func)
            Runtime.run(prog)

        # update message graph and frame
        self._msg_graph.add_edges(u, v)

    def recv(self,
             v=ALL,
             reduce_func="default",
             apply_node_func="default"):
        """Receive and reduce in-coming messages and update representation on node(s) :math:`v`.

        * When all the nodes have zero-in-degrees, nothing will happen.
        * :attr:`reduce_func` returns a dictionary with ``str`` keys and ``tensor`` values, specifying
          different node features. If a type of node feature, specified by the key, already exists,
          the old features will be replaced, otherwise a new type of node feature is added.
        * When some but not all nodes have zero-in-degrees, a placeholder for that attribute will be
          used, by default a zero tensor. See :func:`set_n_initializer` about how to configure
          the placeholder setting for node features. If you do not want to re-initialize those nodes,
          you should not include them in :attr:`v`.

        Once the messages are received (and used to update related features), one needs to perform
        :func:`send` again before another :func:`recv` trial. Otherwise, nothing will happen.

        TODO(minjie): document on zero-in-degree case
        TODO(minjie): document on how returned new features are merged with the old features
        TODO(minjie): document on how many times UDFs will be called

        Parameters
        ----------
        v : node, container or tensor
            The node to be updated.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``.
            ``dict`` has ``str`` keys and ``tensor`` values. nodes are
            :class:`NodeBatch` objects as in :mod:`~dgl.udf`.
        apply_node_func : callable
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`.

        Examples
        --------
        Create a graph object for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
        >>> g.add_edges([0, 1], [1, 2])

        >>> # Define the function for sending node features as messages.
        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> # Set the function defined to be the default message function.
        >>> g.register_message_func(send_source)

        >>> # Sum the messages received and use this to replace the original node feature.
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> # Set the function defined to be the default message reduce function.
        >>> g.register_reduce_func(simple_reduce)

        Send and receive messages. Note that although node :math:`0` has no in-coming edges,
        its feature gets changed from :math:`1` to :math:`0` as it is also included in
        ``g.nodes()``.

        >>> g.send(g.edges())
        >>> g.recv(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])

        Once messages are received, one will need another call of :func:`send` again before
        another call of :func:`recv`. Otherwise, nothing will happen.

        >>> g.recv(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])
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

        with ir.prog() as prog:
            scheduler.schedule_recv(graph=self, recv_nodes=v,
                    reduce_func=reduce_func, apply_func=apply_node_func)
            Runtime.run(prog)

        # FIXME(minjie): multi send bug
        self.reset_messages()

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default"):
        """Send messages along edges and let destinations receive them.
        Optionally, apply a function to update the node features.

        This is a convenient combination of performing in order
        ``send(self, self.edges, message_func)`` and
        ``recv(self, dst, reduce_func, apply_node_func)``, where :attr:`dst`
        are the destinations of the :attr:`edges`.

        Parameters
        ----------
        edges : tuple of 2 tensors, tuple of 2 iterable of int, int, iterable of int, or tensor
            Edges on which to apply :attr:`message_func`. :attr:`edges` can be pair(s) of
            endpoint nodes :math:`(u, v)` represented as a ``tuple of 2 tensors``
            or a ``tuple of 2 iterable of int``. It can also be specified with
            edge ids, using a ``tensor`` of edge ids, an ``int``, an ``iterable of int``.
        message_func : callable, optional
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        Examples
        --------
        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])
        >>> g.add_edges([0, 1], [1, 2])

        >>> # Define the function for sending node features as messages.
        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> # Set the function defined to be the default message function.
        >>> g.register_message_func(send_source)

        >>> # Sum the messages received and use this to replace the original node feature.
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> # Set the function defined to be the default message reduce function.
        >>> g.register_reduce_func(simple_reduce)

        Send and receive messages.

        >>> g.send_and_recv(g.edges())
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [2.]])

        Note that the feature of node :math:`0` remains the same as it has no
        in-coming edges.

        Notes
        -----
        On multigraphs, if u and v are specified, then the messages will be sent
        and received along all edges between u and v.

        See Also
        --------
        send
        recv
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

        with ir.prog() as prog:
            scheduler.schedule_snr(self, (u, v, eid),
                    message_func, reduce_func, apply_node_func)
            Runtime.run(prog)

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default"):
        """Pull messages from the node's predecessors and then update its feature.

        * If all nodes have no in-coming edges, nothing will happen.
        * If some nodes have in-coming edges and some do not, the features for nodes
          with no in-coming edges will be re-initialized with a placeholder. By default
          the placeholder is a zero tensor. See :func:`set_n_initializer` about how to
          configure the placeholder setting for node features. Be careful about that
          and do not include these nodes in :attr:`v` if you do not want to re-initialize
          their features.

        Parameters
        ----------
        v : int, iterable of int, or tensor
            The node to be updated.
        message_func : callable, optional
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        Examples
        --------
        Create a graph for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[0.], [1.], [2.]])

        Use the built-in message function :func:`~dgl.function.copy_src` for copying
        node features as the message.

        >>> m_func = dgl.function.copy_src('x', 'm')
        >>> g.register_message_func(m_func)

        Use the built-int message reducing function :func:`~dgl.function.sum`, which
        sums the messages received and replace the old node features with it.

        >>> m_reduce_func = dgl.function.sum('m', 'x')
        >>> g.register_reduce_func(m_reduce_func)

        As no edges exist, nothing happens.

        >>> g.pull(g.nodes())
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [2.]])

        Add edges ``0 -> 1, 1 -> 2``. Pull messages for the node :math:`2`.

        >>> g.add_edges([0, 1], [1, 2])
        >>> g.pull(2)
        >>> g.ndata['x']
        tensor([[0.],
                [1.],
                [1.]])

        The feature of node :math:`2` changes but the feature of node :math:`1`
        remains the same as we did not :func:`pull` (and reduce) messages for it.

        See Also
        --------
        push
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
        with ir.prog() as prog:
            scheduler.schedule_pull(graph=self, pull_nodes=v,
                    message_func=message_func, reduce_func=reduce_func,
                    apply_func=apply_node_func)
            Runtime.run(prog)

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default"):
        """Send message from the node to its successors and update them.

        Parameters
        ----------
        u : int, iterable of int, or tensor
            The node to be updated.
        message_func : callable, optional
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        Examples
        --------
        Create a graph for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.]])

        Use the built-in message function :func:`~dgl.function.copy_src` for copying
        node features as the message.

        >>> m_func = dgl.function.copy_src('x', 'm')
        >>> g.register_message_func(m_func)

        Use the built-int message reducing function :func:`~dgl.function.sum`, which
        sums the messages received and replace the old node features with it.

        >>> m_reduce_func = dgl.function.sum('m', 'x')
        >>> g.register_reduce_func(m_reduce_func)

        As no edges exist, nothing happens.

        >>> g.push(g.nodes())
        >>> g.ndata['x']
        tensor([[1.],
                [2.],
                [3.]])

        Add edges ``0 -> 1, 1 -> 2``. Send messages from the node :math:`1`. and update.

        >>> g.add_edges([0, 1], [1, 2])
        >>> g.push(1)
        >>> g.ndata['x']
        tensor([[1.],
                [2.],
                [2.]])

        The feature of node :math:`2` changes but the feature of node :math:`1`
        remains the same as we did not :func:`push` for node :math:`0`.

        See Also
        --------
        pull
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
        with ir.prog() as prog:
            scheduler.schedule_push(graph=self, u=u,
                    message_func=message_func, reduce_func=reduce_func,
                    apply_func=apply_node_func)
            Runtime.run(prog)

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Send messages through all edges and update all nodes.

        This is a convenient combination of performing in order
        ``send(self, self.edges(), message_func)`` and
        ``recv(self, self.nodes(), reduce_func, apply_node_func)``.

        Parameters
        ----------
        message_func : callable, optional
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        See Also
        --------
        send
        recv
        """
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func
        assert message_func is not None
        assert reduce_func is not None

        with ir.prog() as prog:
            scheduler.schedule_update_all(graph=self, message_func=message_func,
                    reduce_func=reduce_func, apply_func=apply_node_func)
            Runtime.run(prog)

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
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        Examples
        --------
        Create a graph for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(4)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.], [4.]])
        >>> g.add_edges([0, 1, 1, 2], [1, 2, 3, 3])

        Prepare message function and message reduce function for demo.

        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> g.register_message_func(send_source)
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> g.register_reduce_func(simple_reduce)

        First pull messages for nodes :math:`1, 2` with edges ``0 -> 1`` and
        ``1 -> 2``; and then pull messages for node :math:`3` with edges
        ``1 -> 3`` and ``2 -> 3``.

        >>> g.prop_nodes([[1, 2], [3]])
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [2.],
                [3.]])

        In the first stage, we pull messages for nodes :math:`1, 2`.
        The feature of node :math:`1` is replaced by that of node :math:`0`, i.e. 1
        The feature of node :math:`2` is replaced by that of node :math:`1`, i.e. 2.
        Both of the replacement happen simultaneously.

        In the second stage, we pull messages for node :math:`3`.
        The feature of node :math:`3` becomes the sum of node :math:`1`'s feature and
        :math:`2`'s feature, i.e. 1 + 2 = 3.

        See Also
        --------
        prop_edges
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier,
                    message_func, reduce_func, apply_node_func)

    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Propagate messages using graph traversal by triggering
        :func:`send_and_recv()` on edges.

        The traversal order is specified by the ``edges_generator``. It generates
        edge frontiers, which is an iterable of edge ids or edge end points.

        edge ids can be represented as an int, or an iterable or tensor of ints. Edge
        end points can be represented by a tuple consisting of two tensors/iterables
        of ints. The first tensor/iterable specifies the source nodes and the second
        tensor/iterable specifies the dest nodes.

        Edges in the same frontier will be triggered together, while edges in
        different frontiers will be triggered according to the generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : callable, optional
            Message function on the edges. ``func(edges) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. edges are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used if not
            specified.
        reduce_func : callable, optional
            Reduce function on incoming edges. ``func(nodes) -> dict``. ``dict``
            has ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be used
            if not specified.
        apply_node_func : callable, optional
            Apply function on the nodes. ``func(nodes) -> dict``. ``dict`` has
            ``str`` keys and ``tensor`` values. nodes are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. Registered function will be
            used if not specified.

        Examples
        --------

        Create a graph for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(4)
        >>> g.ndata['x'] = th.tensor([[1.], [2.], [3.], [4.]])
        >>> g.add_edges([0, 1, 1, 2], [1, 2, 3, 3])

        Prepare message function and message reduce function for demo.

        >>> def send_source(edges): return {'m': edges.src['x']}
        >>> g.register_message_func(send_source)
        >>> def simple_reduce(nodes): return {'x': nodes.mailbox['m'].sum(1)}
        >>> g.register_reduce_func(simple_reduce)

        First propagate messages for edges ``0 -> 1``, ``1 -> 3`` and then
        propagate messages for edges ``1 -> 2``, ``2 -> 3``.

        >>> g.prop_edges([([0, 1], [1, 3]), ([1, 2], [2, 3])])
        >>> g.ndata['x']
        tensor([[1.],
                [1.],
                [1.],
                [3.]])

        In the first stage, the following happens simultaneously.

            - The feature of node :math:`1` is replaced by that of
              node :math:`0`, i.e. 1.
            - The feature of node :math:`3` is replaced by that of
              node :math:`1`, i.e. 2.

        In the second stage, the following happens simultaneously.

            - The feature of node :math:`2` is replaced by that of
              node :math:`1`, i.e. 1.
            - The feature of node :math:`3` is replaced by that of
              node :math:`2`, i.e. 3.

        See Also
        --------
        prop_nodes
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

        There are three types of an incidence matrix :math:`I`:

        * ``in``:
            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.
        * ``out``:
            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.
        * ``both``:
            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        type : str
            Can be either ``in``, ``out`` or ``both``
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
            ``func(nodes) -> tensor``. ``nodes`` are :class:`NodeBatch`
            objects as in :mod:`~dgl.udf`. The ``tensor`` returned should
            be a 1-D boolean tensor with :func:`~dgl.udf.NodeBatch.batch_size`
            elements indicating which nodes satisfy the predicate.
        nodes : int, iterable or tensor of ints
            The nodes to filter on.

        Returns
        -------
        tensor
            The filtered nodes.

        Examples
        --------
        Construct a graph object for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [-1.], [1.]])

        Define a function for filtering nodes with feature :math:`1`.

        >>> def has_feature_one(nodes): return (nodes.data['x'] == 1).squeeze(1)

        Filter the nodes with feature :math:`1`.

        >>> g.filter_nodes(has_feature_one)
        tensor([0, 2])

        See Also
        --------
        filter_edges
        """
        if is_all(nodes):
            v = utils.toindex(slice(0, self.number_of_nodes()))
        else:
            v = utils.toindex(nodes)

        n_repr = self.get_n_repr(v)
        nb = NodeBatch(self, v, n_repr)
        n_mask = predicate(nb)

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
            ``func(edges) -> tensor``. ``edges`` are :class:`EdgeBatch`
            objects as in :mod:`~dgl.udf`. The ``tensor`` returned should
            be a 1-D boolean tensor with :func:`~dgl.udf.EdgeBatch.batch_size`
            elements indicating which edges satisfy the predicate.
        edges : edges
            Edges can be pair(s) of endpoint nodes (u, v), or a
            tensor/iterable of edge ids. If the edges are pair(s) of endpoint
            nodes, they will be represented as a tuple of two iterable/tensors
            of nodes. The first one is for the source nodes while the second one
            is for the destinations. The default value is all the edges.

        Returns
        -------
        tensor
            The filtered edges represented by their ids.

        Examples
        --------
        Construct a graph object for demo.

        >>> import dgl
        >>> import torch as th
        >>> g = dgl.DGLGraph()
        >>> g.add_nodes(3)
        >>> g.ndata['x'] = th.tensor([[1.], [-1.], [1.]])
        >>> g.add_edges([0, 1, 2], [2, 2, 1])

        Define a function for filtering edges whose destinations have
        node feature :math:`1`.

        >>> def has_dst_one(edges): return (edges.dst['x'] == 1).squeeze(1)

        Filter the edges whose destination nodes have feature :math:`1`.

        >>> g.filter_edges(has_dst_one)
        tensor([0, 1])

        See Also
        --------
        filter_nodes
        """
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

        e_mask = predicate(eb)

        if is_all(edges):
            return F.nonzero_1d(e_mask)
        else:
            edges = F.tensor(edges)
            return edges[e_mask]
