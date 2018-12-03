"""Views of DGLGraph."""
from __future__ import absolute_import

from collections import MutableMapping, namedtuple

from .base import ALL, is_all, DGLError
from . import backend as F
from . import utils

NodeSpace = namedtuple('NodeSpace', ['data'])

class NodeView(object):
    """A NodeView class to act as G.nodes for a DGLGraph.

    Can be used to get a list of current nodes and get and set node data.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __len__(self):
        return self._graph.number_of_nodes()

    def __getitem__(self, nodes):
        if isinstance(nodes, slice):
            # slice
            if not (nodes.start is None and nodes.stop is None
                    and nodes.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            return NodeSpace(data=NodeDataView(self._graph, ALL))
        else:
            return NodeSpace(data=NodeDataView(self._graph, nodes))

    def __call__(self):
        """Return the nodes."""
        return F.arange(0, len(self))

class NodeDataView(MutableMapping):
    __slots__ = ['_graph', '_nodes']

    def __init__(self, graph, nodes):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._nodes)[key]

    def __setitem__(self, key, val):
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

EdgeSpace = namedtuple('EdgeSpace', ['data'])

class EdgeView(object):
    """A EdgeView class to act as G.edges for a DGLGraph.

    Can be used to get a list of current edges and get and set edge data.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __len__(self):
        return self._graph.number_of_edges()

    def __getitem__(self, edges):
        if isinstance(edges, slice):
            # slice
            if not (edges.start is None and edges.stop is None
                    and edges.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            return EdgeSpace(data=EdgeDataView(self._graph, ALL))
        else:
            return EdgeSpace(data=EdgeDataView(self._graph, edges))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class EdgeDataView(MutableMapping):
    __slots__ = ['_graph', '_edges']

    def __init__(self, graph, edges):
        self._graph = graph
        self._edges = edges

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._edges)[key]

    def __setitem__(self, key, val):
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
