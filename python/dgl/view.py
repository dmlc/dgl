"""Views of DGLGraph."""
from __future__ import absolute_import

from collections import namedtuple
from collections.abc import MutableMapping

import numpy as np

from .base import ALL, is_all, DGLError
from . import backend as F

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
    """The data view class when G.nodes[...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_nodes']

    def __init__(self, graph, nodes):
        self._graph = graph
        self._nodes = nodes

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._nodes)[key]

    def __setitem__(self, key, val):
        if isinstance(val, np.ndarray):
            val = F.zerocopy_from_numpy(val)
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
    """The data view class when G.edges[...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_edges']

    def __init__(self, graph, edges):
        self._graph = graph
        self._edges = edges

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._edges)[key]

    def __setitem__(self, key, val):
        if isinstance(val, np.ndarray):
            val = F.zerocopy_from_numpy(val)
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

class LayerView(object):
    """A LayerView class to act as nflow.layers for a NodeFlow.

    Can be used to get a list of current nodes and get and set node data.
    """
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __len__(self):
        return self._graph.num_layers()

    def __getitem__(self, layer):
        if not isinstance(layer, int):
            raise DGLError('Currently we only support the view of one layer')
        return NodeSpace(data=LayerDataView(self._graph, layer))

    def __call__(self):
        """Return the nodes."""
        return F.arange(0, len(self))

class LayerDataView(MutableMapping):
    """The data view class when G.layers[...].data is called.
    """
    __slots__ = ['_graph', '_layer']

    def __init__(self, graph, layer):
        self._graph = graph
        self._layer = layer

    def __getitem__(self, key):
        return self._graph._node_frames[self._layer][key]

    def __setitem__(self, key, val):
        self._graph._node_frames[self._layer][key] = val

    def __delitem__(self, key):
        del self._graph._node_frames[self._layer][key]

    def __len__(self):
        return len(self._graph._node_frames[self._layer])

    def __iter__(self):
        return iter(self._graph._node_frames[self._layer])

    def __repr__(self):
        data = self._graph._node_frames[self._layer]
        return repr({key : data[key] for key in data})

class BlockView(object):
    """A BlockView class to act as nflow.blocks for a NodeFlow.

    Can be used to get a list of current edges and get and set edge data.
    """
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __len__(self):
        return self._graph.num_blocks

    def __getitem__(self, flow):
        if not isinstance(flow, int):
            raise DGLError('Currently we only support the view of one flow')
        return EdgeSpace(data=BlockDataView(self._graph, flow))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class BlockDataView(MutableMapping):
    """The data view class when G.blocks[...].data is called.
    """
    __slots__ = ['_graph', '_flow']

    def __init__(self, graph, flow):
        self._graph = graph
        self._flow = flow

    def __getitem__(self, key):
        return self._graph._edge_frames[self._flow][key]

    def __setitem__(self, key, val):
        self._graph._edge_frames[self._flow][key] = val

    def __delitem__(self, key):
        del self._graph._edge_frames[self._flow][key]

    def __len__(self):
        return len(self._graph._edge_frames[self._flow])

    def __iter__(self):
        return iter(self._graph._edge_frames[self._flow])

    def __repr__(self):
        data = self._graph._edge_frames[self._flow]
        return repr({key : data[key] for key in data})

class HeteroNodeView(object):
    """A NodeView class to act as G.nodes for a DGLHeteroGraph."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, ntype):
        return HeteroNodeTypeView(self._graph, ntype)

class HeteroNodeTypeView(object):
    """A NodeView class to act as G.nodes[ntype] for a DGLHeteroGraph.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_ntype']

    def __init__(self, graph, ntype):
        self._graph = graph
        self._ntype = ntype

    def __len__(self):
        return self._graph.number_of_nodes(self._graph._ntypes_invmap[self._ntype])

    def __getitem__(self, nodes):
        if isinstance(nodes, slice):
            # slice
            if not (nodes.start is None and nodes.stop is None
                    and nodes.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            return NodeSpace(data=HeteroNodeTypeDataView(self._graph, self._ntype, ALL))
        else:
            return NodeSpace(data=HeteroNodeTypeDataView(self._graph, self._ntype, nodes))

    def __call__(self):
        """Return the nodes."""
        return F.arange(0, len(self))

class HeteroNodeTypeDataView(MutableMapping):
    """The data view class when G.nodes[ntype][...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_ntype', '_nodes']

    def __init__(self, graph, ntype, nodes):
        self._graph = graph
        self._ntype = ntype
        self._nodes = nodes

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._ntype, self._nodes)[key]

    def __setitem__(self, key, val):
        self._graph.set_n_repr(self._ntype, {key : val}, self._nodes)

    def __delitem__(self, key):
        raise DGLError('Delete feature data is not supported on only a subset'
                       ' of nodes. Please use `del G.ndata[key]` instead.')

    def __len__(self):
        return len(self._graph._node_frames[self._graph._ntypes_invmap[self._ntype]])

    def __iter__(self):
        return iter(self._graph.get_n_repr(self._ntype, self._nodes))

    def __repr__(self):
        data = self._graph.get_n_repr(self._ntype, self._nodes)
        return repr({key : data[key]
                     for key in self._graph._node_frames[self._graph._ntypes_invmap[self._ntype]]})

class HeteroNodeDataView(object):
    """The data view class when G.ndata is called."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        return HeteroNodeDataTypeView(self._graph, key)

class HeteroNodeDataTypeView(MutableMapping):
    """The data view class when G.ndata[ntype] is called."""
    __slots__ = ['_graph', '_ntype']

    def __init__(self, graph, ntype):
        self._graph = graph
        self._ntype = ntype

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._ntype)[key]

    def __setitem__(self, key, val):
        self._graph.set_n_repr(self._ntype, {key : val})

    def __delitem__(self, key):
        self._graph.pop_n_repr(self._ntype, key)

    def __len__(self):
        return len(self._graph._node_frames[self._graph._ntypes_invmap[self._ntype]])

    def __iter__(self):
        return iter(self._graph._node_frames[self._graph._ntypes_invmap[self._ntype]])

    def __repr__(self):
        data = self._graph.get_n_repr(self._ntype)
        return repr({key : data[key]
                     for key in self._graph._node_frames[self._graph._ntypes_invmap[self._ntype]]})

class HeteroEdgeView(object):
    """A EdgeView class to act as G.edges for a DGLHeteroGraph."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, etype):
        return HeteroEdgeTypeView(self._graph, etype)

class HeteroEdgeTypeView(object):
    """A EdgeView class to act as G.edges[etype] for a DGLHeteroGraph.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_etype']

    def __init__(self, graph, etype):
        self._graph = graph
        self._etype = etype

    def __len__(self):
        return self._graph.number_of_edges(self._graph._etypes_invmap[self._etype])

    def __getitem__(self, edges):
        if isinstance(edges, slice):
            # slice
            if not (edges.start is None and edges.stop is None
                    and edges.step is None):
                raise DGLError('Currently only full slice ":" is supported')
            return EdgeSpace(data=HeteroEdgeTypeDataView(self._graph, self._etype, ALL))
        else:
            return EdgeSpace(data=HeteroEdgeTypeDataView(self._graph, self._etype, edges))

    def __call__(self):
        """Return the edges."""
        return F.arange(0, len(self))

class HeteroEdgeTypeDataView(MutableMapping):
    """The data view class when G.edges[etype][...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_etype', '_edges']

    def __init__(self, graph, etype, edges):
        self._graph = graph
        self._etype = etype
        self._edges = edges

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._etype, self._edges)[key]

    def __setitem__(self, key, val):
        self._graph.set_e_repr(self._etype, {key : val}, self._edges)

    def __delitem__(self, key):
        raise DGLError('Delete feature data is not supported on only a subset'
                       ' of edges. Please use `del G.edata[key]` instead.')

    def __len__(self):
        return len(self._graph._edge_frames[self._graph._etypes_invmap[self._etype]])

    def __iter__(self):
        return iter(self._graph.get_e_repr(self._etype, self._edges))

    def __repr__(self):
        data = self._graph.get_e_repr(self._etype, self._edges)
        return repr({key : data[key]
                     for key in self._graph._edge_frames[self._graph._etypes_invmap[self._etype]]})

class HeteroEdgeDataView(object):
    """The data view class when G.edata is called."""
    __slots__ = ['_graph']

    def __init__(self, graph):
        self._graph = graph

    def __getitem__(self, key):
        return HeteroEdgeDataTypeView(self._graph, key)

class HeteroEdgeDataTypeView(MutableMapping):
    """The data view class when G.edata[etype] is called."""
    __slots__ = ['_graph', '_etype']

    def __init__(self, graph, etype):
        self._graph = graph
        self._etype = etype

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._etype)[key]

    def __setitem__(self, key, val):
        self._graph.set_e_repr(self._etype, {key : val})

    def __delitem__(self, key):
        self._graph.pop_e_repr(self._etype, key)

    def __len__(self):
        return len(self._graph._edge_frames[self._graph._etypes_invmap[self._etype]])

    def __iter__(self):
        return iter(self._graph._edge_frames[self._graph._etypes_invmap[self._etype]])

    def __repr__(self):
        data = self._graph.get_e_repr(self._etype)
        return repr({key : data[key]
                     for key in self._graph._edge_frames[self._graph._etypes_invmap[self._etype]]})
