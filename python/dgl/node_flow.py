"""Class for NodeFlow data structure."""
from __future__ import absolute_import

from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLGraph

NodeSpace = namedtuple('NodeSpace', ['data'])

class LayerView(object):
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
        return self._graph.num_layers()

    def __getitem__(self, layer):
        if not isinstance(layer, int):
            raise DGLError('Currently we only support the view of one layer')
        return NodeSpace(data=NodeDataView(self._graph, self._graph.layer_nid(layer)))

    def __call__(self):
        """Return the nodes."""
        return F.arange(0, len(self))

EdgeSpace = namedtuple('EdgeSpace', ['data'])

class LayerEdgeView(object):
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

    def __getitem__(self, layer):
        if not isinstance(layer, int):
            raise DGLError('Currently we only support the view of one layer')
        return EdgeSpace(data=EdgeDataView(self._graph, self._graph.layer_eid(layer)))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class NodeFlow(DGLGraph):
    """The NodeFlow class stores the sampling results of Neighbor sampling and Layer-wise sampling.

    These sampling algorithms generate graphs with multiple layers. The edges connect the nodes
    between two layers while there don't exist edges between the nodes in the same layer.

    We store multiple layers of the sampling results in a single graph. We store extra information,
    such as the node and edge mapping from the NodeFlow graph to the parent graph.

    Parameters
    ----------
    parent : DGLGraph
        The parent graph
    graph_index : NodeFlowIndex
        The graph index of the NodeFlow graph.
    """
    def __init__(self, parent, graph_idx):
        super(NodeFlow, self).__init__(graph_data=graph_idx,
                                       readonly=graph_idx.is_readonly())
        self._parent = parent
        self._index = graph_idx
        self._node_mapping = graph_idx.node_mapping
        self._edge_mapping = graph_idx.edge_mapping
        self._layers = graph_idx.layers

    # override APIs
    def add_nodes(self, num, data=None):
        """Add nodes. Disabled because BatchedDGLGraph is read-only."""
        raise DGLError('Readonly graph. Mutation is not allowed.')

    def add_edge(self, u, v, data=None):
        """Add one edge. Disabled because BatchedDGLGraph is read-only."""
        raise DGLError('Readonly graph. Mutation is not allowed.')

    def add_edges(self, u, v, data=None):
        """Add many edges. Disabled because BatchedDGLGraph is read-only."""
        raise DGLError('Readonly graph. Mutation is not allowed.')

    @property
    def num_layers(self):
        """Get the number of layers.

        Returns
        -------
        int
            the number of layers
        """
        return len(self._layers) - 1

    @property
    def layers(self):
        return LayerView(self)

    @property
    def flows(self):
        return LayerEdgeView(self)

    def layer_size(self, layer_id):
        """Return the number of nodes in a specified layer."""
        return self._layers[layer_id + 1] - self._layers[layer_id]

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        if self._parent._node_frame.num_rows != 0:
            self._node_frame = FrameRef(Frame(
                self._parent._node_frame[self._node_mapping]))
        if self._parent._edge_frame.num_rows != 0:
            self._edge_frame = FrameRef(Frame(
                self._parent._edge_frame[self._edge_mapping]))

    def map_to_parent_nid(self, nid):
        """This maps the child node Ids to the parent Ids.

        Parameters
        ----------
        nid : tensor
            The node ID array in the NodeFlow graph.

        Returns
        -------
        Tensor
            The parent node id array.
        """
        return self._node_mapping.tousertensor()[nid]

    def map_to_parent_eid(self, eid):
        """This maps the child edge Ids to the parent Ids.

        Parameters
        ----------
        nid : tensor
            The edge ID array in the NodeFlow graph.

        Returns
        -------
        Tensor
            The parent edge id array.
        """
        return self._edge_mapping.tousertensor()[eid]

    def layer_nid(self, layer_id):
        """Get the node Ids in the specified layer.

        Returns
        -------
        Tensor
            The node id array.
        """
        assert layer_id + 1 < len(self._layers)
        start = self._layers[layer_id]
        end = self._layers[layer_id + 1]
        return F.arange(start, end)

    def layer_parent_nid(self, layer_id):
        """Get the node Ids of the parent graph in the specified layer

        Returns
        -------
        Tensor
            The parent node id array.
        """
        assert layer_id + 1 < len(self._layers)
        start = self._layers[layer_id]
        end = self._layers[layer_id + 1]
        return self._node_mapping.tousertensor()[start:end]

    def layer_eid(self, layer_id):
        pass

    def layer_parent_eid(self, layer_id):
        pass

    def register_layer_computation(layerid, mfunc, rfunc, afunc):
        """Register UDFs for the give layer."""
        pass

    def compute(self):
        """Compute each layer one-by-one using the registered UDFs."""
        pass
