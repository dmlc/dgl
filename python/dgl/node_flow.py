"""Class for NodeFlow data structure."""
from __future__ import absolute_import

from collections import namedtuple
from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLGraph
from .view import NodeDataView
from .view import EdgeDataView
from . import utils

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
        return EdgeSpace(data=EdgeDataView(self._graph, self._graph.flow_eid(layer)))

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

    def _reverse_flow(self, flow_id):
        return self.num_layers - flow_id - 2

    def _reverse_layer(self, layer_id):
        return self.num_layers - layer_id - 1

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
        layer_id = self._reverse_layer(layer_id)
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
        layer_id = self._reverse_layer(layer_id)
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
        layer_id = self._reverse_layer(layer_id)
        assert layer_id + 1 < len(self._layers)
        start = self._layers[layer_id]
        end = self._layers[layer_id + 1]
        return self._node_mapping.tousertensor()[start:end]

    def flow_eid(self, flow_id):
        flow_id = self._reverse_flow(flow_id)
        start = self._layers[flow_id]
        end = self._layers[flow_id + 1]
        vids = F.arange(start, end)
        _, _, eids = self._index.in_edges(utils.toindex(vids))
        return eids

    def flow_parent_eid(self, flow_id):
        flow_id = self._reverse_flow(flow_id)
        start = self._layers[flow_id]
        end = self._layers[flow_id + 1]
        if start == 0:
            prev_num_edges = 0
        else:
            vids = utils.toindex(F.arange(0, start))
            prev_num_edges = F.asnumpy(F.sum(self._index.in_degrees(vids).tousertensor(), 0))
        vids = utils.toindex(F.arange(start, end))
        num_edges = F.asnumpy(F.sum(self._index.in_degrees(vids).tousertensor(), 0))
        return self._edge_mapping.tousertensor()[prev_num_edges:(prev_num_edges + num_edges)]

    def flow_compute(self, flow_id, msg_func, red_func, update_func):
        return self.pull(self.layer_nid(flow_id + 1), msg_func, red_func, update_func)

def create_full_node_flow(g, num_layers):
    seeds = [utils.toindex(F.arange(0, g.number_of_nodes()))]
    nfi = g._graph.neighbor_sampling(seeds, g.number_of_nodes(), num_layers, 'in', None)
    return NodeFlow(g, nfi[0])
