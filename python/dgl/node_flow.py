"""Class for NodeFlow data structure."""
from __future__ import absolute_import

import numpy as np

from collections import namedtuple
from .base import ALL, is_all, DGLError
from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLGraph
from .runtime import ir, scheduler, Runtime
from collections.abc import MutableMapping
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
        return NodeSpace(data=LayerDataView(self._graph, layer))

    def __call__(self):
        """Return the nodes."""
        return F.arange(0, len(self))

class LayerDataView(MutableMapping):
    """The data view class when G.nodes[...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
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

EdgeSpace = namedtuple('EdgeSpace', ['data'])

class FlowView(object):
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
        return self._graph.num_flows

    def __getitem__(self, flow):
        if not isinstance(flow, int):
            raise DGLError('Currently we only support the view of one flow')
        return EdgeSpace(data=FlowDataView(self._graph, flow))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class FlowDataView(MutableMapping):
    """The data view class when G.edges[...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
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

def _copy_to_like(arr1, arr2):
    return F.copy_to(arr1, F.context(arr2))

def _get_frame(frame, names, ids):
    kv = {name: frame[name][_copy_to_like(ids, frame[name])] for name in names}
    if len(kv) == 0:
        return FrameRef(Frame(num_rows=len(ids)))
    else:
        return FrameRef(Frame(kv))


def _update_frame(frame, names, ids, new_frame):
    kv = {name: new_frame[name] for name in names}
    if len(kv) > 0:
        frame.update_rows(ids, FrameRef(Frame(kv)), inplace=True)


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
        self._node_mapping = graph_idx.node_mapping
        self._edge_mapping = graph_idx.edge_mapping
        self._layer_offsets = graph_idx.layers.tonumpy()
        self._flow_offsets = graph_idx.flows.tonumpy()
        self._node_frames = [FrameRef(Frame(num_rows=self.layer_size(i))) for i in range(self.num_layers)]
        self._edge_frames = [FrameRef(Frame(num_rows=self.flow_size(i))) for i in range(self.num_flows)]

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

    def _get_layer_id(self, layer_id):
        if layer_id >= 0:
            return layer_id
        else:
            return self.num_layers + layer_id

    def _get_flow_id(self, flow_id):
        if flow_id >= 0:
            return flow_id
        else:
            return self.num_flows + flow_id

    def _get_node_frame(self, layer_id):
        return self._node_frames[layer_id]

    def _get_edge_frame(self, flow_id):
        return self._edge_frames[flow_id]

    @property
    def num_layers(self):
        """Get the number of layers.

        Returns
        -------
        int
            the number of layers
        """
        return len(self._layer_offsets) - 1

    @property
    def num_flows(self):
        return self.num_layers - 1

    @property
    def layers(self):
        return LayerView(self)

    @property
    def flows(self):
        return FlowView(self)

    def layer_size(self, layer_id):
        """Return the number of nodes in a specified layer."""
        layer_id = self._get_layer_id(layer_id)
        return self._layer_offsets[layer_id + 1] - self._layer_offsets[layer_id]

    def flow_size(self, flow_id):
        flow_id = self._get_flow_id(flow_id)
        return self._flow_offsets[flow_id + 1] - self._flow_offsets[flow_id]

    def copy_from_parent(self, node_embed_names=ALL, edge_embed_names=ALL):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        if self._parent._node_frame.num_rows != 0 and self._parent._node_frame.num_columns != 0:
            if is_all(node_embed_names):
                for i in range(self.num_layers):
                    nid = utils.toindex(self.layer_parent_nid(i))
                    self._node_frames[i] = FrameRef(Frame(self._parent._node_frame[nid]))
            elif node_embed_names is not None:
                assert isinstance(node_embed_names, list) \
                        and len(node_embed_names) == self.num_layers, \
                        "The specified embedding names should be the same as the number of layers."
                for i in range(self.num_layers):
                    nid = self.layer_parent_nid(i)
                    self._node_frames[i] = _get_frame(self._parent._node_frame,
                                                      node_embed_names[i], nid)

        if self._parent._edge_frame.num_rows != 0 and self._parent._edge_frame.num_columns != 0:
            if is_all(edge_embed_names):
                for i in range(self.num_flows):
                    eid = utils.toindex(self.flow_parent_eid(i))
                    self._edge_frames[i] = FrameRef(Frame(self._parent._edge_frame[eid]))
            elif edge_embed_names is not None:
                assert isinstance(edge_embed_names, list) \
                        and len(edge_embed_names) == self.num_flows, \
                        "The specified embedding names should be the same as the number of flows."
                for i in range(self.num_flows):
                    eid = self.flow_parent_eid(i)
                    self._edge_frames[i] = _get_frame(self._parent._edge_frame,
                                                      edge_embed_names[i], eid)

    def copy_to_parent(self, node_embed_names=ALL, edge_embed_names=ALL):
        """Copy node/edge embeddings to the parent graph.
        """
        #TODO We need to take care of the following things:
        #    * copy right node embeddings back. For instance, we should copy temporary
        #      node embeddings back; we don't need to copy read-only node embeddings back.
        #    * When nodes in different layers have the same node embedding, we need
        #      to avoid conflicts.
        if self._parent._node_frame.num_rows != 0 and self._parent._node_frame.num_columns != 0:
            if is_all(node_embed_names):
                for i in range(self.num_layers):
                    nid = utils.toindex(self.layer_parent_nid(i))
                    # We should write data back directly.
                    self._parent._node_frame.update_rows(nid, self._node_frames[i], inplace=True)
            elif node_embed_names is not None:
                assert isinstance(node_embed_names, list) \
                        and len(node_embed_names) == self.num_layers, \
                        "The specified embedding names should be the same as the number of layers."
                for i in range(self.num_layers):
                    nid = utils.toindex(self.layer_parent_nid(i))
                    _update_frame(self._parent._node_frame, node_embed_names[i], nid,
                                  self._node_frames[i])

        if self._parent._edge_frame.num_rows != 0 and self._parent._edge_frame.num_columns != 0:
            if is_all(edge_embed_names):
                for i in range(self.num_flows):
                    eid = utils.toindex(self.flow_parent_eid(i))
                    self._parent._edge_frame.update_rows(eid, self._edge_frames[i], inplace=True)
            elif edge_embed_names is not None:
                assert isinstance(edge_embed_names, list) \
                        and len(edge_embed_names) == self.num_flows, \
                        "The specified embedding names should be the same as the number of flows."
                for i in range(self.num_flows):
                    eid = utils.toindex(self.flow_parent_eid(i))
                    _update_frame(self._parent._edge_frame, edge_embed_names[i], eid,
                                  self._edge_frames[i])

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

    def map_to_layer_nid(self, nid):
        layer_id = np.sum(self._layer_offsets <= nid) - 1
        # TODO do I need to reverse here?
        return int(layer_id), nid - self._layer_offsets[layer_id]

    def map_to_flow_eid(self, eid):
        flow_id = np.sum(self._flow_offsets <= eid) - 1
        # TODO do I need to reverse here?
        return int(flow_id), eid - self._flow_offsets[flow_id]

    def layer_in_degree(self, layer_id):
        return self._graph.in_degrees(utils.toindex(self.layer_nid(layer_id))).tousertensor()

    def layer_out_degree(self, layer_id):
        return self._graph.out_degrees(utils.toindex(self.layer_nid(layer_id))).tousertensor()

    def layer_nid(self, layer_id):
        """Get the node Ids in the specified layer.

        Returns
        -------
        Tensor
            The node id array.
        """
        layer_id = self._get_layer_id(layer_id)
        assert layer_id + 1 < len(self._layer_offsets)
        start = self._layer_offsets[layer_id]
        end = self._layer_offsets[layer_id + 1]
        return F.arange(start, end)

    def layer_parent_nid(self, layer_id):
        """Get the node Ids of the parent graph in the specified layer

        Returns
        -------
        Tensor
            The parent node id array.
        """
        layer_id = self._get_layer_id(layer_id)
        assert layer_id + 1 < len(self._layer_offsets)
        start = self._layer_offsets[layer_id]
        end = self._layer_offsets[layer_id + 1]
        return self._node_mapping.tousertensor()[start:end]

    def flow_eid(self, flow_id):
        flow_id = self._get_flow_id(flow_id)
        start = self._flow_offsets[flow_id]
        end = self._flow_offsets[flow_id + 1]
        return F.arange(start, end)

    def flow_parent_eid(self, flow_id):
        flow_id = self._get_flow_id(flow_id)
        start = self._flow_offsets[flow_id]
        end = self._flow_offsets[flow_id + 1]
        return self._edge_mapping.tousertensor()[start:end]

    def apply_layer(self, layer_id, func="default", v=ALL, inplace=False):
        if func == "default":
            func = self._apply_node_func
        if is_all(v):
            v = utils.toindex(slice(0, self.layer_size(layer_id)))
        else:
            v = v - self._layer_offsets[layer_id]
            #assert F.all(v >= 0), "All vertices have to be in layer " + str(layer_id)
            v = utils.toindex(v)
        with ir.prog() as prog:
            scheduler.schedule_nodeflow_apply_nodes(graph=self,
                                                    layer_id=layer_id,
                                                    v=v,
                                                    apply_func=func,
                                                    inplace=inplace)
            Runtime.run(prog)

    def _layer_local_nid(self, layer_id):
        return F.arange(0, self.layer_size(layer_id))

    def apply_flow(self, flow_id, func="default", edges=ALL, inplace=False):
        if func == "default":
            func = self._apply_edge_func
        assert func is not None

        if is_all(edges):
            u = utils.toindex(self._layer_local_nid(flow_id))
            v = utils.toindex(self._layer_local_nid(flow_id + 1))
            eid = utils.toindex(slice(0, self.flow_size(flow_id)))
        elif isinstance(edges, tuple):
            u, v = edges
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(utils.toindex(u), utils.toindex(v))
            u = utils.toindex(u.tousertensor() - self._layer_offsets[flow_id])
            v = utils.toindex(v.tousertensor() - self._layer_offsets[flow_id + 1])
            eid = utils.toindex(eid.tousertensor() - self._flow_offsets[flow_id])
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)
            u = utils.toindex(u.tousertensor() - self._layer_offsets[flow_id])
            v = utils.toindex(v.tousertensor() - self._layer_offsets[flow_id + 1])
            eid = utils.toindex(edges - self._flow_offsets[flow_id])

        with ir.prog() as prog:
            scheduler.schedule_nodeflow_apply_edges(graph=self,
                                                    flow_id=flow_id,
                                                    u=u,
                                                    v=v,
                                                    eid=eid,
                                                    apply_func=func,
                                                    inplace=inplace)
            Runtime.run(prog)

    def _conv_local_nid(self, nid, layer_id):
        layer_id = self._get_layer_id(layer_id)
        return nid - self._layer_offsets[layer_id]

    def _conv_local_eid(self, eid, flow_id):
        flow_id = self._get_flow_id(flow_id)
        return eid - self._flow_offsets[flow_id]

    def flow_compute(self, flow_id, message_func="default", reduce_func="default",
                     apply_node_func="default", v=ALL, inplace=False):
        if message_func == "default":
            message_func = self._message_func
        if reduce_func == "default":
            reduce_func = self._reduce_func
        if apply_node_func == "default":
            apply_node_func = self._apply_node_func

        assert message_func is not None
        assert reduce_func is not None

        if is_all(v):
            dest_nodes = utils.toindex(self.layer_nid(flow_id + 1))
            u, v, _ = self._graph.in_edges(dest_nodes)
            u = utils.toindex(self._conv_local_nid(u.tousertensor(), flow_id))
            v = utils.toindex(self._conv_local_nid(v.tousertensor(), flow_id + 1))
            dest_nodes = utils.toindex(F.arange(0, self.layer_size(flow_id + 1)))
            eid = utils.toindex(F.arange(0, self.flow_size(flow_id)))
        else:
            dest_nodes = utils.toindex(v)
            u, v, eid = self._graph.in_edges(dest_nodes)
            assert len(u) > 0, "flow_compute must run on edges"
            u = utils.toindex(self._conv_local_nid(u.tousertensor(), flow_id))
            v = utils.toindex(self._conv_local_nid(v.tousertensor(), flow_id + 1))
            dest_nodes = utils.toindex(self._conv_local_nid(dest_nodes.tousertensor(),
                                                            flow_id + 1))
            eid = utils.toindex(self._conv_local_eid(eid, flow_id))

        with ir.prog() as prog:
            scheduler.schedule_nodeflow_compute(graph=self,
                                                flow_id=flow_id,
                                                u=u,
                                                v=v,
                                                eid=eid,
                                                dest_nodes = dest_nodes,
                                                message_func=message_func,
                                                reduce_func=reduce_func,
                                                apply_func=apply_node_func,
                                                inplace=inplace)
            Runtime.run(prog)

    def prop_flows(self, message_func="default", reduce_func="default",
                   apply_node_func="default", flow_range=ALL, inplace=False):
        if isinstance(flow_range, int):
            self.flow_compute(flow_range, message_func, reduce_func, apply_node_func,
                              inplace=inplace)
        else:
            if is_all(flow_range):
                flow_range=range(0, self.num_flows)
            elif isinstance(flow_range, slice):
                if slice.step is not 1:
                    raise DGLError("We can't propogate flows and skip some of them")
                flow_range=range(flow_range.start, flow_range.stop)
            else:
                raise DGLError("unknown flow range")

            for i in flow_range:
                self.flow_compute(i, message_func, reduce_func, apply_node_func,
                                  inplace=inplace)


def create_full_node_flow(g, num_layers):
    seeds = [utils.toindex(F.arange(0, g.number_of_nodes()))]
    nfi = g._graph.neighbor_sampling(seeds, g.number_of_nodes(), num_layers, 'in', None)
    return NodeFlow(g, nfi[0])
