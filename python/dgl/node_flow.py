"""Class for NodeFlow data structure."""
from __future__ import absolute_import

from collections import namedtuple
from collections.abc import MutableMapping

from .base import ALL, is_all, DGLError
from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLBaseGraph
from .runtime import ir, scheduler, Runtime
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

class BlockView(object):
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
        return self._graph.num_blocks

    def __getitem__(self, flow):
        if not isinstance(flow, int):
            raise DGLError('Currently we only support the view of one flow')
        return EdgeSpace(data=BlockDataView(self._graph, flow))

    def __call__(self, *args, **kwargs):
        """Return all the edges."""
        return self._graph.all_edges(*args, **kwargs)

class BlockDataView(MutableMapping):
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
    col_dict = {name: frame[name][_copy_to_like(ids, frame[name])] for name in names}
    if len(col_dict) == 0:
        return FrameRef(Frame(num_rows=len(ids)))
    else:
        return FrameRef(Frame(col_dict))


def _update_frame(frame, names, ids, new_frame):
    col_dict = {name: new_frame[name] for name in names}
    if len(col_dict) > 0:
        frame.update_rows(ids, FrameRef(Frame(col_dict)), inplace=True)


class NodeFlow(DGLBaseGraph):
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
        super(NodeFlow, self).__init__(graph_idx)
        self._parent = parent
        self._node_mapping = graph_idx.node_mapping
        self._edge_mapping = graph_idx.edge_mapping
        self._layer_offsets = graph_idx.layers.tonumpy()
        self._block_offsets = graph_idx.flows.tonumpy()
        self._node_frames = [FrameRef(Frame(num_rows=self.layer_size(i))) \
                             for i in range(self.num_layers)]
        self._edge_frames = [FrameRef(Frame(num_rows=self.block_size(i))) \
                             for i in range(self.num_blocks)]
        # registered functions
        self._message_funcs = [None] * self.num_blocks
        self._reduce_funcs = [None] * self.num_blocks
        self._apply_node_funcs = [None] * self.num_blocks
        self._apply_edge_funcs = [None] * self.num_blocks

    def _get_layer_id(self, layer_id):
        if layer_id >= 0:
            return layer_id
        else:
            return self.num_layers + layer_id

    def _get_block_id(self, block_id):
        if block_id >= 0:
            return block_id
        else:
            return self.num_blocks + block_id

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
    def num_blocks(self):
        """Get the number of blocks.

        Returns
        -------
        int
            the number of blocks
        """
        return self.num_layers - 1

    @property
    def layers(self):
        """Return a LayerView of this NodeFlow.

        This is mainly for usage like:
        * `g.layers[2].data['h']` to get the node features of layer#2.
        * `g.layers(2)` to get the nodes of layer#2.
        """
        return LayerView(self)

    @property
    def blocks(self):
        """Return a BlockView of this NodeFlow.

        This is mainly for usage like:
        * `g.blocks[1,2].data['h']` to get the edge features of blocks from layer#1 to layer#2.
        * `g.blocks(1, 2)` to get the edge ids of blocks #1->#2.
        """
        return BlockView(self)

    def layer_size(self, layer_id):
        """Return the number of nodes in a specified layer.

        Parameters
        ----------
        layer_id : int
            the specified layer to return the number of nodes.
        """
        layer_id = self._get_layer_id(layer_id)
        return int(self._layer_offsets[layer_id + 1]) - int(self._layer_offsets[layer_id])

    def block_size(self, block_id):
        """Return the number of edges in a specified block.

        Parameters
        ----------
        block_id : int
            the specified block to return the number of edges.
        """
        block_id = self._get_block_id(block_id)
        return int(self._block_offsets[block_id + 1]) - int(self._block_offsets[block_id])

    def copy_from_parent(self, node_embed_names=ALL, edge_embed_names=ALL):
        """Copy node/edge features from the parent graph.

        Parameters
        ----------
        node_embed_names : a list of lists of strings, optional
            The names of embeddings in each layer.
        edge_embed_names : a list of lists of strings, optional
            The names of embeddings in each block.
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
                for i in range(self.num_blocks):
                    eid = utils.toindex(self.block_parent_eid(i))
                    self._edge_frames[i] = FrameRef(Frame(self._parent._edge_frame[eid]))
            elif edge_embed_names is not None:
                assert isinstance(edge_embed_names, list) \
                        and len(edge_embed_names) == self.num_blocks, \
                        "The specified embedding names should be the same as the number of flows."
                for i in range(self.num_blocks):
                    eid = self.block_parent_eid(i)
                    self._edge_frames[i] = _get_frame(self._parent._edge_frame,
                                                      edge_embed_names[i], eid)

    def copy_to_parent(self, node_embed_names=ALL, edge_embed_names=ALL):
        """Copy node/edge embeddings to the parent graph.

        Parameters
        ----------
        node_embed_names : a list of lists of strings, optional
            The names of embeddings in each layer.
        edge_embed_names : a list of lists of strings, optional
            The names of embeddings in each block.
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
                for i in range(self.num_blocks):
                    eid = utils.toindex(self.block_parent_eid(i))
                    self._parent._edge_frame.update_rows(eid, self._edge_frames[i], inplace=True)
            elif edge_embed_names is not None:
                assert isinstance(edge_embed_names, list) \
                        and len(edge_embed_names) == self.num_blocks, \
                        "The specified embedding names should be the same as the number of flows."
                for i in range(self.num_blocks):
                    eid = utils.toindex(self.block_parent_eid(i))
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

    def layer_in_degree(self, layer_id):
        """Return the in-degree of the nodes in the specified layer.

        Parameters
        ----------
        layer_id : int
            The layer Id.

        Returns
        -------
        Tensor
            The degree of the nodes in the specified layer.
        """
        return self._graph.in_degrees(utils.toindex(self.layer_nid(layer_id))).tousertensor()

    def layer_out_degree(self, layer_id):
        """Return the out-degree of the nodes in the specified layer.

        Parameters
        ----------
        layer_id : int
            The layer Id.

        Returns
        -------
        Tensor
            The degree of the nodes in the specified layer.
        """
        return self._graph.out_degrees(utils.toindex(self.layer_nid(layer_id))).tousertensor()

    def layer_nid(self, layer_id):
        """Get the node Ids in the specified layer.

        Parameters
        ----------
        layer_id : int
            The layer to get the node Ids.

        Returns
        -------
        Tensor
            The node id array.
        """
        layer_id = self._get_layer_id(layer_id)
        assert layer_id + 1 < len(self._layer_offsets)
        start = self._layer_offsets[layer_id]
        end = self._layer_offsets[layer_id + 1]
        return F.arange(int(start), int(end))

    def layer_parent_nid(self, layer_id):
        """Get the node Ids of the parent graph in the specified layer

        Parameters
        ----------
        layer_id : int
            The layer to get the node Ids.

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

    def block_eid(self, block_id):
        """Get the edge Ids in the specified block.

        Parameters
        ----------
        block_id : int
            the specified block to get edge Ids.

        Returns
        -------
        Tensor
            The edge id array.
        """
        block_id = self._get_block_id(block_id)
        start = self._block_offsets[block_id]
        end = self._block_offsets[block_id + 1]
        return F.arange(int(start), int(end))

    def block_parent_eid(self, block_id):
        """Get the edge Ids of the parent graph in the specified block.

        Parameters
        ----------
        block_id : int
            the specified block to get edge Ids.

        Returns
        -------
        Tensor
            The parent edge id array.
        """
        block_id = self._get_block_id(block_id)
        start = self._block_offsets[block_id]
        end = self._block_offsets[block_id + 1]
        return self._edge_mapping.tousertensor()[start:end]

    def set_n_initializer(self, initializer, layer_id=ALL, field=None):
        """Set the initializer for empty node features.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        When a subset of the nodes are assigned a new feature, initializer is
        used to create feature for rest of the nodes.

        Parameters
        ----------
        initializer : callable
            The initializer.
        layer_id : int
            the layer to set the initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        if is_all(layer_id):
            for i in range(self.num_layers):
                self._node_frames[i].set_initializer(initializer, field)
        else:
            self._node_frames[i].set_initializer(initializer, field)

    def set_e_initializer(self, initializer, block_id=ALL, field=None):
        """Set the initializer for empty edge features.

        Initializer is a callable that returns a tensor given the shape, data
        type and device context.

        When a subset of the edges are assigned a new feature, initializer is
        used to create feature for rest of the edges.

        Parameters
        ----------
        initializer : callable
            The initializer.
        block_id : int
            the block to set the initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        if is_all(block_id):
            for i in range(self.num_blocks):
                self._edge_frames[i].set_initializer(initializer, field)
        else:
            self._edge_frames[block_id].set_initializer(initializer, field)


    def register_message_func(self, func, block_id=ALL):
        """Register global message function for a block.

        Once registered, ``func`` will be used as the default
        message function in message passing operations, including
        :func:`block_compute`, :func:`prop_flow`.

        Parameters
        ----------
        func : callable
            Message function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        block_id : int or ALL
            the block to register the message function.
        """
        if is_all(block_id):
            self._message_funcs = [func] * self.num_blocks
        else:
            self._message_funcs[block_id] = func

    def register_reduce_func(self, func, block_id=ALL):
        """Register global message reduce function for a block.

        Once registered, ``func`` will be used as the default
        message reduce function in message passing operations, including
        :func:`block_compute`, :func:`prop_flow`.

        Parameters
        ----------
        func : callable
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        block_id : int or ALL
            the block to register the reduce function.
        """
        if is_all(block_id):
            self._reduce_funcs = [func] * self.num_blocks
        else:
            self._reduce_funcs[block_id] = func

    def register_apply_node_func(self, func, block_id=ALL):
        """Register global node apply function for a block.

        Once registered, ``func`` will be used as the default apply
        node function. Related operations include :func:`apply_layer`,
        :func:`block_compute`, :func:`prop_flow`.

        Parameters
        ----------
        func : callable
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        block_id : int or ALL
            the block to register the apply node function.
        """
        if is_all(block_id):
            self._apply_node_funcs = [func] * self.num_blocks
        else:
            self._apply_node_funcs[block_id] = func

    def register_apply_edge_func(self, func, block_id=ALL):
        """Register global edge apply function for a block.

        Once registered, ``func`` will be used as the default apply
        edge function in :func:`apply_block`.

        Parameters
        ----------
        func : callable
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        block_id : int or ALL
            the block to register the apply edge function.
        """
        if is_all(block_id):
            self._apply_edge_funcs = [func] * self.num_blocks
        else:
            self._apply_edge_funcs[block_id] = func

    def apply_layer(self, layer_id, func="default", v=ALL, inplace=False):
        """Apply node update function on the node embeddings in the specified layer.

        Parameters
        ----------
        layer_id : int
            The specified layer to update node embeddings.
        func : callable or None, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : a list of vertex Ids or ALL.
            The vertices to run the node update function.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if func == "default":
            func = self._apply_node_funcs[block_id]
        if is_all(v):
            v = utils.toindex(slice(0, self.layer_size(layer_id)))
        else:
            v = v - int(self._layer_offsets[layer_id])
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

    def apply_block(self, block_id, func="default", edges=ALL, inplace=False):
        """Apply edge update function on the edge embeddings in the specified layer.

        Parameters
        ----------
        block_id : int
            The specified block to update edge embeddings.
        func : callable or None, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        e : a list of edge Ids or ALL.
            The edges to run the edge update function.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if func == "default":
            func = self._apply_edge_funcs[block_id]
        assert func is not None

        if is_all(edges):
            u = utils.toindex(self._layer_local_nid(block_id))
            v = utils.toindex(self._layer_local_nid(block_id + 1))
            eid = utils.toindex(slice(0, self.block_size(block_id)))
        elif isinstance(edges, tuple):
            u, v = edges
            # Rewrite u, v to handle edge broadcasting and multigraph.
            u, v, eid = self._graph.edge_ids(utils.toindex(u), utils.toindex(v))
            u = utils.toindex(u.tousertensor() - int(self._layer_offsets[block_id]))
            v = utils.toindex(v.tousertensor() - int(self._layer_offsets[block_id + 1]))
            eid = utils.toindex(eid.tousertensor() - int(self._block_offsets[block_id]))
        else:
            eid = utils.toindex(edges)
            u, v, _ = self._graph.find_edges(eid)
            u = utils.toindex(u.tousertensor() - int(self._layer_offsets[block_id]))
            v = utils.toindex(v.tousertensor() - int(self._layer_offsets[block_id + 1]))
            eid = utils.toindex(edges - int(self._block_offsets[block_id]))

        with ir.prog() as prog:
            scheduler.schedule_nodeflow_apply_edges(graph=self,
                                                    block_id=block_id,
                                                    u=u,
                                                    v=v,
                                                    eid=eid,
                                                    apply_func=func,
                                                    inplace=inplace)
            Runtime.run(prog)

    def _conv_local_nid(self, nid, layer_id):
        layer_id = self._get_layer_id(layer_id)
        return nid - int(self._layer_offsets[layer_id])

    def _conv_local_eid(self, eid, block_id):
        block_id = self._get_block_id(block_id)
        return eid - int(self._block_offsets[block_id])

    def block_compute(self, block_id, message_func="default", reduce_func="default",
                      apply_node_func="default", v=ALL, inplace=False):
        """Perform the computation on the specified block. It's similar to `pull`
        in DGLGraph.
        On the given block i, it runs `pull` on nodes in layer i+1, which generates
        messages on edges in block i, runs the reduce function and node update
        function on nodes in layer i+1.

        Parameters
        ----------
        block_id : int
            The block to run the computation.
        message_func : callable, optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_func : callable, optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_func : callable, optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        v : a list of vertex Ids or ALL.
            The specified nodes in layer i+1 to run the computation.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if message_func == "default":
            message_func = self._message_funcs[block_id]
        if reduce_func == "default":
            reduce_func = self._reduce_funcs[block_id]
        if apply_node_func == "default":
            apply_node_func = self._apply_node_funcs[block_id]

        assert message_func is not None
        assert reduce_func is not None

        if is_all(v):
            dest_nodes = utils.toindex(self.layer_nid(block_id + 1))
            u, v, _ = self._graph.in_edges(dest_nodes)
            u = utils.toindex(self._conv_local_nid(u.tousertensor(), block_id))
            v = utils.toindex(self._conv_local_nid(v.tousertensor(), block_id + 1))
            dest_nodes = utils.toindex(F.arange(0, self.layer_size(block_id + 1)))
            eid = utils.toindex(F.arange(0, self.block_size(block_id)))
        else:
            dest_nodes = utils.toindex(v)
            u, v, eid = self._graph.in_edges(dest_nodes)
            assert len(u) > 0, "block_compute must run on edges"
            u = utils.toindex(self._conv_local_nid(u.tousertensor(), block_id))
            v = utils.toindex(self._conv_local_nid(v.tousertensor(), block_id + 1))
            dest_nodes = utils.toindex(self._conv_local_nid(dest_nodes.tousertensor(),
                                                            block_id + 1))
            eid = utils.toindex(self._conv_local_eid(eid.tousertensor(), block_id))

        with ir.prog() as prog:
            scheduler.schedule_nodeflow_compute(graph=self,
                                                block_id=block_id,
                                                u=u,
                                                v=v,
                                                eid=eid,
                                                dest_nodes=dest_nodes,
                                                message_func=message_func,
                                                reduce_func=reduce_func,
                                                apply_func=apply_node_func,
                                                inplace=inplace)
            Runtime.run(prog)

    def prop_flow(self, message_funcs="default", reduce_funcs="default",
                  apply_node_funcs="default", flow_range=ALL, inplace=False):
        """Perform the computation on flows. By default, it runs on all blocks, one-by-one.
        On block i, it runs `pull` on nodes in layer i+1, which generates
        messages on edges in block i, runs the reduce function and node update
        function on nodes in layer i+1.

        Users can specify a list of message functions, reduce functions and
        node apply functions, one for each block. Thus, when a list is given,
        the length of the list should be the same as the number of blocks.

        Parameters
        ----------
        message_funcs : a callable, a list of callable, optional
            Message functions on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        reduce_funcs : a callable, a list of callable, optional
            Reduce functions on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        apply_node_funcs : a callable, a list of callable, optional
            Apply functions on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.
        range : int or a slice or ALL.
            The specified blocks to run the computation.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if is_all(flow_range):
            flow_range = range(0, self.num_blocks)
        elif isinstance(flow_range, slice):
            if slice.step is not 1:
                raise DGLError("We can't propogate flows and skip some of them")
            flow_range = range(flow_range.start, flow_range.stop)
        else:
            raise DGLError("unknown flow range")

        for i in flow_range:
            if message_funcs == "default":
                message_func = self._message_funcs[i]
            elif isinstance(message_funcs, list):
                message_func = message_funcs[i]
            else:
                message_func = message_funcs

            if reduce_funcs == "default":
                reduce_func = self._reduce_funcs[i]
            elif isinstance(reduce_funcs, list):
                reduce_func = reduce_funcs[i]
            else:
                reduce_func = reduce_funcs

            if apply_node_funcs == "default":
                apply_node_func = self._apply_node_funcs[i]
            elif isinstance(apply_node_funcs, list):
                apply_node_func = apply_node_funcs[i]
            else:
                apply_node_func = apply_node_funcs

            self.block_compute(i, message_func, reduce_func, apply_node_func,
                               inplace=inplace)


def create_full_node_flow(g, num_layers):
    """Convert a full graph to NodeFlow to run a L-layer GNN model.

    Parameters
    ----------
    g : DGLGraph
        a DGL graph
    num_layers : int
        The number of layers

    Returns
    -------
    NodeFlow
        a NodeFlow with a specified number of layers.
    """
    seeds = [utils.toindex(F.arange(0, g.number_of_nodes()))]
    nfi = g._graph.neighbor_sampling(seeds, g.number_of_nodes(), num_layers, 'in', None)
    return NodeFlow(g, nfi[0])
