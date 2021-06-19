"""Class for NodeFlow data structure."""
from __future__ import absolute_import

from .._ffi.object import register_object, ObjectBase
from .._ffi.function import _init_api
from ..base import ALL, is_all, DGLError, dgl_warning
from .. import backend as F
from .frame import Frame, FrameRef
from .graph import DGLBaseGraph
from ..graph_index import transform_ids
from .runtime import ir, scheduler, Runtime
from .. import utils
from .view import LayerView, BlockView

__all__ = ['NodeFlow']

@register_object('graph.NodeFlow')
class NodeFlowObject(ObjectBase):
    """NodeFlow object"""

    @property
    def graph(self):
        """The graph structure of this nodeflow.

        Returns
        -------
        GraphIndex
        """
        return _CAPI_NodeFlowGetGraph(self)

    @property
    def layer_offsets(self):
        """The offsets of each layer.

        Returns
        -------
        NDArray
        """
        return _CAPI_NodeFlowGetLayerOffsets(self)

    @property
    def block_offsets(self):
        """The offsets of each block.

        Returns
        -------
        NDArray
        """
        return _CAPI_NodeFlowGetBlockOffsets(self)

    @property
    def node_mapping(self):
        """Mapping array from nodeflow node id to parent graph

        Returns
        -------
        NDArray
        """
        return _CAPI_NodeFlowGetNodeMapping(self)

    @property
    def edge_mapping(self):
        """Mapping array from nodeflow edge id to parent graph

        Returns
        -------
        NDArray
        """
        return _CAPI_NodeFlowGetEdgeMapping(self)

class NodeFlow(DGLBaseGraph):
    """The NodeFlow class stores the sampling results of Neighbor
    sampling and Layer-wise sampling.

    These sampling algorithms generate graphs with multiple layers. The
    edges connect the nodes between two layers, which forms *blocks*, while
    there don't exist edges between the nodes in the same layer. As illustrated
    in the figure, the last layer stores the target (seed) nodes where neighbors
    are sampled from. Neighbors reached in different hops are placed in different
    layers. Edges that connect to the neighbors in the next hop are placed
    in a block.
    We store extra information, such as the node and edge mapping from
    the NodeFlow graph to the parent graph.

    .. image:: https://data.dgl.ai/api/sampling.nodeflow.png

    DO NOT create NodeFlow object directly. Use sampling method to
    generate NodeFlow instead.

    Parameters
    ----------
    parent : DGLGraphStale
        The parent graph.
    nfobj : NodeFlowObject
        The nodeflow object
    """
    def __init__(self, parent, nfobj):
        super(NodeFlow, self).__init__(nfobj.graph)
        dgl_warning('NodeFlow APIs are deprecated starting from v0.5. Please read our'
                    ' guide<link> for how to use the new sampling APIs.')
        self._parent = parent
        self._node_mapping = utils.toindex(nfobj.node_mapping)
        self._edge_mapping = utils.toindex(nfobj.edge_mapping)
        self._layer_offsets = utils.toindex(nfobj.layer_offsets).tonumpy()
        self._block_offsets = utils.toindex(nfobj.block_offsets).tonumpy()
        # node/edge frames
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
        """The layer Id might be negative. We need to convert it to the actual layer Id.
        """
        if layer_id >= 0:
            return layer_id
        else:
            return self.num_layers + layer_id

    def _get_block_id(self, block_id):
        """The block Id might be negative. We need to convert it to the actual block Id.
        """
        if block_id >= 0:
            return block_id
        else:
            return self.num_blocks + block_id

    def _get_node_frame(self, layer_id):
        return self._node_frames[layer_id]

    def _get_edge_frame(self, block_id):
        return self._edge_frames[block_id]

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
        """
        return LayerView(self)

    @property
    def blocks(self):
        """Return a BlockView of this NodeFlow.

        This is mainly for usage like:
        * `g.blocks[1].data['h']` to get the edge features of blocks from layer#1 to layer#2.
        """
        return BlockView(self)

    def node_attr_schemes(self, layer_id):
        """Return the node feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Parameters
        ----------
        layer_id : int
            the specified layer to get node data scheme.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        layer_id = self._get_layer_id(layer_id)
        return self._node_frames[layer_id].schemes

    def edge_attr_schemes(self, block_id):
        """Return the edge feature schemes.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Parameters
        ----------
        block_id : int
            the specified block to get edge data scheme.

        Returns
        -------
        dict of str to schemes
            The schemes of edge feature columns.
        """
        block_id = self._get_block_id(block_id)
        return self._edge_frames[block_id].schemes

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

    def copy_from_parent(self, node_embed_names=ALL, edge_embed_names=ALL, ctx=None):
        """Copy node/edge features from the parent graph.

        Parameters
        ----------
        node_embed_names : a list of lists of strings, optional
            The names of embeddings in each layer.
        edge_embed_names : a list of lists of strings, optional
            The names of embeddings in each block.
        ctx : Context
            The device to copy tensor to. If None, features will stay at its original device
        """
        if self._parent._node_frame.num_rows != 0 and self._parent._node_frame.num_columns != 0:
            if is_all(node_embed_names):
                for i in range(self.num_layers):
                    nid = utils.toindex(self.layer_parent_nid(i))
                    self._node_frames[i] = FrameRef(Frame(_copy_frame(
                        self._parent._node_frame[nid], ctx)))
            elif node_embed_names is not None:
                assert isinstance(node_embed_names, list) \
                        and len(node_embed_names) == self.num_layers, \
                        "The specified embedding names should be the same as the number of layers."
                for i in range(self.num_layers):
                    nid = self.layer_parent_nid(i)
                    self._node_frames[i] = _get_frame(self._parent._node_frame,
                                                      node_embed_names[i], nid, ctx)

        if self._parent._edge_frame.num_rows != 0 and self._parent._edge_frame.num_columns != 0:
            if is_all(edge_embed_names):
                for i in range(self.num_blocks):
                    eid = utils.toindex(self.block_parent_eid(i))
                    self._edge_frames[i] = FrameRef(Frame(_copy_frame(
                        self._parent._edge_frame[eid], ctx)))
            elif edge_embed_names is not None:
                assert isinstance(edge_embed_names, list) \
                        and len(edge_embed_names) == self.num_blocks, \
                        "The specified embedding names should be the same as the number of flows."
                for i in range(self.num_blocks):
                    eid = self.block_parent_eid(i)
                    self._edge_frames[i] = _get_frame(self._parent._edge_frame,
                                                      edge_embed_names[i], eid, ctx)

    def copy_to_parent(self, node_embed_names=ALL, edge_embed_names=ALL):
        """Copy node/edge embeddings to the parent graph.

        Note: if a node in the parent graph appears in multiple layers and they
        in the NodeFlow has node data with the same name, the data of this node
        in the lower layer will overwrite the node data in previous layer.

        For example, node 5 in the parent graph appears in layer 0 and 1 and
        they have the same node data 'h'. The node data in layer 1 of this node
        will overwrite its data in layer 0 when copying the data back.

        To avoid this, users can give node data in each layer a different name.

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
        nid = utils.toindex(nid)
        return F.gather_row(self._node_mapping.tousertensor(), nid.tousertensor())

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
        eid = utils.toindex(eid)
        return F.gather_row(self._edge_mapping.tousertensor(), eid.tousertensor())

    def map_from_parent_nid(self, layer_id, parent_nids, remap_local=False):
        """Map parent node Ids to NodeFlow node Ids in a certain layer.

        If `remap_local` is True, it returns the node Ids local to the layer.
        Otherwise, the node Ids are unique in the NodeFlow.

        Parameters
        ----------
        layer_id : int
            The layer Id.
        parent_nids: list or Tensor
            Node Ids in the parent graph.
        remap_local: boolean
            Remap layer/block-level local Id if True; otherwise, NodeFlow-level Id.

        Returns
        -------
        Tensor
            Node Ids in the NodeFlow.
        """
        layer_id = self._get_layer_id(layer_id)
        parent_nids = utils.toindex(parent_nids)
        layers = self._layer_offsets
        start = int(layers[layer_id])
        end = int(layers[layer_id + 1])
        # TODO(minjie): should not directly use []
        mapping = self._node_mapping.tousertensor()
        mapping = mapping[start:end]
        mapping = utils.toindex(mapping)
        nflow_ids = transform_ids(mapping, parent_nids)
        if remap_local:
            return nflow_ids.tousertensor()
        else:
            return nflow_ids.tousertensor() + int(self._layer_offsets[layer_id])

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
        layer_id = self._get_layer_id(layer_id)
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
        layer_id = self._get_layer_id(layer_id)
        return self._graph.out_degrees(utils.toindex(self.layer_nid(layer_id))).tousertensor()

    def layer_nid(self, layer_id):
        """Get the node Ids in the specified layer.

        The returned node Ids are unique in the NodeFlow.

        Parameters
        ----------
        layer_id : int
            The layer to get the node Ids.

        Returns
        -------
        Tensor
            The node ids.
        """
        layer_id = self._get_layer_id(layer_id)
        assert layer_id + 1 < len(self._layer_offsets)
        start = self._layer_offsets[layer_id]
        end = self._layer_offsets[layer_id + 1]
        return F.arange(int(start), int(end))

    def layer_parent_nid(self, layer_id):
        """Get the node Ids of the parent graph in the specified layer

        layer_parent_nid(-1) returns seed vertices for this NodeFlow.

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
        # TODO(minjie): should not directly use []
        return self._node_mapping.tousertensor()[start:end]

    def block_eid(self, block_id):
        """Get the edge Ids in the specified block.

        The returned edge Ids are unique in the NodeFlow.

        Parameters
        ----------
        block_id : int
            the specified block to get edge Ids.

        Returns
        -------
        Tensor
            The edge ids of the block in the NodeFlow.
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
            The edge ids of the block in the parent graph.
        """
        block_id = self._get_block_id(block_id)
        start = self._block_offsets[block_id]
        end = self._block_offsets[block_id + 1]
        # TODO(minjie): should not directly use []
        ret = self._edge_mapping.tousertensor()[start:end]
        # If `add_self_loop` is enabled, the returned parent eid can be -1.
        # We have to make sure this case doesn't happen.
        assert F.asnumpy(ret == -1).sum(0) == 0, "The eid in the parent graph is invalid."
        return ret

    def block_edges(self, block_id, remap_local=False):
        """Return the edges in a block.

        If remap_local is True, returned indices u, v, eid will be remapped to local
        Ids (i.e. starting from 0) in the block or in the layer. Otherwise,
        u, v, eid are unique in the NodeFlow.

        Parameters
        ----------
        block_id : int
            The specified block to return the edges.
        remap_local : boolean
            Remap layer/block-level local Id if True; otherwise, NodeFlow-level Id.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        block_id = self._get_block_id(block_id)
        layer0_size = self._layer_offsets[block_id + 1] - self._layer_offsets[block_id]
        rst = _CAPI_NodeFlowGetBlockAdj(self._graph, "coo",
                                        int(layer0_size),
                                        int(self._layer_offsets[block_id + 1]),
                                        int(self._layer_offsets[block_id + 2]),
                                        remap_local)
        idx = utils.toindex(rst(0)).tousertensor()
        eid = utils.toindex(rst(1))
        num_edges = int(len(idx) / 2)
        assert len(eid) == num_edges
        return idx[num_edges:len(idx)], idx[0:num_edges], eid.tousertensor()

    def block_adjacency_matrix(self, block_id, ctx):
        """Return the adjacency matrix representation for a specific block in a NodeFlow.

        A row of the returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        Parameters
        ----------
        block_id : int
            The specified block to return the adjacency matrix.
        ctx : context
            The context of the returned matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        Tensor
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        block_id = self._get_block_id(block_id)
        fmt = F.get_preferred_sparse_format()
        # We need to extract two layers.
        layer0_size = self._layer_offsets[block_id + 1] - self._layer_offsets[block_id]
        rst = _CAPI_NodeFlowGetBlockAdj(self._graph, fmt,
                                        int(layer0_size),
                                        int(self._layer_offsets[block_id + 1]),
                                        int(self._layer_offsets[block_id + 2]),
                                        True)
        num_rows = self.layer_size(block_id + 1)
        num_cols = self.layer_size(block_id)

        if fmt == "csr":
            indptr = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            indices = F.copy_to(utils.toindex(rst(1)).tousertensor(), ctx)
            shuffle = utils.toindex(rst(2))
            dat = F.ones(indices.shape, dtype=F.float32, ctx=ctx)
            return F.sparse_matrix(dat, ('csr', indices, indptr),
                                   (num_rows, num_cols))[0], shuffle.tousertensor()
        elif fmt == "coo":
            ## FIXME(minjie): data type
            idx = F.copy_to(utils.toindex(rst(0)).tousertensor(), ctx)
            m = self.block_size(block_id)
            idx = F.reshape(idx, (2, m))
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            adj, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (num_rows, num_cols))
            return adj, shuffle_idx
        else:
            raise Exception("unknown format")

    def block_incidence_matrix(self, block_id, typestr, ctx):
        """Return the incidence matrix representation of the block.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are two types of an incidence matrix `I`:

        * ``in``:

            - I[v, e] = 1 if e is the in-edge of v (or v is the dst node of e);
            - I[v, e] = 0 otherwise.

        * ``out``:

            - I[v, e] = 1 if e is the out-edge of v (or v is the src node of e);
            - I[v, e] = 0 otherwise.

        "both" isn't defined in the block of a NodeFlow.

        Parameters
        ----------
        block_id : int
            The specified block to return the incidence matrix.
        typestr : str
            Can be either "in", "out" or "both"
        ctx : context
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        Tensor
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        block_id = self._get_block_id(block_id)
        src, dst, eid = self.block_edges(block_id, remap_local=True)
        src = F.copy_to(src, ctx)  # the index of the ctx will be cached
        dst = F.copy_to(dst, ctx)  # the index of the ctx will be cached
        eid = F.copy_to(eid, ctx)  # the index of the ctx will be cached
        if typestr == 'in':
            n = self.layer_size(block_id + 1)
            m = self.block_size(block_id)
            row = F.unsqueeze(dst, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        elif typestr == 'out':
            n = self.layer_size(block_id)
            m = self.block_size(block_id)
            row = F.unsqueeze(src, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.cat([row, col], dim=0)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        else:
            raise DGLError('Invalid incidence matrix type: %s' % str(typestr))
        return inc, shuffle_idx

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
            layer_id = self._get_layer_id(layer_id)
            self._node_frames[layer_id].set_initializer(initializer, field)

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
            block_id = self._get_block_id(block_id)
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
            block_id = self._get_block_id(block_id)
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
            block_id = self._get_block_id(block_id)
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
            block_id = self._get_block_id(block_id)
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
            block_id = self._get_block_id(block_id)
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
            The vertex Ids (unique in the NodeFlow) to run the node update function.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
        """
        layer_id = self._get_layer_id(layer_id)
        if func == "default":
            func = self._apply_node_funcs[layer_id]
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


    def apply_block(self, block_id, func="default", edges=ALL, inplace=False):
        """Apply edge update function on the edge embeddings in the specified layer.

        Parameters
        ----------
        block_id : int
            The specified block to update edge embeddings.
        func : callable or None, optional
            Apply function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.
        edges : a list of edge Ids or ALL.
            The edges Id to run the edge update function.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.
        """
        block_id = self._get_block_id(block_id)
        if func == "default":
            func = self._apply_edge_funcs[block_id]
        assert func is not None

        if is_all(edges):
            u, v, _ = self.block_edges(block_id, remap_local=True)
            u = utils.toindex(u)
            v = utils.toindex(v)
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

    def _glb2lcl_nid(self, nid, layer_id):
        layer_id = self._get_layer_id(layer_id)
        return nid - int(self._layer_offsets[layer_id])

    def _glb2lcl_eid(self, eid, block_id):
        block_id = self._get_block_id(block_id)
        return eid - int(self._block_offsets[block_id])

    def block_compute(self, block_id, message_func="default", reduce_func="default",
                      apply_node_func="default", v=ALL, inplace=False):
        """Perform the computation on the specified block. It's similar to `pull`
        in DGLGraphStale.
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
            The Node Ids (unique in the NodeFlow) in layer block_id+1 to run the computation.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        block_id = self._get_block_id(block_id)
        if message_func == "default":
            message_func = self._message_funcs[block_id]
        if reduce_func == "default":
            reduce_func = self._reduce_funcs[block_id]
        if apply_node_func == "default":
            apply_node_func = self._apply_node_funcs[block_id]

        assert message_func is not None
        assert reduce_func is not None

        if is_all(v):
            with ir.prog() as prog:
                scheduler.schedule_nodeflow_update_all(graph=self,
                                                       block_id=block_id,
                                                       message_func=message_func,
                                                       reduce_func=reduce_func,
                                                       apply_func=apply_node_func)
                Runtime.run(prog)
        else:
            dest_nodes = utils.toindex(v)
            u, v, eid = self._graph.in_edges(dest_nodes)
            assert len(u) > 0, "block_compute must run on edges"
            u = utils.toindex(self._glb2lcl_nid(u.tousertensor(), block_id))
            v = utils.toindex(self._glb2lcl_nid(v.tousertensor(), block_id + 1))
            dest_nodes = utils.toindex(
                self._glb2lcl_nid(dest_nodes.tousertensor(), block_id + 1))
            eid = utils.toindex(self._glb2lcl_eid(eid.tousertensor(), block_id))

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
        flow_range : int or a slice or ALL.
            The specified blocks to run the computation.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        if is_all(flow_range):
            flow_range = range(0, self.num_blocks)
        elif isinstance(flow_range, slice):
            if flow_range.step != 1:
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

    @property
    def canonical_etype(self):
        """Return canonical edge type to be compatible with GraphAdapter
        """
        return (None, None, None)

def _copy_to_like(arr1, arr2):
    return F.copy_to(arr1, F.context(arr2))

def _get_frame(frame, names, ids, ctx):
    col_dict = {}
    for name in names:
        col = F.gather_row(frame[name], _copy_to_like(ids, frame[name]))
        if ctx:
            col = F.copy_to(col, ctx)
        col_dict[name] = col
    if len(col_dict) == 0:
        return FrameRef(Frame(num_rows=len(ids)))
    else:
        return FrameRef(Frame(col_dict))

def _copy_frame(frame, ctx):
    new_frame = {}
    for name in frame:
        new_frame[name] = F.copy_to(frame[name], ctx) if ctx else frame[name]
    return new_frame


def _update_frame(frame, names, ids, new_frame):
    col_dict = {name: new_frame[name] for name in names}
    if len(col_dict) > 0:
        # This will raise error for tensorflow, because inplace update is not supported
        frame.update_rows(ids, FrameRef(Frame(col_dict)), inplace=True)

_init_api("dgl._deprecate.nodeflow", __name__)
