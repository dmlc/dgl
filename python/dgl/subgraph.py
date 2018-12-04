"""Class for subgraph data structure."""
from __future__ import absolute_import

import networkx as nx

from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLGraph
from . import utils
from .graph_index import map_to_subgraph_nid

class DGLSubGraph(DGLGraph):
    """The subgraph class.

    There are two subgraph modes: shared and non-shared.

    For the "non-shared" mode, the user needs to explicitly call
    ``copy_from_parent`` to copy node/edge features from its parent graph.
    * If the user tries to get node/edge features before ``copy_from_parent``,
      s/he will get nothing.
    * If the subgraph already has its own node/edge features, ``copy_from_parent``
      will override them.
    * Any update on the subgraph's node/edge features will not be seen
      by the parent graph. As such, the memory consumption is of the order
      of the subgraph size.
    * To write the subgraph's node/edge features back to parent graph. There are two options:
      (1) Use ``copy_to_parent`` API to write node/edge features back.
      (2) [TODO] Use ``dgl.merge`` to merge multiple subgraphs back to one parent.

    The "shared" mode is currently not supported.

    The subgraph is read-only on structure; graph mutation is not allowed.

    Parameters
    ----------
    parent : DGLGraph
        The parent graph
    parent_nid : utils.Index
        The induced parent node ids in this subgraph.
    parent_eid : utils.Index
        The induced parent edge ids in this subgraph.
    graph_idx : GraphIndex
        The graph index.
    shared : bool, optional
        Whether the subgraph shares node/edge features with the parent graph.
    """
    def __init__(self, parent, parent_nid, parent_eid, graph_idx, shared=False):
        super(DGLSubGraph, self).__init__(graph_data=graph_idx,
                                          readonly=graph_idx.is_readonly())
        self._parent = parent
        self._parent_nid = parent_nid
        self._parent_eid = parent_eid

    # override APIs
    def add_nodes(self, num, reprs=None):
        """Add nodes. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edge(self, u, v, reprs=None):
        """Add one edge. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    def add_edges(self, u, v, reprs=None):
        """Add many edges. Disabled because BatchedDGLGraph is read-only."""
        raise RuntimeError('Readonly graph. Mutation is not allowed.')

    @property
    def parent_nid(self):
        """Get the parent node ids.

        The returned tensor can be used as a map from the node id
        in this subgraph to the node id in the parent graph.

        Returns
        -------
        Tensor
            The parent node id array.
        """
        return self._parent_nid.tousertensor()

    def _get_parent_eid(self):
        # The parent eid might be lazily evaluated and thus may not
        # be an index. Instead, it's a lambda function that returns
        # an index.
        if isinstance(self._parent_eid, utils.Index):
            return self._parent_eid
        else:
            return self._parent_eid()

    @property
    def parent_eid(self):
        """Get the parent edge ids.

        The returned tensor can be used as a map from the edge id
        in this subgraph to the edge id in the parent graph.

        Returns
        -------
        Tensor
            The parent edge id array.
        """
        return self._get_parent_eid().tousertensor()

    def copy_to_parent(self, inplace=False):
        """Write node/edge features to the parent graph.

        Parameters
        ----------
        inplace : bool
            If true, use inplace write (no gradient but faster)
        """
        self._parent._node_frame.update_rows(
                self._parent_nid, self._node_frame, inplace=inplace)
        if self._parent._edge_frame.num_rows != 0:
            self._parent._edge_frame.update_rows(
                    self._get_parent_eid(), self._edge_frame, inplace=inplace)

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        if self._parent._node_frame.num_rows != 0:
            self._node_frame = FrameRef(Frame(
                self._parent._node_frame[self._parent_nid]))
        if self._parent._edge_frame.num_rows != 0:
            self._edge_frame = FrameRef(Frame(
                self._parent._edge_frame[self._get_parent_eid()]))

    def map_to_subgraph_nid(self, parent_vids):
        """Map the node Ids in the parent graph to the node Ids in the subgraph.

        Parameters
        ----------
        parent_vids : list, tensor
            The node ID array in the parent graph.

        Returns
        -------
        tensor
            The node ID array in the subgraph.
        """
        return map_to_subgraph_nid(self._graph, utils.toindex(parent_vids)).tousertensor()
