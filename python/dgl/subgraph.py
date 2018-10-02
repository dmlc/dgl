"""DGLSubGraph"""
from __future__ import absolute_import

import networkx as nx

from . import backend as F
from .frame import Frame, FrameRef
from .graph import DGLGraph
from .nx_adapt import nx_init
from . import utils

class DGLSubGraph(DGLGraph):
    # TODO(gaiyu): ReadOnlyGraph
    def __init__(self,
                 parent,
                 nodes):
        super(DGLSubGraph, self).__init__()
        # relabel nodes
        self._parent = parent
        self._parent_nid = utils.toindex(nodes)
        self._graph, self._parent_eid = parent._graph.node_subgraph(self._parent_nid)
        self.reset_messages()

    def copy_to_parent(self, inplace=False):
        self._parent._node_frame.update_rows(self._parent_nid, self._node_frame, inplace=inplace)
        self._parent._edge_frame.update_rows(self._parent_eid, self._edge_frame, inplace=inplace)

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        if self._parent._node_frame.num_rows != 0:
            self._node_frame = FrameRef(Frame(self._parent._node_frame[self._parent_nid]))
        if self._parent._edge_frame.num_rows != 0:
            self._edge_frame = FrameRef(Frame(self._parent._edge_frame[self._parent_eid]))
