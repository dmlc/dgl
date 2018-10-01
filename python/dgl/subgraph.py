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
        self._parent_nid = utils.toindex(nodes)
        self._graph, self._parent_eid = parent._graph.node_subgraph(self._parent_nid)
        if parent._node_frame.num_rows != 0:
            self._node_frame = FrameRef(Frame(parent._node_frame[self._parent_nid]))
        if parent._edge_frame.num_rows != 0:
            self._edge_frame = FrameRef(Frame(parent._edge_frame[self._parent_eid]))
        self.reset_messages()

    def copy_from(self, parent):
        """Copy node/edge features from the parent graph.
        TODO do we need this?

        All old features will be removed.

        Parameters
        ----------
        parent : DGLGraph
            The parent graph to copy from.
        """
        pass
