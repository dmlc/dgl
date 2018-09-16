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
        self._node_mapping = utils.build_relabel_dict(nodes)
        self._parent_nid = utils.toindex(nodes)
        eids = []
        # create subgraph
        for eid, (u, v) in enumerate(parent.edge_list):
            if u in self._node_mapping and v in self._node_mapping:
                self.add_edge(self._node_mapping[u],
                              self._node_mapping[v])
                eids.append(eid)
        self._parent_eid = utils.toindex(eids)

    def copy_from(self, parent):
        """Copy node/edge features from the parent graph.

        All old features will be removed.

        Parameters
        ----------
        parent : DGLGraph
            The parent graph to copy from.
        """
        if parent._node_frame.num_rows != 0:
            self._node_frame = FrameRef(Frame(parent._node_frame[self._parent_nid]))
        if parent._edge_frame.num_rows != 0:
            self._edge_frame = FrameRef(Frame(parent._edge_frame[self._parent_eid]))
