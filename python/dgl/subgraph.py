"""DGLSubGraph"""
from __future__ import absolute_import

import networkx as nx
import dgl.backend as F
from dgl.frame import Frame, FrameRef
from dgl.graph import DGLGraph
from dgl.nx_adapt import nx_init
import dgl.utils as utils

class DGLSubGraph(DGLGraph):
    # TODO(gaiyu): ReadOnlyGraph
    def __init__(self,
                 parent,
                 nodes=None,
                 edges=None):
        '''
        Constructs a subgraph from either given nodes or given edges.

        nodes and edges are mutually exclusive.
        '''
        # create subgraph and relabel
        if edges is None:
            nx_sg = nx.DiGraph.subgraph(parent, nodes)
            edges = list(nx_sg.edges)
        else:
            nx_sg = nx.DiGraph.edge_subgraph(parent, edges)
            nodes = list(nx_sg.nodes)
        # node id
        # TODO(minjie): context
        nid = F.tensor(nodes, dtype=F.int64)
        # edge id
        # TODO(minjie): slow, context
        u, v = zip(*edges)
        u = list(u)
        v = list(v)
        eid = parent.cached_graph.get_edge_id(u, v)

        # relabel
        self._node_mapping = utils.build_relabel_dict(nodes)
        nx_sg = nx.relabel.relabel_nodes(nx_sg, self._node_mapping)

        # init
        self._edge_list = []
        nx_init(self,
                self._add_node_callback,
                self._add_edge_callback,
                self._del_node_callback,
                self._del_edge_callback,
                nx_sg,
                **parent.graph)
        # cached graph and storage
        self._cached_graph = None
        if parent._node_frame.num_rows == 0:
            self._node_frame = FrameRef()
        else:
            self._node_frame = FrameRef(Frame(parent._node_frame[nid]))
        if parent._edge_frame.num_rows == 0:
            self._edge_frame = FrameRef()
        else:
            self._edge_frame = FrameRef(Frame(parent._edge_frame[eid]))
        # other class members
        self._msg_graph = None
        self._msg_frame = FrameRef()
        self._message_func = parent._message_func
        self._reduce_func = parent._reduce_func
        self._update_func = parent._update_func
        self._edge_func = parent._edge_func
        self._context = parent._context
