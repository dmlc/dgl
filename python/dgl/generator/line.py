"""Line graph generator."""
from __future__ import absolute_import

import networkx as nx
import numpy as np

import dgl.backend as F
from dgl.graph import DGLGraph
from dgl.frame import FrameRef

def line_graph(G, no_backtracking=False):
    """Create the line graph that shares the underlying features.

    The node features of the result line graph will share the edge features
    of the given graph.

    Parameters
    ----------
    G : DGLGraph
        The input graph.
    no_backtracking : bool
        Whether the backtracking edges are included in the line graph.
        If i~j and j~i are two edges in original graph G, then
        (i,j)~(j,i) and (j,i)~(i,j) are the "backtracking" edges on
        the line graph.
    """
    L = nx.DiGraph()
    for eid, from_node in enumerate(G.edge_list):
        L.add_node(from_node)
        for to_node in G.edges(from_node[1]):
            if no_backtracking and to_node[1] == from_node[0]:
                continue
            L.add_edge(from_node, to_node)
    relabel_map = {}
    for i, e in enumerate(G.edge_list):
        relabel_map[e] = i
    nx.relabel.relabel_nodes(L, relabel_map)
    return DGLGraph(L, node_frame=G._edge_frame)
