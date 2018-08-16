"""Line graph generator."""
from __future__ import absolute_import

import networkx as nx

from dgl.graph import DGLGraph

def line_graph(G):
    """Create the line graph that shares the underlying features.

    The node features of the result line graph will share the edge features
    of the given graph.

    Parameters
    ----------
    G : DGLGraph
        The input graph.
    """
    LG = nx.line_graph(G)
    relabel_map = {}
    for i, e in enumerate(G.edge_list):
        relabel_map[e] = i
    nx.relabel.relabel_nodes(LG, relabel_map)
    return DGLGraph(LG, node_frame=G._edge_frame)
