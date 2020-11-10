#-*- coding:utf-8 -*-

# Utility file for graph queries

import tkinter
import matplotlib
matplotlib.use('TkAgg')

import networkx as nx
import matplotlib.pylab as plt
import torch as th
import dgl
from dgl.sampling import sample_neighbors


def extract_subgraph(graph, seed_nodes, hops=2):
    """
    For the explainability, extract the subgraph of a seed node with the hops specified.

    Parameters
    ----------
    graph:      DGLGraph, the full graph to extract from. This time, assume it is a homograph
    seed_nodes:  Tensor, index of a node in the graph
    hops:       Integer, the number of hops to extract

    Returns
    -------
    sub_graph: DGLGraph, a sub graph
    """

    one_hop = sample_neighbors(graph, seed_nodes, -1)
    one_hop_neighbors = one_hop.edges()[0]

    one_hop_seeds = th.cat([one_hop_neighbors, seed_nodes])
    two_hop = sample_neighbors(graph, one_hop_seeds, -1)

    old_src, old_dst = two_hop.edges()[0], two_hop.edges()[1]

    edge_all = th.cat([old_src, old_dst])
    nidx, new_edges_all = th.unique(edge_all, return_inverse=True)

    n = int(new_edges_all.shape[0] / 2)

    new_src = new_edges_all[:n]
    new_dst = new_edges_all[n:]

    new_two_hop = dgl.DGLGraph((new_src, new_dst))

    new_nidx = th.nonzero(nidx==seed_nodes, as_tuple=True)[0][0]

    return new_two_hop, new_nidx


def visualize_sub_graph(sub_graph, edge_weights=None):
    """
    Use networkx to visualize the sub_graph and,
    if edge weights are given, set edges with different fading of blue.

    Parameters
    ----------
    sub_graph: DGLGraph, the sub_graph to be visualized.
    edge_weights: Tensor, the same number of edges. Values are (0,1), default is None

    Returns
    show the sub_graph
    -------

    """
    # Extract original idx and map to the new networkx graph


    # Convert to networkx graph
    g = dgl.to_networkx(sub_graph)
    nx_edges = g.edges(data=True)

    pos = nx.spring_layout(g)

    if edge_weights is None:
        options = {"node_size": 1000,
                   "alpha": 0.9,
                   "font_size":24,
                   "width": 4,
                   }
    else:

        ec = [edge_weights[e[2]['id']][0] for e in nx_edges]
        print(ec)
        options = {"node_size": 1000,
                   "alpha": 0.9,
                   "font_size": 24,
                   "edge_color": ec,
                   "width": 4,
                   "edge_cmap": plt.cm.Reds,
                   "edge_vmin": 0,
                   "edge_vmax": 1,
                   "connectionstyle":"arc3,rad=0.1"}
    nx.draw(g, pos, with_labels=True, node_color='r', **options)
    plt.show()


if __name__ == '__main__':
    """
    Only for debugging purpose
    """

