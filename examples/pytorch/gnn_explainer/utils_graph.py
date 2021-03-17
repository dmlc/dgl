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
    origin_nodes: List, list of node ids in the origin graph, sorted from small to large, whose order is the new id. e.g
               [2, 51, 53, 79] means in the new sug_graph, their new node id is [0,1,2,3], the mapping is 2<>0, 51<>1, 53<>2,
               and 79 <> 3.
    new_seed_node: Scalar, the node index of seed_nodes
    """
    seeds=seed_nodes
    for i in range(hops):
        i_hop = sample_neighbors(graph, seeds, -1)
        seeds = th.cat([seeds, i_hop.edges()[0]])
    
    ori_src, ori_dst = i_hop.edges()
    edge_all = th.cat([ori_src, ori_dst])
    origin_nodes, new_edges_all = th.unique(edge_all, return_inverse=True)

    n = int(new_edges_all.shape[0] / 2)
    new_src = new_edges_all[:n]
    new_dst = new_edges_all[n:]

    sub_graph = dgl.DGLGraph((new_src, new_dst))
    new_seed_node = th.nonzero(origin_nodes==seed_nodes, as_tuple=True)[0][0]

    return sub_graph, origin_nodes, new_seed_node


def visualize_sub_graph(sub_graph, edge_weights=None, origin_nodes=None, center_node=None):
    """
    Use networkx to visualize the sub_graph and,
    if edge weights are given, set edges with different fading of blue.

    Parameters
    ----------
    sub_graph: DGLGraph, the sub_graph to be visualized.
    edge_weights: Tensor, the same number of edges. Values are (0,1), default is None
    origin_nodes: List, list of node ids that will be used to replace the node ids in the subgraph in visualization
    center_node: Tensor, the node id in origin node list to be highlighted with different color

    Returns
    show the sub_graph
    -------

    """
    # Extract original idx and map to the new networkx graph
    # Convert to networkx graph
    g = dgl.to_networkx(sub_graph)
    nx_edges = g.edges(data=True)

    if not (origin_nodes is None):
        n_mapping = {new_id: old_id for new_id, old_id in enumerate(origin_nodes.tolist())}
        g = nx.relabel_nodes(g, mapping=n_mapping)

    pos = nx.spring_layout(g)

    if edge_weights is None:
        options = {"node_size": 1000,
                   "alpha": 0.9,
                   "font_size":24,
                   "width": 4,
                   }
    else:

        ec = [edge_weights[e[2]['id']][0] for e in nx_edges]
        options = {"node_size": 1000,
                   "alpha": 0.3,
                   "font_size": 12,
                   "edge_color": ec,
                   "width": 4,
                   "edge_cmap": plt.cm.Reds,
                   "edge_vmin": 0,
                   "edge_vmax": 1,
                   "connectionstyle":"arc3,rad=0.1"}

    nx.draw(g, pos, with_labels=True, node_color='b', **options)
    if not (center_node is None):
        nx.draw(g, pos, nodelist=center_node.tolist(), with_labels=True, node_color='r', **options)

    plt.show()
    
