from inspect import signature
from math import sqrt

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch import Tensor

NID, EID = '_ID', '_ID'


def visualize_subgraph(graph: dgl.DGLGraph, node_idx: int, num_hops: int, node_alpha: Tensor = None,
                       edge_alpha: Tensor = None, seed: int = 10, **kwargs):
    r"""Visualizes the subgraph of given node alpha or given edge alpha.

    Args:
        graph (dgl.DGLGraph): The graph to visualize.
        node_idx (int): The node index to start the subgraph.
        num_hops (int): The number of hops to explore.
        node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating transparency of each node. Defaults to None.
        edge_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating transparency of each edge. Defaults to None.
        seed (int, optional): Random seed of `network.spring_layout` function. Defaults to 10.
        **kwargs: Additional arguments passed to `network.draw` function.

    Returns:
        ax (matplotlib.axes.Axes): The axes of the plot.
        nx_g (networkx.DiGraph): The graph of the subgraph.

    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """

    device = graph.device
    if node_alpha is not None:
        assert node_alpha.size(0) == graph.number_of_nodes()
        assert ((node_alpha >= 0) & (node_alpha <= 1)).all()
    else:
        node_alpha = torch.ones(graph.number_of_nodes()).to(device)

    if edge_alpha is not None:
        assert edge_alpha.size(0) == graph.number_of_edges()
        assert ((edge_alpha >= 0) & (edge_alpha <= 1)).all()
    else:
        edge_alpha = torch.ones(graph.number_of_edges()).to(device)

    # Only operate on a k-hop subgraph around `node_idx`.
    sg, _ = dgl.khop_in_subgraph(graph, node_idx, num_hops)
    # Get the node and edge indices of the subgraph.
    subnode_idx = sg.ndata[NID].long()
    subedge_idx = sg.edata[EID].long()
    # Get the node and edge attributes of the subgraph.
    edge_alpha_subset = edge_alpha.gather(0, subedge_idx)
    node_alpha_subset = node_alpha[subnode_idx]
    sg.edata['importance'] = edge_alpha_subset
    sg.ndata['importance'] = node_alpha_subset

    # Transfer the subgraph to networkx.
    nx_g = sg.cpu().to_networkx(node_attrs=['importance'], edge_attrs=['importance'])
    mapping = {k: i for k, i in enumerate(subnode_idx.tolist())}
    nx_g = nx.relabel_nodes(nx_g, mapping)

    # Initialize node and label arguments passed to `network.draw` function
    node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
    node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
    node_kwargs['node_size'] = kwargs.get('node_size') or 800
    node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

    label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
    label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
    label_kwargs['font_size'] = kwargs.get('font_size') or 10

    # Set axes and draw the subgraph.
    pos = nx.spring_layout(nx_g, seed=seed)
    ax = plt.gca()
    for source, target, data in nx_g.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->",
                alpha=max(data['importance'].item(), 0.1),
                color='black',
                shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                connectionstyle="arc3,rad=0.1",
            ))

    node_color = [0] * sg.number_of_nodes()

    nx.draw_networkx_nodes(nx_g, pos, alpha=node_alpha_subset.tolist(), node_color=node_color, **node_kwargs)
    nx.draw_networkx_labels(nx_g, pos, **label_kwargs)

    return ax, nx_g
