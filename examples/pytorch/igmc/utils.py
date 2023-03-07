import os
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

import numpy as np
import torch as th

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x

def igmc_subgraph_extraction_labeling(graph, edge, h=1, sample_ratio=1.0, max_nodes_per_hop=200):
    """Extract the h-hop sampled enclosing subgraph around given edge with node labels in IGMC model.

    DGL induces an enclosing subgraph for a node pair ``(u, v)`` by the union of ``u`` and ``v``'s sampled 
    neighbors up to ``h`` hops. DGL assigns node with labels :math:`2 \\times i + j`, if the node appears
    at the :math:`i`-th hop, where :math:`j` is 0 if the node has the same type as node ``u``, otherwise 1.

    For more information, please see paper 
    `Inductive Matrix Completion Based on Graph Neural Networks <https://arxiv.org/abs/1904.12058>`__

    Parameters
    ----------
    graph: DGLGraph.
        The graph to extract subgraph from. Must be homogeneous graph with bipartite structure.
    edge: tuple of (int, int), ``(u, v)``
        The target edge for extracting subgraph.
    h: int, default 1
        The number of hop.
    sample_ratio: float, default 1.0
        The sampling ratio of neighbor nodes at each hop. 
    max_nodes_per_hop: int, default 200
        Max number of node to be sampled at each hop.

    Returns
    -------
    subgraph: DGLGraph
        The subgraph extracted with extra node attributes ``nlabel``.
    
    Examples
    --------
    The following example uses PyTorch backend.

    >>> u = th.tensor([0, 1, 1, 2, 2, 2])
    >>> v = th.tensor([3, 3, 4, 3, 4, 5])
    >>> g = dgl.graph((th.cat([u, v]), th.cat([v, u])))
    >>> edge = (g.edges()[0][1], g.edges()[1][1]) # edge = (tensor(1), tensor(3))
    >>> sg = igmc_subgraph_extraction_labeling(g, edge)
    >>> sg.ndata[dgl.NID], sg.ndata['nlabel']
    (tensor([1, 0, 2, 3, 4]), tensor([0, 2, 2, 1, 3]))

    """

    # 1. neighbor nodes sampling
    dist = 0
    u_nodes, v_nodes = th.as_tensor([edge[0]]), th.as_tensor([edge[1]])
    u_dist, v_dist = th.tensor([0]), th.tensor([0])
    u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
    u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

    for dist in range(1, h+1):
        # sample neigh alternately
        u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.in_edges(u_fringe)[0]
        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        v_visited = th.unique(th.cat([v_visited, v_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(v_fringe))
            v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(v_fringe):
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = th.cat([u_nodes, u_fringe])
        v_nodes = th.cat([v_nodes, v_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
        v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])

    nodes = th.cat([u_nodes, v_nodes])

    # 2. node labeling
    u_node_labels = th.stack([x*2 for x in u_dist])
    v_node_labels = th.stack([x*2+1 for x in v_dist])
    node_labels = th.cat([u_node_labels, v_node_labels])
    
    # 3. extract subgraph with sampled nodes
    subgraph = graph.subgraph(nodes)
    subgraph.ndata['nlabel'] = node_labels

    return subgraph