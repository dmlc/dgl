import torch
import dgl
import numpy as np


def neighbors(node, graph):
    r"""Find the one-hop neighbours of a node for a given graph.

    Parameters
    ----------
    node: int
        The node to get the neighbours.
    graph : DGLGraph
        A homogeneous graph.

    Returns
    -------
    Function
        Returns the largest weakly connected components in graph
    """
    nx_graph = dgl.to_networkx(graph.cpu())
    neighbours = nx_graph.neighbors(node)

    return list(neighbours)


def marginal_contribution(
    graph, exclude_masks, include_masks, value_func, features
):
    r"""Calculate the marginal value for the sample coalition nodes (identified by
    inluded_masks).

    Parameters
    ----------
    graph : DGLGraph
        A homogeneous graph.
    exclude_masks : Tensor
        Node mask of shape :math:`(1, D)`, where :math:`D`
        is the number of nodes in the graph.
    include_masks: Tensor
        Node mask of shape :math:`(1, D)`, where :math:`D`
        is the number of nodes in the graph.
    value_func : function
        T The value function that will be used to get the prediction.
    features : Tensor
        The input feature of shape :math:`(N, D)`. :math:`N` is the
        number of nodes, and :math:`D` is the feature size.

    Returns
    -------
    list of Tensor
        Returns the marginal contribution for each node index that are in the coalition
    """
    num_nodes = graph.num_nodes()

    marginal_contribution_list = []

    for exclude_mask, include_mask in zip(exclude_masks, include_masks):
        exclude_nodes = [j * exclude_mask[j] for j in range(num_nodes)]
        include_nodes = [j * include_mask[j] for j in range(num_nodes)]

        exclude_subgraph = dgl.node_subgraph(graph, exclude_nodes)
        include_subgraph = dgl.node_subgraph(graph, include_nodes)

        exclude_nodes_map = {
            i: exclude_nodes[i] for i in range(len(exclude_nodes))
        }
        include_nodes_map = {
            i: include_nodes[i] for i in range(len(include_nodes))
        }

        exclude_features = features[list(exclude_nodes_map.values())]
        include_features = features[list(include_nodes_map.values())]

        exclude_subgraph = dgl.add_self_loop(exclude_subgraph)
        include_subgraph = dgl.add_self_loop(include_subgraph)

        exclude_values = value_func(exclude_subgraph, exclude_features)
        include_values = value_func(include_subgraph, include_features)

        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def mc_l_shapley(
    value_func, graph, subgraph_nodes, local_radius, sample_num, features
):
    r"""Monte carlo sampling approximation of the l_shapley value.

    Parameters
    ----------
    value_func: function
        The value function that will be used to get the prediction.
    graph: DGLGraph
        A homogeneous graph.
    subgraph_nodes: list
        The node ids of the subgraph that are associated with this tree node.
    local_radius: int
        Number of local radius to calculate :obj:`l_shapley`.
    sample_num: int
        Sampling time of monte carlo sampling approximation for
        :obj:`mc_shapley`.
    features : Tensor
        The input feature of shape :math:`(N, D)`. :math:`N` is the
        number of nodes, and :math:`D` is the feature size.

    Returns
    -------
    float
        Returns the mc_l_shapley value based on the subgraph nodes
    """

    num_nodes = graph.num_nodes()

    local_region = subgraph_nodes.tolist()
    for k in range(local_radius - 1):
        k_neighbourhood = []
        for node in local_region:
            k_neighbourhood += neighbors(node, graph)
        local_region += k_neighbourhood
        local_region = list(set(local_region))

    coalition = subgraph_nodes.tolist()
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for i in range(sample_num):
        subset_nodes_from = list(set(local_region) - set(subgraph_nodes))
        random_nodes_permutation = subset_nodes_from + [coalition_placeholder]
        random_nodes_permutation = np.random.permutation(
            random_nodes_permutation
        )
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[
            0
        ][0]

        selected_nodes = random_nodes_permutation[:split_idx]

        set_exclude_mask = np.ones(num_nodes, dtype=np.int8)
        set_exclude_mask[local_region] = 0.0
        set_exclude_mask[selected_nodes] = 1.0
        set_include_mask = set_exclude_mask.copy()
        set_include_mask[coalition] = 1.0

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_masks = np.stack(set_exclude_masks, axis=0)
    include_masks = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(
        graph, exclude_masks, include_masks, value_func, features
    )

    mc_l_shapley_value = marginal_contributions.mean().item()
    return mc_l_shapley_value
