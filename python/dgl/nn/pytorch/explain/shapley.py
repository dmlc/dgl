import torch
import dgl
from ....base import NID, EID
import numpy as np
import copy

# https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py#L67


# Original implementation of this func and shapley score funcs have a subgraph_building_method passed in.
# subgraph_building_method either:
#   graph_build_zero_filling : " subgraph building through masking the unselected nodes with zero features "
#   graph_build_split: " subgraph building through spliting the selected nodes from the original graph "

# To simplify, we'll only have 1 way of building subgraphs from masks. In addition, won't rely on class
# MarginalSubgraphDataset(Dataset).

def marginal_contribution(graph, exclude_masks, include_masks, value_func, features):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """

    # Seems redundant to do : (shapley func) node index list ---> turned into node mask passed into
    # marginal_contribution ---> (marginal_contribution) node index list

    num_nodes = graph.num_nodes()

    marginal_contribution_list = []

    for exclude_mask, include_mask in zip(exclude_masks, include_masks):

        exclude_nodes = [j * exclude_mask[j] for j in range(num_nodes)]
        include_nodes = [j * include_mask[j] for j in range(num_nodes)]

        exclude_subgraph = dgl.node_subgraph(graph, exclude_nodes)
        include_subgraph = dgl.node_subgraph(graph, include_nodes)

        exclude_nodes_map = {i: exclude_nodes[i] for i in range(len(exclude_nodes))}
        include_nodes_map = {i: include_nodes[i] for i in range(len(include_nodes))}

        exclude_features = features[list(exclude_nodes_map.values())]
        include_features = features[list(include_nodes_map.values())]

        exclude_subgraph = dgl.add_self_loop(exclude_subgraph)
        include_subgraph = dgl.add_self_loop(include_subgraph)

        # WARNING: subgraph method relabels graph nodes but original labels can be recovered with parent_nid.
        # Do we have to take this into account in the value_func methods?
        exclude_values = value_func(exclude_subgraph, exclude_features)
        include_values = value_func(include_subgraph, include_features)

        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def neighbors(node, graph):
  # Assume neighbors are reachable in one hop; out-edges from node
  nx_graph = dgl.to_networkx(graph)
  neighbours = nx_graph.neighbors(node)

  return list(neighbours)


# https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py
# Change subgraph from list to dl.DGLGraph? What would be gained?
def mc_l_shapley(value_func, graph, subgraph_nodes, local_radius, sample_num, features):
    """ monte carlo sampling approximation of the l_shapley value """
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
        random_nodes_permutation =subset_nodes_from + [coalition_placeholder]
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
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
    marginal_contributions = marginal_contribution(graph,
                                                   exclude_masks,
                                                   include_masks,
                                                   value_func,
                                                   features)

    mc_l_shapley_value = marginal_contributions.mean().item()
    return mc_l_shapley_value


