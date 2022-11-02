import torch
import dgl
import numpy as np
import copy

# https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py#L67


# Original implementation of this func and shapley score funcs have a subgraph_building_method passed in.
# subgraph_building_method either:
#   graph_build_zero_filling : " subgraph building through masking the unselected nodes with zero features "
#   graph_build_split: " subgraph building through spliting the selected nodes from the original graph "

# To simplify, we'll only have 1 way of building subgraphs from masks. In addition, won't rely on class
# MarginalSubgraphDataset(Dataset).

def marginal_contribution(graph, exclude_masks, include_masks,
                          value_func):
    """ Calculate the marginal value for each pair. Here exclude_mask and include_mask are node mask. """

    # Seems redundant to do : (shapley func) node index list ---> turned into node mask passed into
    # marginal_contribution ---> (marginal_contribution) nod eindex list

    num_masks = np.shape(exclude_masks)[0]
    num_nodes = graph.num_nodes()

    marginal_contribution_list = []

    for i in range(num_masks):
        exclude_mask = exclude_masks[i]
        include_mask = include_masks[i]

        exclude_nodes = [j * exclude_mask[j] for j in range(num_nodes)]
        include_nodes = [j * include_mask[j] for j in range(num_nodes)]

        # exclude_subgraph = dgl.node_subgraph(graph, exclude_nodes, relabel_nodes=False)
        # include_subgraph = dgl.node_subgraph(graph, include_nodes, relabel_nodes=False)
        exclude_subgraph = dgl.node_subgraph(graph, exclude_nodes)
        include_subgraph = dgl.node_subgraph(graph, include_nodes)

        exclude_subgraph = dgl.add_self_loop(exclude_subgraph)
        include_subgraph = dgl.add_self_loop(include_subgraph)

        # WARNING: subgraph method relabels graph nodes but original labels can be recovered with parent_nid.
        # Dow we have to take this into account in the value_func methods?

        exclude_values = value_func(exclude_subgraph)
        include_values = value_func(include_subgraph)

        margin_values = include_values - exclude_values

        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def neighbors(node, graph):
  # Assume neighbors are reachable in one hop; out-edges from node
  neigh = graph.successors(node).tolist()
  return neigh


# https://github.com/divelab/DIG/blob/dig-stable/dig/xgraph/method/shapley.py

''' def mc_l_shapley(coalition: list, data: Data, local_radius: int,
                 value_func: str, subgraph_building_method='zero_filling',
                 sample_num=1000) -> float: '''


# Change subgraph from list to dl.DGLGraph? What would be gained?
def mc_l_shapley(value_func, graph, subgraph, local_radius, sample_num):
    """ monte carlo sampling approximation of the l_shapley value """
    num_nodes = graph.num_nodes()

    local_region = copy.copy(subgraph)
    for k in range(local_radius - 1):
        k_neighborhoood = []
        for node in local_region:
            # How to get neighboring nodes with dgl?
            # k_neighborhoood += list(graph.neighbors(node))
            k_neighborhoood += neighbors(node, graph)

        local_region += k_neighborhoood
        local_region = list(set(local_region))

    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for i in range(sample_num):
        subset_nodes_from = [node for node in local_region if node not in subgraph]
        random_nodes_permutation = np.array(subset_nodes_from + [coalition_placeholder])
        random_nodes_permutation = np.random.permutation(random_nodes_permutation)
        split_idx = np.where(random_nodes_permutation == coalition_placeholder)[0][0]
        selected_nodes = random_nodes_permutation[:split_idx]
        # Change to np.ones(num_nodex, dtype=np.int8) as an optimization?
        # set_exclude_mask = np.ones(num_nodes)
        set_exclude_mask = np.ones(num_nodes, dtype=np.int8)
        # set_exclude_mask[local_region] = 0.0
        # set_exclude_mask[selected_nodes] = 1.0
        set_exclude_mask[local_region] = 0
        set_exclude_mask[selected_nodes] = 1
        set_include_mask = set_exclude_mask.copy()
        # set_include_mask[subgraph] = 1.0
        set_include_mask[subgraph] = 1

        set_exclude_masks.append(set_exclude_mask)
        set_include_masks.append(set_include_mask)

    exclude_masks = np.stack(set_exclude_masks, axis=0)
    include_masks = np.stack(set_include_masks, axis=0)
    marginal_contributions = marginal_contribution(graph, exclude_masks, include_masks, value_func)
    # marginal_contribution(data, exclude_mask, include_mask, value_func, subgraph_build_func)

    mc_l_shapley_value = marginal_contributions.mean().item()
    return mc_l_shapley_value


