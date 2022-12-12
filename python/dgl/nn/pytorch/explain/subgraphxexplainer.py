"""Torch Module for SubgraphX"""
import math

import networkx as nx
import numpy as np
import torch
from torch import nn

import dgl

__all__ = ["SubgraphXExplainer"]


def marginal_contribution(graph, exclude_masks, include_masks, model, feat):
    r"""Calculate the marginal value for the sample coalition nodes (identified by
    include_masks).

    Parameters
    ----------
    graph: DGLGraph
        A homogeneous graph.
    exclude_masks: Tensor
        Node mask of shape :math:`(1, D)`, where :math:`D`
        is the number of nodes in the graph.
    include_masks: Tensor
        Node mask of shape :math:`(1, D)`, where :math:`D`
        is the number of nodes in the graph.
    model: nn.Module
        The GNN model to explain.
    feat: Tensor
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

        exclude_feat = feat[list(exclude_nodes_map.values())]
        include_feat = feat[list(include_nodes_map.values())]

        exclude_subgraph = dgl.add_self_loop(exclude_subgraph)
        include_subgraph = dgl.add_self_loop(include_subgraph)

        exclude_values = model(exclude_subgraph, exclude_feat)
        include_values = model(include_subgraph, include_feat)

        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def shapley(
    model, graph, subgraph_nodes, num_gnn_layers, mc_sampling_steps, feat
):
    r"""Monte carlo sampling approximation of the shapley value.

    Parameters
    ----------
    model: nn.Module
        The GNN model to explain.
    graph: DGLGraph
        A homogeneous graph.
    subgraph_nodes: tensor
        The tensor node ids of the subgraph that are associated with this tree node.
    num_gnn_layers: int
        Number of layers of GNN model. Needed to calculate the l-hop neighbouring
        nodes of graph. Computing shapley values for big and complex graphs
        is time and resource consuming. This algorithm efficiently approximate
        shapley values by finding coalition of l-hop neighboring nodes of the graph.
        The GNN model has l layers.
    mc_sampling_steps: int
        Monte carlo sampling steps.
    feat: Tensor
        The input feature of shape :math:`(N, D)`. :math:`N` is the
        number of nodes, and :math:`D` is the feature size.

    Returns
    -------
    float
        Returns the shapley value based on the subgraph nodes.
    """

    num_nodes = graph.num_nodes()

    local_region = subgraph_nodes.tolist()
    for _ in range(num_gnn_layers - 1):
        k_neighbourhood = []
        for node in local_region:
            k_neighbourhood += set(
                graph.successors(node).tolist()
                + graph.predecessors(node).tolist()
            )
        local_region += k_neighbourhood
        local_region = list(set(local_region))

    coalition = subgraph_nodes.tolist()
    coalition_placeholder = num_nodes
    set_exclude_masks = []
    set_include_masks = []
    for _ in range(mc_sampling_steps):
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
        graph, exclude_masks, include_masks, model, feat
    )

    shapley_value = marginal_contributions.mean().item()
    return shapley_value


class MCTSNode:
    r"""Monte Carlo Tree Search Node.

    Parameters
    ----------
    nodes: list
        The node ids of the graph that are associated with this tree node.
    coef: float
        The hyperparameter that controls the trade-off between exploration and exploitation.
        A higher exploration rate encourages exploration of relatively unvisited nodes.
        So, the more a node has been visited (and the more certain is its total reward value),
        the less it will be visited. For example, if a node has been visited many times and
        has a high total reward value, a high coef will cause the algorithm to explore other nodes
        instead of continuing to visit that node.
    num_visit: int
        Number times this node has been visited.
    total_reward: float
        The total reward for all node visits.
    immediate_reward: float
        Immediate reward for selecting this node (property score).
    """

    def __init__(
        self,
        nodes,
        coef=10.0,
        num_visit=0,
        total_reward=0.0,
        immediate_reward=0.0,
    ):
        self.nodes = nodes
        self.coef = coef
        self.num_visit = num_visit
        self.total_reward = total_reward
        self.immediate_reward = immediate_reward
        self.children = []

    def average_reward(self):
        r"""Get the average reward for multiple visits.

        Returns
        -------
        float
            Return the averaged reward for multiple visits.
        """
        return self.total_reward / self.num_visit if self.num_visit > 0 else 0

    def action_selection_criteria(self, total_visit_count):
        r"""Get the action selection criteria of node.

        Parameters
        ----------
        total_visit_count: float
            The total visiting counts for all possible actions of node.

        Returns
        -------
        float
            Returns the action selection criteria of node.
        """
        return (
            self.coef
            * self.immediate_reward
            * math.sqrt(total_visit_count)
            / (1 + self.num_visit)
        )


class SubgraphXExplainer(nn.Module):
    r"""SubgraphXExplainer model from `SubgraphXExplainer: On Explainability of
    Graph Neural Networks via Subgraph Explorations <https://arxiv.org/pdf/2102.05152.pdf>`.

    It identifies the most important subgraph from the original graph that plays a
    critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring different subgraphs
    for explanation and uses Shapley values as the measure of subgraph importance.

    Parameters
    ----------
    model: nn.Module
        The GNN model to explain.

        * The required arguments of its forward function are `DGLGraph` graph and its
          node features.
        * The output of its forward function is the logits for the predicted
          graph classes.
    num_gnn_layers: int
        Number of layers of GNN model. Needed to calculate the l-hop neighbouring
        nodes of graph. Computing Shapley values for big and complex graphs
        is time and resource consuming. The Shapley algorithm implemented in the paper
        efficiently approximates Shapley values by finding coalitions of l-hop neighboring
        nodes of the graph, where the GNN model has l GNN layers.
    coef: float
        The hyperparameter that controls the trade-off between exploration and exploitation.
        A higher exploration rate encourages exploration of relatively unvisited nodes.
        So, the more a node has been visited (and the more certain is its total reward value),
        the less it will be visited. For example, if a node has been visited many times and
        has a high total reward value, a high coef will cause the algorithm to explore other nodes
        instead of continuing to visit that node. Default: 10.0.
    high2low: bool, optional
        Pruning action either "High2low" or "Low2high" (refer to paper).
        "High2low": Whether to expand children nodes from high degree to low degree when
        extend the child nodes in the search tree. "Low2high" is opposite of "High2low".
        If True, it will use "High2low" pruning action, otherwise "Low2high". Default: True.
    num_child_expand: int, optional
        Max number of children a tree node is allowed to have/expand to.
    max_iter: int, optional
        Max number of iterations for MCTS.
    node_min: int, optional
        The leaf threshold node number.
    mc_sampling_steps: int, optional
        Monte carlo sampling steps.
    """

    def __init__(
        self,
        model,
        num_gnn_layers,
        coef=10.0,
        high2low=True,
        num_child_expand=2,
        max_iter=20,
        node_min=3,
        mc_sampling_steps=100,
    ):
        super(SubgraphXExplainer, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.coef = coef
        self.high2low = high2low
        self.num_child_expand = num_child_expand
        self.max_iter = max_iter
        self.node_min = node_min
        self.mc_sampling_steps = mc_sampling_steps

        self.model = model

    def biggest_weak_component(self, graph):
        r"""Find the weakly connected components in subgraph and
        return the biggest one.

        Parameters
        ----------
        graph: DGLGraph
            A homogeneous graph.

        Returns
        -------
        Function
            Returns the largest weakly connected components in graph.
        """

        # Ensure graph is homogenous.
        new_graph = dgl.to_homogeneous(graph)
        # Turn graph into a bidirected graph
        new_graph = dgl.to_bidirected(new_graph)
        # Convert to a networkx graph object
        nx_graph = dgl.to_networkx(new_graph).to_undirected()
        # Find and sort graph components by size from largest to smallest and take the biggest one.
        biggest_comp = list(
            sorted(nx.connected_components(nx_graph), key=len, reverse=True)
        )[0]
        # Convert back to DGLGraph object.
        return list(biggest_comp)

    def prune_graph(self, graph, high2low=True):
        r"""Find the graph based on the chosen strategy. Once prunes, return the
        list of subgraphs, list of subgraph nodes and list of pruned nodes.

        Parameters
        ----------
        graph: DGLGraph
            A homogeneous graph.
        high2low: bool, optional
            The strategy based on which the pruning will happen, "High2low" or "Low2high". If set
            True, then "High2low" strategy will be followed.

        Returns
        -------
        tuple(list, list, list)
            Return a tuple consists of list of subgraphs based on pruned action,
            list of subgraphs nodes, and list of pruned nodes.

        """
        subgraphs = []
        subgraphs_nodes_mapping = []

        out_degrees = graph.out_degrees()
        in_degrees = graph.in_degrees()
        nodes = graph.nodes()
        node_degree = []

        for node, in_degree, out_degree in zip(nodes, in_degrees, out_degrees):
            node_degree.append((node, in_degree + out_degree))

        node_degree = sorted(
            node_degree, key=lambda x: x[1], reverse=bool(high2low)
        )

        if len(node_degree) < self.num_child_expand:
            pruned_nodes = nodes
        else:
            pruned_nodes = [
                node_degree[i][0] for i in range(self.num_child_expand)
            ]

        for prune_node in pruned_nodes:
            new_subgraph_nodes = np.array(
                [node for node in nodes if node != prune_node]
            )
            new_subgraph = dgl.node_subgraph(graph, new_subgraph_nodes)
            biggest_comp = self.biggest_weak_component(new_subgraph)
            new_subgraph_big_comp = dgl.node_subgraph(graph, biggest_comp)

            subgraphs_nodes_mapping.append(new_subgraph_nodes[biggest_comp])
            subgraphs.append(new_subgraph_big_comp)

        return subgraphs, subgraphs_nodes_mapping, pruned_nodes

    def explain_graph(self, graph, feat, **kwargs):
        r"""Find the subgraph that plays a crucial role to explain the prediction made
        by the GNN for a graph.

        Parameters
        ----------
        graph: DGLGraph
            A homogeneous graph.
        feat: Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs: dict
            Additional arguments passed to the GNN model.

        Returns
        -------
        list
            Return list of nodes that represent the most important subgraph.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import GraphConv, SubgraphXExplainer

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_dim, hidden_dim, n_classes):
        ...         super(Model, self).__init__()
        ...         self.conv1 = GraphConv(in_dim, hidden_dim)
        ...         self.conv2 = GraphConv(hidden_dim, hidden_dim)
        ...         self.classify = nn.Linear(hidden_dim, n_classes)
        ...
        ...     def forward(self, g, h):
        ...         h = F.relu(self.conv1(g, h))
        ...         h = F.relu(self.conv2(g, h))
        ...         with g.local_scope():
        ...             g.ndata['h'] = h
        ...             hg = dgl.mean_nodes(g, 'h')
        ...             return self.classify(hg)

        >>> # Train the model
        >>> feat_size = data[0][0].ndata['attr'].shape[1]
        >>> model = Model(feat_size, 128, data.gclasses)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        >>> for bg, labels in dataloader:
        ...     logits = model(bg, bg.ndata['attr'])
        ...     loss = criterion(logits, labels)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = SubgraphXExplainer(model, num_gnn_layers=2)

        >>> # Explain the prediction for graph 0
        >>> graph, l = data[0]
        >>> graph_feat = graph.ndata.pop("attr")
        >>> g_nodes_explain = explainer.explain_graph(graph, graph_feat)
        >>> g_nodes_explain
        tensor([14, 15, 16, 17, 18, 19])
        """
        self.model.eval()

        # MCTS initialization
        tree_root = MCTSNode(nodes=graph.nodes(), coef=self.coef)

        leaf_set = set()

        for _ in range(self.max_iter):
            # print("iteration number=", i)
            curr_node = tree_root

            while len(curr_node.nodes) > self.node_min:
                # print("curr_node.nodes = ", len(curr_node.nodes))

                # check if tree node hasn't been expanded before
                if len(curr_node.children) == 0:
                    # make sure to add nodes to curr_node's children
                    subgraph = dgl.node_subgraph(graph, curr_node.nodes)
                    (
                        subgraphs,
                        subgraphs_nodes_mapping,
                        pruned_nodes,
                    ) = self.prune_graph(subgraph, self.high2low)

                    for j, _ in enumerate(subgraphs):
                        new_child_node = MCTSNode(
                            curr_node.nodes[subgraphs_nodes_mapping[j]],
                            pruned_nodes[j],
                        )
                        new_child_node.immediate_reward = shapley(
                            self.model,
                            graph,
                            new_child_node.nodes,
                            self.num_gnn_layers,
                            self.mc_sampling_steps,
                            feat,
                        )
                        curr_node.children.append(new_child_node)

                next_node = max(
                    curr_node.children,
                    key=lambda x, curr_node=curr_node: x.average_reward()
                    + x.action_selection_criteria(
                        np.sum(
                            [
                                child_node.num_visit
                                for child_node in curr_node.children
                            ]
                        )
                    ),
                )
                curr_node = next_node

            leaf_set.add(curr_node)

            score_leaf_node = shapley(
                self.model,
                graph,
                curr_node.nodes,
                self.num_gnn_layers,
                self.mc_sampling_steps,
                feat,
            )

            for node in leaf_set:
                node.num_visit += 1
                node.total_reward += score_leaf_node

        # Select subgraph with the highest score
        best_node = max(leaf_set, key=lambda x: x.immediate_reward)

        return best_node.nodes
