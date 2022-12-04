"""Torch Module for SubgraphX"""
import math
import torch
import numpy as np
from torch import nn
import networkx as nx
import dgl

__all__ = ["SubgraphXExplainer"]


def marginal_contribution(graph, exclude_masks, include_masks, model, features):
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
    features: Tensor
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

        exclude_values = model(exclude_subgraph, exclude_features)
        include_values = model(include_subgraph, include_features)

        margin_values = include_values - exclude_values
        marginal_contribution_list.append(margin_values)

    marginal_contributions = torch.cat(marginal_contribution_list, dim=0)
    return marginal_contributions


def shapley(
    model, graph, subgraph_nodes, num_gnn_layers, mc_sampling_steps, features
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
    features: Tensor
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
        graph, exclude_masks, include_masks, model, features
    )

    shapley_value = marginal_contributions.mean().item()
    return shapley_value


class MCTSNode:
    r"""Monte Carlo Tree Search Node.

    Parameters
    ----------
    nodes: list
        The node ids of the graph that are associated with this tree node.
    pruning_action: str
        A representation of the pruning action used to get to this node.
    coef: float
        The hyperparameter that encourages exploration, so high number encourages
        exploration. It is high for moves with few simulations.
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
        pruning_action,
        coef=10.0,
        num_visit=0,
        total_reward=0.0,
        immediate_reward=0.0,
    ):
        self.nodes = nodes
        self.pruning_action = pruning_action
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

    It identifies the most important subgraph from the original graph that play a
    critical role in GNN-based graph classification.

    It employs Monte Carlo tree search in efficiently exploring different subgraphs for explanation
    and uses Shapley values as the measure of subgraph importance.

    Parameters
    ----------
    model: nn.Module
        The GNN model to explain.
    coef: float
        The hyperparameter that encourages exploration, so high number encourages
        exploration. It is high for moves with few simulations.
    pruning_action: str
        Pruning action either "High2low" or "Low2high" (refer to paper).
        "High2low": Whether to expand children nodes from high degree to low degree when
        extend the child nodes in the search tree. "Low2high" is opposite of "High2low".
    num_child_expand: int
        Max number of children a tree node is allowed to have/expand to.
    max_iter: int
            Max number of iteration for MCTS.
    node_min: int
        The leaf threshold node number.
    """

    def __init__(
        self,
        model,
        coef=10.0,
        pruning_action="High2low",
        num_child_expand=2,
        max_iter=20,
        node_min=3,
    ):
        super(SubgraphXExplainer, self).__init__()

        self.coef = coef
        self.pruning_action = pruning_action
        self.num_child_expand = num_child_expand
        self.max_iter = max_iter
        self.node_min = node_min

        self.score_func = shapley

        self.model = model
        self.model.eval()

    def get_value_func(self, features, **kwargs):
        r"""Get the value function.

        Parameters
        ----------
        features: Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs: dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        Function
            Returns the value function.
        """

        def value_func(graph):
            r"""Get the model predictions.

            Parameters
            ----------
            graph: DGLGraph
                A homogeneous graph.

            Returns
            -------
            Tensor
                Returns a tensor of probabilities.
            """
            with torch.no_grad():
                logits = self.model(graph, features, **kwargs)
                probs = nn.functional.softmax(logits, dim=-1)
                return probs

        return value_func

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

    def prune_graph(self, graph, strategy):
        r"""Find the graph based on the chosen strategy. Once prunes, return the
        list of subgraphs, list of subgraph nodes and list of pruned nodes.

        Parameters
        ----------
        graph: DGLGraph
            A homogeneous graph.
        strategy: str
            The strategy based on which the pruning will happen, "High2low" or "Low2high".

        Returns
        -------
        tuple(list, list, list)
            Return a tuple consists of list of subgraphs based on pruned action,
            list of subgraphs nodes, and list of pruned nodes.

        """
        subgraphs = []
        subgraphs_nodes_mapping = []

        rev = False
        if strategy == "High2low":
            rev = True

        out_degrees = graph.out_degrees()
        in_degrees = graph.in_degrees()
        nodes = graph.nodes()
        node_degree = []

        for node, in_degree, out_degree in zip(nodes, in_degrees, out_degrees):
            node_degree.append((node, in_degree + out_degree))

        node_degree = sorted(node_degree, key=lambda x: x[1], reverse=rev)

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

    def explain_graph(
        self, graph, features, num_gnn_layers=4, mc_sampling_steps=100, **kwargs
    ):
        r"""Find the subgraph that play a crucial role to explain the prediction made
        by the GNN for a graph.

        Parameters
        ----------
        graph: DGLGraph
            A homogeneous graph.
        features: Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        num_gnn_layers: int
            Number of layers of GNN model. Needed to calculate the l-hop neighbouring
            nodes of graph. Computing shapley values for big and complex graphs
            is time and resource consuming. This algorithm efficiently approximate
            shapley values by finding coalition of l-hop neighboring nodes of the graph.
            The GNN model has l layers.
        mc_sampling_steps: int
            Monte carlo sampling steps.
        kwargs: dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        list
            Return list of nodes that represent the most important subgraph.

        Examples
        --------

        >>> import dgl.function as fn
        >>> import torch
        >>> import torch.nn as nn
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import AvgPooling, SubgraphXExplainer

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

        >>> # Define a model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_feats, out_feats):
        ...         super(Model, self).__init__()
        ...         self.linear = nn.Linear(in_feats, out_feats)
        ...         self.pool = AvgPooling()
        ...
        ...     def forward(self, graph, feat, eweight=None):
        ...         with graph.local_scope():
        ...             feat = self.linear(feat)
        ...             graph.ndata['h'] = feat
        ...             if eweight is None:
        ...                 graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        ...             else:
        ...                 graph.edata['w'] = eweight
        ...                 graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
        ...             return self.pool(graph, graph.ndata['h'])

        >>> # Train the model
        >>> feat_size = data[0][0].ndata['attr'].shape[1]
        >>> model = Model(feat_size, data.gclasses)
        >>> criterion = nn.CrossEntropyLoss()
        >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        >>> for bg, labels in dataloader:
        ...     logits = model(bg, bg.ndata['attr'])
        ...     loss = criterion(logits, labels)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Initialize the explainer
        >>> explainer = SubgraphXExplainer(model, coef=6, pruning_action="High2low",
        ...     max_iter=50, node_min=6)

        >>> # Explain the prediction for graph 0
        >>> graph, l = data[0]
        >>> graph_feat = graph.ndata.pop("attr")
        >>> g_nodes_explain = explainer.explain_graph(graph, features=graph_feat)
        >>> g_nodes_explain
        tensor([14, 15, 16, 17, 18, 19])
        """
        self.value_func = self.get_value_func(features, **kwargs)

        # MCTS initialization
        self.tree_root = MCTSNode(
            nodes=graph.nodes(), pruning_action="-1", coef=self.coef
        )

        leaf_set = set()

        for _ in range(self.max_iter):
            # print("iteration number=", i)
            curr_node = self.tree_root
            curr_path = [curr_node]

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
                    ) = self.prune_graph(subgraph, self.pruning_action)

                    for j, _ in enumerate(subgraphs):
                        new_child_node = MCTSNode(
                            curr_node.nodes[subgraphs_nodes_mapping[j]],
                            str(pruned_nodes[j]),
                        )
                        new_child_node.R = self.score_func(
                            self.model,
                            graph,
                            new_child_node.nodes,
                            num_gnn_layers,
                            mc_sampling_steps,
                            features,
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
                curr_path.append(next_node)

            leaf_set.add(curr_node)

            score_leaf_node = self.score_func(
                self.model,
                graph,
                curr_node.nodes,
                num_gnn_layers,
                mc_sampling_steps,
                features,
            )

            for node in leaf_set:
                node.num_visit += 1
                node.total_reward += score_leaf_node

        # Select subgraph with the highest score
        best_node = max(leaf_set, key=lambda x: x.immediate_reward)

        return best_node.nodes
