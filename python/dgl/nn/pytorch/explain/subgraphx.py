"""Torch Module for SubgraphX"""
import math

import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from .... import to_heterogeneous, to_homogeneous
from ....base import NID
from ....convert import to_networkx
from ....subgraph import node_subgraph
from ....transforms.functional import remove_nodes

__all__ = ["SubgraphX", "HeteroSubgraphX"]


class MCTSNode:
    r"""Monte Carlo Tree Search Node

    Parameters
    ----------
    nodes : Tensor
        The node IDs of the graph that are associated with this tree node
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        r"""Get the string representation of the node.

        Returns
        -------
        str
            The string representation of the node
        """
        return str(self.nodes)


class SubgraphX(nn.Module):
    r"""SubgraphX from `On Explainability of Graph Neural Networks via Subgraph
    Explorations <https://arxiv.org/abs/2102.05152>`

    It identifies the most important subgraph from the original graph that
    plays a critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring
    different subgraphs for explanation and uses Shapley values as the measure
    of subgraph importance.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat)`.
        * The output of its forward function is the logits.
    num_hops : int
        Number of message passing layers in the model
    coef : float, optional
        This hyperparameter controls the trade-off between exploration and
        exploitation. A higher value encourages the algorithm to explore
        relatively unvisited nodes. Default: 10.0
    high2low : bool, optional
        If True, it will use the "High2low" strategy for pruning actions,
        expanding children nodes from high degree to low degree when extending
        the children nodes in the search tree. Otherwise, it will use the
        "Low2high" strategy. Default: True
    num_child : int, optional
        This is the number of children nodes to expand when extending the
        children nodes in the search tree. Default: 12
    num_rollouts : int, optional
        This is the number of rollouts for MCTS. Default: 20
    node_min : int, optional
        This is the threshold to define a leaf node based on the number of
        nodes in a subgraph. Default: 3
    shapley_steps : int, optional
        This is the number of steps for Monte Carlo sampling in estimating
        Shapley values. Default: 100
    log : bool, optional
        If True, it will log the progress. Default: False
    """

    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        r"""Compute Shapley value with Monte Carlo approximation.

        Parameters
        ----------
        subgraph_nodes : tensor
            The tensor node ids of the subgraph that are associated with this
            tree node

        Returns
        -------
        float
            Shapley value
        """
        num_nodes = self.graph.num_nodes()
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            in_neighbors, _ = self.graph.in_edges(local_region)
            _, out_neighbors = self.graph.out_edges(local_region)
            neighbors = torch.cat([in_neighbors, out_neighbors]).tolist()
            local_region = list(set(local_region + neighbors))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [
            split_point
        ]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = torch.ones(num_nodes)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1).to(device)
            include_feat = self.feat * include_mask.unsqueeze(1).to(device)

            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_feat, **self.kwargs
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_feat, **self.kwargs
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        node_degrees = subg.out_degrees() + subg.in_degrees()
        k = min(subg.num_nodes(), self.num_child)
        chosen_nodes = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices

        mcts_children_maps = dict()

        for node in chosen_nodes:
            new_subg = remove_nodes(subg, node.to(subg.idtype), store_ids=True)
            # Get the largest weakly connected component in the subgraph.
            nx_graph = to_networkx(new_subg.cpu())
            largest_cc_nids = list(
                max(nx.weakly_connected_components(nx_graph), key=len)
            )
            # Map to the original node IDs.
            largest_cc_nids = new_subg.ndata[NID][largest_cc_nids].long()
            largest_cc_nids = subg.ndata[NID][largest_cc_nids].sort().values
            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        r"""Perform a MCTS rollout.

        Parameters
        ----------
        mcts_node : MCTSNode
            Starting node for MCTS

        Returns
        -------
        float
            Reward for visiting the node this time
        """
        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, target_class, **kwargs):
        r"""Find the most important subgraph from the original graph for the
        model to classify the graph into the target class.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph
        feat : Tensor
            The input node feature of shape :math:`(N, D)`, :math:`N` is the
            number of nodes, and :math:`D` is the feature size
        target_class : int
            The target class to explain
        kwargs : dict
            Additional arguments passed to the GNN model

        Returns
        -------
        Tensor
            Nodes that represent the most important subgraph

        Examples
        --------

        >>> import torch
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> from dgl.data import GINDataset
        >>> from dgl.dataloading import GraphDataLoader
        >>> from dgl.nn import GraphConv, AvgPooling, SubgraphX

        >>> # Define the model
        >>> class Model(nn.Module):
        ...     def __init__(self, in_dim, n_classes, hidden_dim=128):
        ...         super().__init__()
        ...         self.conv1 = GraphConv(in_dim, hidden_dim)
        ...         self.conv2 = GraphConv(hidden_dim, n_classes)
        ...         self.pool = AvgPooling()
        ...
        ...     def forward(self, g, h):
        ...         h = F.relu(self.conv1(g, h))
        ...         h = self.conv2(g, h)
        ...         return self.pool(g, h)

        >>> # Load dataset
        >>> data = GINDataset('MUTAG', self_loop=True)
        >>> dataloader = GraphDataLoader(data, batch_size=64, shuffle=True)

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
        >>> explainer = SubgraphX(model, num_hops=2)

        >>> # Explain the prediction for graph 0
        >>> graph, l = data[0]
        >>> graph_feat = graph.ndata.pop("attr")
        >>> g_nodes_explain = explainer.explain_graph(graph, graph_feat,
        ...                                           target_class=l)
        """
        self.model.eval()
        assert (
            graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root = MCTSNode(graph.nodes())
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return best_leaf.nodes


class HeteroSubgraphX(nn.Module):
    r"""SubgraphX from `On Explainability of Graph Neural Networks via Subgraph
    Explorations <https://arxiv.org/abs/2102.05152>`__, adapted for heterogeneous graphs

    It identifies the most important subgraph from the original graph that
    plays a critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring
    different subgraphs for explanation and uses Shapley values as the measure
    of subgraph importance.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat)`.
        * The output of its forward function is the logits.
    num_hops : int
        Number of message passing layers in the model
    coef : float, optional
        This hyperparameter controls the trade-off between exploration and
        exploitation. A higher value encourages the algorithm to explore
        relatively unvisited nodes. Default: 10.0
    high2low : bool, optional
        If True, it will use the "High2low" strategy for pruning actions,
        expanding children nodes from high degree to low degree when extending
        the children nodes in the search tree. Otherwise, it will use the
        "Low2high" strategy. Default: True
    num_child : int, optional
        This is the number of children nodes to expand when extending the
        children nodes in the search tree. Default: 12
    num_rollouts : int, optional
        This is the number of rollouts for MCTS. Default: 20
    node_min : int, optional
        This is the threshold to define a leaf node based on the number of
        nodes in a subgraph. Default: 3
    shapley_steps : int, optional
        This is the number of steps for Monte Carlo sampling in estimating
        Shapley values. Default: 100
    log : bool, optional
        If True, it will log the progress. Default: False
    """

    def __init__(
        self,
        model,
        num_hops,
        coef=10.0,
        high2low=True,
        num_child=12,
        num_rollouts=20,
        node_min=3,
        shapley_steps=100,
        log=False,
    ):
        super().__init__()
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log

        self.model = model

    def shapley(self, subgraph_nodes):
        r"""Compute Shapley value with Monte Carlo approximation.

        Parameters
        ----------
        subgraph_nodes : dict[str, Tensor]
            subgraph_nodes[nty] gives the tensor node IDs of node type nty
            in the subgraph, which are associated with this tree node

        Returns
        -------
        float
            Shapley value
        """
        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_regions = {
            ntype: nodes.tolist() for ntype, nodes in subgraph_nodes.items()
        }
        for _ in range(self.num_hops - 1):
            for c_etype in self.graph.canonical_etypes:
                src_ntype, _, dst_ntype = c_etype
                if (
                    src_ntype not in local_regions
                    or dst_ntype not in local_regions
                ):
                    continue

                in_neighbors, _ = self.graph.in_edges(
                    local_regions[dst_ntype], etype=c_etype
                )
                _, out_neighbors = self.graph.out_edges(
                    local_regions[src_ntype], etype=c_etype
                )
                local_regions[src_ntype] = list(
                    set(local_regions[src_ntype] + in_neighbors.tolist())
                )
                local_regions[dst_ntype] = list(
                    set(local_regions[dst_ntype] + out_neighbors.tolist())
                )

        split_point = self.graph.num_nodes()
        coalition_space = {
            ntype: list(
                set(local_regions[ntype]) - set(subgraph_nodes[ntype].tolist())
            )
            + [split_point]
            for ntype in subgraph_nodes.keys()
        }

        marginal_contributions = []
        for _ in range(self.shapley_steps):
            selected_node_map = dict()
            for ntype, nodes in coalition_space.items():
                permuted_space = np.random.permutation(nodes)
                split_idx = int(np.where(permuted_space == split_point)[0])
                selected_node_map[ntype] = permuted_space[:split_idx]

            # Mask for coalition set S_i
            exclude_mask = {
                ntype: torch.ones(self.graph.num_nodes(ntype))
                for ntype in self.graph.ntypes
            }
            for ntype, region in local_regions.items():
                exclude_mask[ntype][region] = 0.0
            for ntype, selected_nodes in selected_node_map.items():
                exclude_mask[ntype][selected_nodes] = 1.0

            # Mask for set S_i and g_i
            include_mask = {
                ntype: exclude_mask[ntype].clone()
                for ntype in self.graph.ntypes
            }
            for ntype, subgn in subgraph_nodes.items():
                exclude_mask[ntype][subgn] = 1.0

            exclude_feat = {
                ntype: self.feat[ntype]
                * exclude_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }
            include_feat = {
                ntype: self.feat[ntype]
                * include_mask[ntype].unsqueeze(1).to(self.feat[ntype].device)
                for ntype in self.graph.ntypes
            }

            with torch.no_grad():
                exclude_probs = self.model(
                    self.graph, exclude_feat, **self.kwargs
                ).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]
                include_probs = self.model(
                    self.graph, include_feat, **self.kwargs
                ).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        r"""Get the children of the MCTS node for the search.

        Parameters
        ----------
        mcts_node : MCTSNode
            Node in MCTS

        Returns
        -------
        list
            Children nodes after pruning
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        subg = node_subgraph(self.graph, mcts_node.nodes)
        # Choose k nodes based on the highest degree in the subgraph
        node_degrees_map = {
            ntype: torch.zeros(
                subg.num_nodes(ntype), device=subg.nodes(ntype).device
            )
            for ntype in subg.ntypes
        }
        for c_etype in subg.canonical_etypes:
            src_ntype, _, dst_ntype = c_etype
            node_degrees_map[src_ntype] += subg.out_degrees(etype=c_etype)
            node_degrees_map[dst_ntype] += subg.in_degrees(etype=c_etype)

        node_degrees_list = [
            ((ntype, i), degree)
            for ntype, node_degrees in node_degrees_map.items()
            for i, degree in enumerate(node_degrees)
        ]
        node_degrees = torch.stack([v for _, v in node_degrees_list])
        k = min(subg.num_nodes(), self.num_child)
        chosen_node_indicies = torch.topk(
            node_degrees, k, largest=self.high2low
        ).indices
        chosen_nodes = [node_degrees_list[i][0] for i in chosen_node_indicies]

        mcts_children_maps = dict()

        for ntype, node in chosen_nodes:
            new_subg = remove_nodes(subg, node, ntype, store_ids=True)

            if new_subg.num_edges() > 0:
                new_subg_homo = to_homogeneous(new_subg)
                # Get the largest weakly connected component in the subgraph.
                nx_graph = to_networkx(new_subg_homo.cpu())
                largest_cc_nids = list(
                    max(nx.weakly_connected_components(nx_graph), key=len)
                )
                largest_cc_homo = node_subgraph(new_subg_homo, largest_cc_nids)
                largest_cc_hetero = to_heterogeneous(
                    largest_cc_homo, new_subg.ntypes, new_subg.etypes
                )

                # Follow steps for backtracking to original graph node ids
                # 1. retrieve instanced homograph from connected-component homograph
                # 2. retrieve instanced heterograph from instanced homograph
                # 3. retrieve hetero-subgraph from instanced heterograph
                # 4. retrieve orignal graph ids from subgraph node ids
                cc_nodes = {
                    ntype: subg.ndata[NID][ntype][
                        new_subg.ndata[NID][ntype][
                            new_subg_homo.ndata[NID][
                                largest_cc_homo.ndata[NID][indicies]
                            ]
                        ]
                    ]
                    for ntype, indicies in largest_cc_hetero.ndata[NID].items()
                }
            else:
                available_ntypes = [
                    ntype
                    for ntype in new_subg.ntypes
                    if new_subg.num_nodes(ntype) > 0
                ]
                chosen_ntype = np.random.choice(available_ntypes)
                # backtrack from subgraph node ids to entire graph
                chosen_node = subg.ndata[NID][chosen_ntype][
                    np.random.choice(new_subg.nodes[chosen_ntype].data[NID])
                ]
                cc_nodes = {
                    chosen_ntype: torch.tensor(
                        [chosen_node],
                        device=subg.device,
                    )
                }

            if str(cc_nodes) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(cc_nodes)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(cc_nodes)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(
                    child_mcts_node.nodes
                )

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        r"""Perform a MCTS rollout.

        Parameters
        ----------
        mcts_node : MCTSNode
            Starting node for MCTS

        Returns
        -------
        float
            Reward for visiting the node this time
        """
        if (
            sum(len(nodes) for nodes in mcts_node.nodes.values())
            <= self.node_min
        ):
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = max(
            children_nodes,
            key=lambda c: c.total_reward / max(c.num_visit, 1)
            + self.coef
            * c.immediate_reward
            * children_visit_sum_sqrt
            / (1 + c.num_visit),
        )
        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, target_class, **kwargs):
        r"""Find the most important subgraph from the original graph for the
        model to classify the graph into the target class.

        Parameters
        ----------
        graph : DGLGraph
            A heterogeneous graph
        feat : dict[str, Tensor]
            The dictionary that associates input node features (values) with
            the respective node types (keys) present in the graph.
            The input features are of shape :math:`(N_t, D_t)`. :math:`N_t` is the
            number of nodes for node type :math:`t`, and :math:`D_t` is the feature size for
            node type :math:`t`
        target_class : int
            The target class to explain
        kwargs : dict
            Additional arguments passed to the GNN model

        Returns
        -------
        dict[str, Tensor]
            The dictionary associating tensor node ids (values) to
            node types (keys) that represents the most important subgraph

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch as th
        >>> import torch.nn as nn
        >>> import torch.nn.functional as F
        >>> from dgl.nn import HeteroSubgraphX

        >>> class Model(nn.Module):
        ...     def __init__(self, in_dim, num_classes, canonical_etypes):
        ...         super(Model, self).__init__()
        ...         self.etype_weights = nn.ModuleDict(
        ...             {
        ...                 "_".join(c_etype): nn.Linear(in_dim, num_classes)
        ...                 for c_etype in canonical_etypes
        ...             }
        ...         )
        ...
        ...     def forward(self, graph, feat):
        ...         with graph.local_scope():
        ...             c_etype_func_dict = {}
        ...             for c_etype in graph.canonical_etypes:
        ...                 src_type, etype, dst_type = c_etype
        ...                 wh = self.etype_weights["_".join(c_etype)](feat[src_type])
        ...                 graph.nodes[src_type].data[f"h_{c_etype}"] = wh
        ...                 c_etype_func_dict[c_etype] = (
        ...                     fn.copy_u(f"h_{c_etype}", "m"),
        ...                     fn.mean("m", "h"),
        ...                 )
        ...             graph.multi_update_all(c_etype_func_dict, "sum")
        ...             hg = 0
        ...             for ntype in graph.ntypes:
        ...                 if graph.num_nodes(ntype):
        ...                     hg = hg + dgl.mean_nodes(graph, "h", ntype=ntype)
        ...             return hg

        >>> input_dim = 5
        >>> num_classes = 2
        >>> g = dgl.heterograph({("user", "plays", "game"): ([0, 1, 1, 2], [0, 0, 1, 1])})
        >>> g.nodes["user"].data["h"] = th.randn(g.num_nodes("user"), input_dim)
        >>> g.nodes["game"].data["h"] = th.randn(g.num_nodes("game"), input_dim)

        >>> transform = dgl.transforms.AddReverse()
        >>> g = transform(g)

        >>> # define and train the model
        >>> model = Model(input_dim, num_classes, g.canonical_etypes)
        >>> feat = g.ndata["h"]
        >>> optimizer = th.optim.Adam(model.parameters())
        >>> for epoch in range(10):
        ...     logits = model(g, feat)
        ...     loss = F.cross_entropy(logits, th.tensor([1]))
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()

        >>> # Explain for the graph
        >>> explainer = HeteroSubgraphX(model, num_hops=1)
        >>> explainer.explain_graph(g, feat, target_class=1)
        {'game': tensor([0, 1]), 'user': tensor([1, 2])}
        """
        self.model.eval()
        assert (
            graph.num_nodes() > self.node_min
        ), f"The number of nodes in the\
            graph {graph.num_nodes()} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs

        # book all nodes in MCTS
        self.mcts_node_maps = dict()

        root_dict = {ntype: graph.nodes(ntype) for ntype in graph.ntypes}
        root = MCTSNode(root_dict)
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                print(
                    f"Rollout {i}/{self.num_rollouts}, \
                    {len(self.mcts_node_maps)} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        return best_leaf.nodes
