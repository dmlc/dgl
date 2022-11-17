"""Torch Module for SubgraphX"""
import math
import networkx as nx
from torch import nn
import torch.nn.functional as F
from .shapley import *

__all__ = ["SubgraphXExplainer"]


class MCTSNode:
    r"""Monte Carlo Tree Search Node.

    Parameters
    ----------
    nodes: list
        The node ids of the graph that are associated with this tree node.
    pruning_action: str
        A representation of the pruning action used to get to this node.
    c_puct: flaot
        The hyper-parameter to encourage exploration while searching.
    W: float
        The sum of the node .
    N: int
        Number times this node has been visited.
    P: float
        Immediate reward for selecting this node (property score).
    """

    def __init__(self, nodes, pruning_action, c_puct=10.0, w=0.0, n=0, p=0.0):
        self.nodes = nodes
        self.a = pruning_action
        self.c_puct = c_puct
        self.w = w
        self.n = n
        self.p = p
        self.children = []

    def Q(self):
        return self.w / self.n if self.n > 0 else 0

    def U(self, n) -> float:
        return self.c_puct * self.p * math.sqrt(n) / (1 + self.n)


class SubgraphXExplainer(nn.Module):
    r"""SubgraphXExplainer model from `SubgraphXExplainer: On Explainability of
    Graph Neural Networks via Subgraph Explorations <https://arxiv.org/pdf/2102.05152.pdf>`

    It identifies subgraphs from the original graph  that play a
    critical role in GNN-based graph classification.

    To generate an explanation, by efficiently exploring different subgraphs with
    Monte Carlo tree search. They  use Shapley values as a measure of subgraph importance

    Parameters
    ----------
    model: nn.Module
        The GNN model to explain.
    hyperparam: float
        The hyperparameter that encourages exploration.
    pruning_action: str
        Pruning action either "high2low" or "low2high" (refer to paper).
    num_child_expand: int
        Max number of children a tree node is allowed to have/expand to.
    local_radius: int
        Number of local radius to calculate :obj:`l_shapley`.
    sample_num: int
        Sampling time of monte carlo sampling approximation for
        :obj:`mc_shapley`.
    """

    def __init__(
        self,
        model,
        hyperparam,
        pruning_action,
        num_child_expand=2,
        local_radius=4,
        sample_num=100,
    ):
        super(SubgraphXExplainer, self).__init__()

        self.hyperparam = hyperparam
        self.pruning_action = pruning_action
        self.num_child_expand = num_child_expand
        self.local_radius = local_radius
        self.sample_num = sample_num

        self.score_func = mc_l_shapley

        self.model = model
        self.model.eval()

    def get_value_func(self, features, **kwargs):
        r"""Get the value function.

        Parameters
        ----------
        features : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        Function
            Returns the value function
        """

        def value_func(graph):
            r"""Get the model predictions.

            Parameters
            ----------
            graph : DGLGraph
                A homogeneous graph.

            Returns
            -------
            Tensor
                Returns a tensor of probabilities
            """
            with torch.no_grad():
                logits = self.model(graph, features, **kwargs)
                probs = F.softmax(logits, dim=-1)
                return probs

        return value_func

    def biggest_weak_component(self, graph):
        r"""Find the weakly connected components in subgraph and
        return the biggest one.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.

        Returns
        -------
        Function
            Returns the largest weakly connected components in graph
        """

        # Ensure graph is homogenous.
        new_graph = dgl.to_homogeneous(graph)
        # Turn graph into a bidirected graph
        new_graph = dgl.to_bidirected(new_graph)
        # Convert to a networkx graph object
        nx_graph = dgl.to_networkx(new_graph).to_undirected()
        # Find and sort graph components by size from largest to smallest and take the biggest one.
        biggest_comp = list(sorted(nx.connected_components(nx_graph), key=len, reverse=True))
        # Convert back to DGLGraph object.
        return biggest_comp

    def prune_graph(self, graph, strategy):
        r"""Find the graph based on the chosen strategy. Once prunes, return the
        lsit of subgraphs, list of subgraph nodes and list of pruned nodes

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        strategy: str
            The strategy based on which the pruning will happen, "high2low" or "low2high"

        Returns
        -------
        tuple(list, list, list)
            Return a tuple consists of list of subgraphs based on pruned action,
            list of subgraphs nodes, and list of pruned nodes

        """
        subgraphs = []
        subgraphs_nodes_mapping = []

        rev = False
        if strategy == "high2low":
            rev = True

        out_degrees = graph.out_degrees()
        in_degrees = graph.in_degrees()
        nodes = graph.nodes()
        node_degree = []

        for i, _ in enumerate(out_degrees):
            node_degree.append((nodes[i], out_degrees[i] + in_degrees[i]))

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

    def explain_graph(self, graph, m, n_min, features, **kwargs):
        r"""Find the subgraph that play a crucial role to explain the prediction made
        by the GNN for a graph.

        Parameters
        ----------
        graph : DGLGraph
            A homogeneous graph.
        m: int
            Max number of iteration for MCTS
        n_min: int
            The leaf threshold node number
        features : Tensor
            The input feature of shape :math:`(N, D)`. :math:`N` is the
            number of nodes, and :math:`D` is the feature size.
        kwargs : dict
            Additional arguments passed to the GNN model. Tensors whose
            first dimension is the number of nodes or edges will be
            assumed to be node/edge features.

        Returns
        -------
        list
            Return list of nodes that represent the most important subgraph

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

        >>> # Explain the prediction for graph 0
        >>> explainer = SubgraphXExplainer(model, hyperparam=6, pruning_action="high2low")
        >>> g_nodes_explain = explainer.explain_graph(graph, m=50, n_min=6, features=features)
        >>> g_nodes_explain
        tensor([10, 11, 12, 13, 14])
        """
        self.value_func = self.get_value_func(features, **kwargs)

        # MCTS initialization
        self.tree_root = MCTSNode(
            nodes=graph.nodes(), pruning_action="-1", c_puct=self.hyperparam
        )

        leaf_set = set()

        for _ in range(m):
            # print("iteration number=", i)
            curr_node = self.tree_root
            curr_path = [curr_node]

            while len(curr_node.nodes) > n_min:
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

                    for j in range(len(subgraphs)):
                        new_child_node = MCTSNode(
                            curr_node.nodes[subgraphs_nodes_mapping[j]],
                            str(pruned_nodes[j]),
                        )
                        new_child_node.R = self.score_func(
                            self.model,
                            graph,
                            new_child_node.nodes,
                            self.local_radius,
                            self.sample_num,
                            features,
                        )
                        curr_node.children.append(new_child_node)

                sum_n = 0
                for child_node in curr_node.children:
                    sum_n += child_node.n

                next_node = max(
                    curr_node.children, key=lambda x: x.Q() + x.U(sum_n)
                )
                curr_node = next_node
                curr_path.append(next_node)

            leaf_set.add(curr_node)

            score_leaf_node = self.score_func(
                self.model,
                graph,
                curr_node.nodes,
                self.local_radius,
                self.sample_num,
                features,
            )

            for node in leaf_set:
                node.n += 1
                node.w += score_leaf_node

        # Select subgraph with the highest score (P value)
        best_node = max(leaf_set, key=lambda x: x.p)

        return best_node.nodes
