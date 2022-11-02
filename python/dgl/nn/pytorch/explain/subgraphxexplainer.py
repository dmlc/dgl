import math
import networkx as nx
from .shapley import *
from torch import nn
import torch.nn.functional as torch_func

__all__ = ['SubgraphXExplainer']


class MCTSNode:
    def __init__(self, subgraph, pruning_action):
        self.children = list()
        self.C = 0  # Number times this node has been visited.
        self.R = 0.0  # Immediate reward for selecting this node.
        self.W = 0.0  # total reward for all visits.
        self.subgraph = subgraph  # associated subgraph of this tree node.
        self.a = pruning_action  # merely a representation of the pruning action used to get to this node.
        self.subgraph_str = str(self.subgraph.nodes().tolist())

        # Will DGLGraph.nodes() always return a sorted list of nodes?
        # Should we instead index by the subgraph itself rather than the subgraph and pruning action that produced it?
        self.state_map_index = str(self.subgraph.nodes()) + " : " + self.a

        # self.state_map_index = str(self.subgraph.nodes().tolist()) + " , " + str(pruning_action)

    def Q(self):
        if self.C == 0:
            return 0.0
        return self.W / self.C

    def U(self, hyperparam, sum_C) -> float:
        return hyperparam * self.R * math.sqrt(sum_C) / (1 + self.C)


# WARNING: when iterating over subgraphs, may have disconnected components.
class SubgraphXExplainer(nn.Module):
    # Pruning actions: dict of {dgl.DGLGraph : set(node ids to prune/remove from subgraph)}
    # For now, leave it empty as ref impl doesn't really implement any special pruning actions.
    def __init__(self, model, hyperparam, pruning_action, num_child_expand=2, local_radius=1, sample_num=1):
        super(SubgraphXExplainer, self).__init__()

        self.hyperparam = hyperparam

        # Pruning action either "high2low" or "low2high" (refer to paper).
        self.pruning_action = pruning_action

        # Max number of children a tree node is allowed to have/expand to.
        self.num_child_expand = num_child_expand

        self.local_radius = local_radius

        self.sample_num = sample_num

        # self.state_map = {self.tree_root.state_map_index: self.tree_root}
        self.state_map = dict()

        # Change it so that score_func is one of 4 options
        self.score_func = mc_l_shapley

        self.model = model
        self.model.eval()

    def gen_value_func(self, model, features, **kwargs):
        def value_func(gr):
            with torch.no_grad():
                logits = self.model(gr, features, **kwargs)
                probs = torch_func.softmax(logits, dim=-1)
                # score = probs[:, target_class]
            # return score
                return probs

        return value_func

    # Without good pruning startegies, have to try every subgraph ---> exponential complexity.
    # Return list of subgraphs (dgl.DGLGraph) pruned from input graph.
    #
    # ". In addition, if multiple disconnected subgraphs are obtained after removing a node, the largest subgraph is kept"
    #
    # When finding components in dierected subgraphs, consider weakly connected components or stringly connected components?

    # Find weakly connected components in subgraph and return the biggest one.
    def biggest_weak_component(self, graph):
        # 1. Ensure graph is homogenous.
        new_graph = dgl.to_homogeneous(graph)
        # 2. Turn graph into a bidirected graph first
        new_graph = dgl.to_bidirected(new_graph)
        # 3. Convert to a networkx graph object.
        nx_graph = dgl.to_networkx(new_graph).to_undirected()
        # 4. Find and sort graph components by size from largest to smallest and take the biggest one.
        biggest_comp = list([c for c in sorted(nx.connected_components(nx_graph), key=len, reverse=True)][0])
        # 5. Convert back to DGLGraph object.
        return dgl.node_subgraph(graph, biggest_comp)

    def prune_graph(self, graph, strategy):
        # self.num_child_expand
        subgraphs = []
        pruned_nodes = []
        rev = False
        if strategy == "high2low":
            rev = True

        out_degrees = graph.out_degrees()
        in_degrees = graph.in_degrees()
        nodes = graph.nodes()
        node_degree = list()
        for i in range(len(out_degrees)):
            node_degree.append((nodes[i], out_degrees[i] + in_degrees[i]))
        node_degree = sorted(node_degree, key=lambda x: x[1], reverse=rev)
        if len(node_degree) < self.num_child_expand:
            pruned_nodes = nodes
        else:
            pruned_nodes = [node_degree[i][0] for i in range(self.num_child_expand)]
        for prune_node in pruned_nodes:
            new_subgraph_nodes = [node for node in nodes if node != prune_node]
            # subgraphs.append(dgl.node_subgraph(graph, new_subgraph_nodes, relabel_nodes=False))
            new_subgraph = dgl.node_subgraph(graph, new_subgraph_nodes)
            new_subgraph = self.biggest_weak_component(new_subgraph)
            subgraphs.append(new_subgraph)

        return (subgraphs, pruned_nodes)

    def explain_graph(self, graph, M, N_min, features, **kwargs):
        # self.value_func = self.gen_value_func(self.model, label, features, **kwargs)
        self.value_func = self.gen_value_func(self.model, features, **kwargs)

        # MCTS initialization
        ''' for each (N_i, a_j ) pair , initialize its C, W ,
            Q, and R variables as 0. The root of search tree is N0  
            associated with graph G. The leaf set is set to S` = {}.'''

        self.tree_root = MCTSNode(subgraph=graph, pruning_action="-1")
        self.state_map[self.tree_root.state_map_index] = self.tree_root

        leaf_set = set()

        for i in range(M):
            print("i=", i)
            curr_node = self.state_map[self.tree_root.state_map_index]
            curr_path = [curr_node]

            while curr_node.subgraph.num_nodes() > N_min:
                print("curr_node.subgraph.num_nodes() = ", curr_node.subgraph.num_nodes())
                # "for all possible pruning actions"
                # check if tree node hasn't been expanded before.
                if len(self.state_map[curr_node.state_map_index].children) == 0:
                    # "for each node, pruning it and get the remaining sub-graph..."
                    # make sure to add nodes to curr_node's children
                    (subgraphs, pruned_nodes) = self.prune_graph(curr_node.subgraph, self.pruning_action)
                    for j in range(len(subgraphs)):
                        new_child_node = MCTSNode(subgraphs[j], str(pruned_nodes[j]))
                        new_child_node.R = self.score_func(self.value_func, graph, new_child_node.subgraph.nodes().tolist(), self.local_radius, self.sample_num)
                        self.state_map[curr_node.state_map_index].children.append(new_child_node)
                        self.state_map[new_child_node.state_map_index] = new_child_node

                sum_C = 0
                for child_node in self.state_map[curr_node.state_map_index].children:
                    sum_C += child_node.C

                next_node = max(self.state_map[curr_node.state_map_index].children,
                                key=lambda x: x.Q() + x.U(self.hyperparam, sum_C))
                curr_node = next_node
                curr_path.append(next_node)

            leaf_set.add(curr_node)

            score_leaf_node = self.score_func(self.model, graph, curr_node.subgraph.nodes().tolist(),
                                              self.local_radius, self.sample_num)
            # Update nodes in curr_path
            for node in curr_node:
                self.state_map[node.state_map_index].C += 1
                self.state_map[node.state_map_index].W += score_leaf_node

        # Select subgraph with the highest score (R value) from S_l.
        best_node = max(leaf_set, key=lambda x: x.R)
        return best_node.subgraph

    def explain_node(self):
        pass