import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.distributions import Bernoulli, Categorical


class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.node_gating = nn.Sequential(
            nn.Linear(node_hidden_size, 1),
            nn.Sigmoid()
        )
        self.node_to_graph = nn.Linear(node_hidden_size,
                                       self.graph_hidden_size)

    def forward(self, g_list):
        # With our current batched implementation of DGMG, new nodes
        # are not added for any graph until all graphs are done with
        # adding edges starting from the last node. Therefore all graphs
        # in the graph_list should have the same number of nodes.
        if g_list[0].number_of_nodes() == 0:
            return torch.zeros(len(g_list), self.graph_hidden_size)

        bg = dgl.batch(g_list)
        bhv = bg.ndata['hv']
        bg.ndata['hg'] = self.node_gating(bhv) * self.node_to_graph(bhv)

        return dgl.sum_nodes(bg, 'hg')


class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        node_update_funcs = []
        self.reduce_funcs = []

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(nn.Linear(2 * node_hidden_size + 1,
                                           self.node_activation_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size,
                           node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """
        For an edge u->v, return concat([h_u, x_uv])
        """
        return {'m': torch.cat([edges.src['hv'],
                                edges.data['he']],
                               dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {'a': node_activation}

    def forward(self, g_list):
        # Merge small graphs into a large graph.
        bg = dgl.batch(g_list)

        if bg.number_of_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                bg.update_all(message_func=self.dgmg_msg,
                              reduce_func=self.reduce_funcs[t])
                bg.ndata['hv'] = self.node_update_funcs[t](
                    bg.ndata['a'], bg.ndata['hv'])

        return dgl.unbatch(bg)


def bernoulli_action_log_prob(logit, action):
    """
    Calculate the log p of an action with respect to a Bernoulli
    distribution across a batch of actions. Use logit rather than
    prob for numerical stability.
    """
    log_probs = torch.cat([F.logsigmoid(-logit), F.logsigmoid(logit)], dim=1)
    return log_probs.gather(1, torch.tensor(action).unsqueeze(1))


class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddNode, self).__init__()

        self.graph_op = {'embed': graph_embed_func}

        self.stop = 1
        self.add_node = nn.Linear(graph_embed_func.graph_hidden_size, 1)

        # If to add a node, initialize its hv
        self.node_type_embed = nn.Embedding(1, node_hidden_size)
        self.initialize_hv = nn.Linear(node_hidden_size + \
                                       graph_embed_func.graph_hidden_size,
                                       node_hidden_size)

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)

    def _initialize_node_repr(self, g, node_type, graph_embed):
        num_nodes = g.number_of_nodes()
        hv_init = self.initialize_hv(
            torch.cat([
                self.node_type_embed(torch.LongTensor([node_type])),
                graph_embed], dim=1))
        g.nodes[num_nodes - 1].data['hv'] = hv_init
        g.nodes[num_nodes - 1].data['a'] = self.init_node_activation

    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for AddNode actions.
        Each element is a tensor of shape [batch_size, 1].
        """
        self.log_prob = []

    def forward(self, g_list, a=None):
        """
        Decide if a new node should be added for each graph in
        the `g_list`. If a new node is added, initialize its
        node representations. Record graphs for which a new node
        is added.

        During training, the action is passed rather than made
        and the log P of the action is recorded.

        During inference, the action is sampled from a Bernoulli
        distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of dgl.DGLGraph objects
        a : None or list
            - During training, a is a list of integers specifying
              whether a new node should be added.
            - During inference, a is None.

        Returns
        -------
        g_non_stop : list
            list of indices to specify which graphs in the
            g_list have a new node added
        """

        # Graphs for which a node is added
        g_non_stop = []

        batch_graph_embed = self.graph_op['embed'](g_list)
        batch_logit = self.add_node(batch_graph_embed)
        batch_prob = torch.sigmoid(batch_logit)

        if not self.training:
            a = Bernoulli(batch_prob).sample().squeeze(1).tolist()

        for i, g in enumerate(g_list):
            action = a[i]
            stop = bool(action == self.stop)

            if not stop:
                g_non_stop.append(g.index)
                g.add_nodes(1)
                self._initialize_node_repr(g, action,
                                           batch_graph_embed[i:i+1, :])

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(batch_logit, a)
            self.log_prob.append(sample_log_prob)

        return g_non_stop


class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddEdge, self).__init__()

        self.graph_op = {'embed': graph_embed_func}
        self.add_edge = nn.Linear(graph_embed_func.graph_hidden_size + \
                                  node_hidden_size, 1)

    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for AddEdge actions.
        Each element is a tensor of shape [batch_size, 1].
        """
        self.log_prob = []

    def forward(self, g_list, a=None):
        """
        Decide if a new edge should be added for each graph in
        the `g_list`. Record graphs for which a new edge is to
        be added.

        During training, the action is passed rather than made
        and the log P of the action is recorded.

        During inference, the action is sampled from a Bernoulli
        distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of dgl.DGLGraph objects
        a : None or list
            - During training, a is a list of integers specifying
              whether a new edge should be added.
            - During inference, a is None.

        Returns
        -------
        g_to_add_edge : list
            list of indices to specify which graphs in the
            g_list need a new edge to be added
        """

        # Graphs for which an edge is to be added.
        g_to_add_edge = []

        batch_graph_embed = self.graph_op['embed'](g_list)
        batch_src_embed = torch.cat([g.nodes[g.number_of_nodes() - 1].data['hv']
                                     for g in g_list], dim=0)
        batch_logit = self.add_edge(torch.cat([batch_graph_embed,
                                               batch_src_embed], dim=1))
        batch_prob = torch.sigmoid(batch_logit)

        if not self.training:
            a = Bernoulli(batch_prob).sample().squeeze(1).tolist()

        for i, g in enumerate(g_list):
            action = a[i]

            if action == 0:
                g_to_add_edge.append(g.index)

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(batch_logit, a)
            self.log_prob.append(sample_log_prob)

        return g_to_add_edge


class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size):
        super(ChooseDestAndUpdate, self).__init__()

        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data['he'] = edge_repr

    def prepare_training(self):
        """
        This function will only be called during training.
        It stores all log probabilities for ChooseDest actions.
        Each element is a tensor of shape [1, 1].
        """
        self.log_prob = []

    def forward(self, g_list, d=None):
        """
        For each g in g_list, add an edge (src, dest)
        if (src, dst) does not exist. The src is just the latest
        node in g. Initialize edge features if new edges are added.

        During training, dst is passed rather than chosen and the
        log P of the action is recorded.

        During inference, dst is sampled from a Categorical
        distribution modeled.

        Parameters
        ----------
        g_list : list
            A list of dgl.DGLGraph objects
        d : None or list
            - During training, d is a list of integers specifying dst for
              each graph in g_list.
            - During inference, d is None.
        """

        for i, g in enumerate(g_list):
            src = g.number_of_nodes() - 1
            possible_dests = range(src)

            src_embed_expand = g.nodes[src].data['hv'].expand(src, -1)
            possible_dests_embed = g.nodes[possible_dests].data['hv']

            dests_scores = self.choose_dest(
                torch.cat([possible_dests_embed,
                           src_embed_expand], dim=1)).view(1, -1)
            dests_probs = F.softmax(dests_scores, dim=1)

            if not self.training:
                dest = Categorical(dests_probs).sample().item()
            else:
                dest = d[i]

            # Note that we are not considering multigraph here.
            if not g.has_edge_between(src, dest):
                # For undirected graphs, we add edges for both
                # directions so that we can perform graph propagation.
                src_list = [src, dest]
                dest_list = [dest, src]

                g.add_edges(src_list, dest_list)
                self._initialize_edge_repr(g, src_list, dest_list)

            if self.training:
                if dests_probs.nelement() > 1:
                    self.log_prob.append(
                        F.log_softmax(dests_scores, dim=1)[:, dest: dest + 1])

class DGMG(nn.Module):
    def __init__(self, v_max, node_hidden_size,
                 num_prop_rounds):
        super(DGMG, self).__init__()

        # Graph configuration
        self.v_max = v_max

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        self.graph_prop = GraphProp(num_prop_rounds,
                                    node_hidden_size)

        # Actions
        self.add_node_agent = AddNode(
            self.graph_embed, node_hidden_size)
        self.add_edge_agent = AddEdge(
            self.graph_embed, node_hidden_size)
        self.choose_dest_agent = ChooseDestAndUpdate(
            self.graph_prop, node_hidden_size)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        from utils import weights_init, dgmg_message_weight_init

        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)

        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

    def prepare(self, batch_size):
        # Track how many actions have been taken for each graph.
        self.step_count = [0] * batch_size
        self.g_list = []
        # indices for graphs being generated
        self.g_active = list(range(batch_size))

        for i in range(batch_size):
            g = dgl.DGLGraph()
            g.index = i

            # If there are some features for nodes and edges,
            # zero tensors will be set for those of new nodes and edges.
            g.set_n_initializer(dgl.frame.zero_initializer)
            g.set_e_initializer(dgl.frame.zero_initializer)

            self.g_list.append(g)

        if self.training:
            self.add_node_agent.prepare_training()
            self.add_edge_agent.prepare_training()
            self.choose_dest_agent.prepare_training()

    def _get_graphs(self, indices):
        return [self.g_list[i] for i in indices]

    def get_action_step(self, indices):
        """
        This function should only be called during training.

        Collect the number of actions taken for each graph
        whose index is in the indices. After collecting
        the number of actions, increment it by 1.
        """

        old_step_count = []

        for i in indices:
            old_step_count.append(self.step_count[i])
            self.step_count[i] += 1

        return old_step_count

    def get_actions(self, mode):
        """
        This function should only be called during training.

        Decide which graphs are related with the next batched
        decision and extract the actions to take for each of
        the graph.
        """

        if mode == 'node':
            # Graphs being generated
            indices = self.g_active
        elif mode == 'edge':
            # Graphs having more edges to be added
            # starting from the latest node.
            indices = self.g_to_add_edge
        else:
            raise ValueError("Expected mode to be in ['node', 'edge'], "
                             "got {}".format(mode))

        action_indices = self.get_action_step(indices)
        # Actions for all graphs indexed by indices at timestep t
        actions_t = []

        for i, j in enumerate(indices):
            actions_t.append(self.actions[j][action_indices[i]])

        return actions_t

    def add_node_and_update(self, a=None):
        """
        Decide if to add a new node for each graph being generated.
        If a new node should be added, update the graph.

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """
        g_list = self._get_graphs(self.g_active)
        g_non_stop = self.add_node_agent(g_list, a)

        self.g_active = g_non_stop
        # For all newly added nodes we need to decide
        # if an edge is to be added for each of them.
        self.g_to_add_edge = g_non_stop

        return len(self.g_active) == 0

    def add_edge_or_not(self, a=None):
        """
        Decide if a new edge should be added for each
        graph that may need one more edge.

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """
        g_list = self._get_graphs(self.g_to_add_edge)
        g_to_add_edge = self.add_edge_agent(g_list, a)
        self.g_to_add_edge = g_to_add_edge

        return len(self.g_to_add_edge) > 0

    def choose_dest_and_update(self, a=None):
        """
        For each graph that requires one more edge, choose
        destination and connect it to the latest node.
        Add edges for both directions and update the graph.

        The action(s) a are passed during training and
        sampled (hence None) during inference.
        """
        g_list = self._get_graphs(self.g_to_add_edge)
        self.choose_dest_agent(g_list, a)

        # Graph propagation and update node features.
        updated_g_list = self.graph_prop(g_list)

        for i, g in enumerate(updated_g_list):
            g.index = self.g_to_add_edge[i]
            self.g_list[g.index] = g

    def get_log_prob(self):
        return torch.cat(self.add_node_agent.log_prob).sum()\
               + torch.cat(self.add_edge_agent.log_prob).sum()\
               + torch.cat(self.choose_dest_agent.log_prob).sum()

    def forward_train(self, actions):
        """
        Go through all decisions in actions and record their
        log probabilities for calculating the loss.

        Parameters
        ----------
        actions : list
            list of decisions extracted for generating a graph using DGMG

        Returns
        -------
        tensor of shape torch.Size([])
            log P(Generate a batch of graphs using DGMG)
        """
        self.actions = actions

        stop = self.add_node_and_update(a=self.get_actions('node'))

        # Some graphs haven't been completely generated.
        while not stop:
            to_add_edge = self.add_edge_or_not(a=self.get_actions('edge'))

            # Some graphs need more edges to be added for the latest node.
            while to_add_edge:
                self.choose_dest_and_update(a=self.get_actions('edge'))
                to_add_edge = self.add_edge_or_not(a=self.get_actions('edge'))
            stop = self.add_node_and_update(a=self.get_actions('node'))

        return self.get_log_prob()

    def forward_inference(self):
        """
        Generate graph(s) on the fly.

        Returns
        -------
        self.g_list : list
            A list of dgl.DGLGraph objects.
        """
        stop = self.add_node_and_update()

        # Some graphs haven't been completely generated and their numbers of
        # nodes do not exceed the limit of self.v_max.
        while (not stop) and (self.g_list[self.g_active[0]].number_of_nodes()
                              < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not()

            # Some graphs need more edges to be added for the latest node and
            # the number of trials does not exceed the number of maximum possible
            # edges. Note that this limit on the number of edges eliminate the
            # possibility of multi-graph and one may want to remove it.
            while to_add_edge and (num_trials <
                                   self.g_list[self.g_active[0]].number_of_nodes() - 1):
                self.choose_dest_and_update()
                num_trials += 1
                to_add_edge = self.add_edge_or_not()
            stop = self.add_node_and_update()

        return self.g_list

    def forward(self, batch_size=1, actions=None):
        if self.training:
            batch_size = len(actions)
        self.prepare(batch_size)

        if self.training:
            return self.forward_train(actions)
        else:
            return self.forward_inference()
