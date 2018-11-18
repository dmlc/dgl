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

    def forward(self, g):
        if g.number_of_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            hvs = g.ndata['hv']
            return (self.node_gating(hvs) *
                    self.node_to_graph(hvs)).sum(0, keepdim=True)


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

            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size,
                           node_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([h_u, x_uv])"""
        return {'m': torch.cat([edges.src['hv'],
                                edges.data['he']],
                               dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {'hv': self.node_update_funcs[round](node_activation, hv_old)}

    def forward(self, g):
        if g.number_of_nodes() == 0:
            return

        for t in range(self.num_prop_rounds):
            if g.number_of_edges() > 0:
                g.update_all(message_func=self.dgmg_msg,
                             reduce_func=self.reduce_funcs[t])


# Define two functions for calculating p and log p with
# samples from Bernoulli distributions
def bernoulli_action_prob(prob, action):
    return 1. - prob if action == 0 else prob

def bernoulli_action_log_prob(logit, action):
    # Use logit rather than prob for numerical stability
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)


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

    def _initialize_node_repr(self, g, node_type, graph_embed):
        num_nodes = g.number_of_nodes()
        hv_init = self.initialize_hv(
            torch.cat([
                self.node_type_embed(torch.LongTensor([node_type])),
                graph_embed], dim=1))
        if num_nodes == 1:
            g.ndata['hv'] = hv_init
        else:
            g.nodes[num_nodes - 1].data['hv'] = hv_init

    def prepare_training(self):
        self.prob = []
        self.log_prob = []

    def forward(self, g, **kwargs):
        graph_embed = self.graph_op['embed'](g)

        logit = self.add_node(graph_embed)
        prob = torch.sigmoid(logit)

        if self.training:
            action = kwargs.get('a')
        else:
            action = Bernoulli(prob).sample().item()
        stop = bool(action == self.stop)

        if not stop:
            g.add_nodes(1)
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            sample_prob = bernoulli_action_prob(prob, action)
            sample_log_prob = bernoulli_action_log_prob(logit, action)

            self.prob.append(sample_prob.detach())
            self.log_prob.append(sample_log_prob)

        return stop


class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddEdge, self).__init__()

        self.graph_op = {'embed': graph_embed_func}
        self.add_edge = nn.Linear(graph_embed_func.graph_hidden_size + \
                                  node_hidden_size, 1)

    def prepare_training(self):
        self.prob = []
        self.log_prob = []

    def forward(self, g, **kwargs):
        graph_embed = self.graph_op['embed'](g)
        src_embed = g.nodes[g.number_of_nodes() - 1].data['hv']

        logit = self.add_edge(torch.cat(
            [graph_embed, src_embed], dim=1))
        prob = torch.sigmoid(logit)

        if self.training:
            action = kwargs.get('a')
        else:
            action = Bernoulli(prob).sample().item()
        to_add_edge = bool(action == 0)

        if self.training:
            sample_prob = bernoulli_action_prob(prob, action)
            sample_log_prob = bernoulli_action_log_prob(logit, action)

            self.prob.append(sample_prob.detach())
            self.log_prob.append(sample_log_prob)

        return to_add_edge


class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {'prop': graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)

        if g.number_of_edges() - len(src_list) == 0:
            g.edata['he'] = edge_repr
        else:
            g.edges[src_list, dest_list].data['he'] = edge_repr

    def prepare_training(self):
        self.prob = []
        self.log_prob = []

    def forward(self, g, **kwargs):
        src = g.number_of_nodes() - 1
        possible_dests = range(src)

        src_embed_expand = g.nodes[src].data['hv'].expand(src, -1)
        possible_dests_embed = g.nodes[possible_dests].data['hv']

        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed,
                       src_embed_expand], dim=1)).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)

        if self.training:
            dest = kwargs.get('a')
        else:
            dest = Categorical(dests_probs).sample().item()

        if not g.has_edge_between(src, dest):
            # For undirected graphs, we add edges for both directions
            # so that we can perform graph propagation.
            src_list = [src, dest]
            dest_list = [dest, src]

            g.add_edges(src_list, dest_list)
            self._initialize_edge_repr(g, src_list, dest_list)

            self.graph_op['prop'](g)

        if self.training:
            if dests_probs.nelement() > 1:
                self.prob.append(dests_probs[:, dest: dest + 1])
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

    @property
    def graph_size(self):
        return self.g.number_of_nodes()

    @property
    def action_step(self):
        old_step_count = self.step_count
        self.step_count += 1

        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0

        self.add_node_agent.prepare_training()
        self.add_edge_agent.prepare_training()
        self.choose_dest_agent.prepare_training()

    def add_node_and_update(self, **kwargs):
        """Decide if to add a new node.
        If a new node should be added, update the graph."""

        return self.add_node_agent(self.g, **kwargs)

    def add_edge_or_not(self, **kwargs):
        """Decide if a new edge should be added."""

        return self.add_edge_agent(self.g, **kwargs)

    def choose_dest_and_update(self, **kwargs):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""

        self.choose_dest_agent(self.g, **kwargs)

    def get_log_prob(self):
        return torch.cat(self.add_node_agent.log_prob).sum()\
               + torch.cat(self.add_edge_agent.log_prob).sum()\
               + torch.cat(self.choose_dest_agent.log_prob).sum()

    def get_prob(self):
        return torch.cat(self.add_node_agent.prob).prod()\
               * torch.cat(self.add_edge_agent.prob).prod()\
               * torch.cat(self.choose_dest_agent.prob).prod()

    def forward_train(self, actions):
        self.prepare_for_train()

        stop = self.add_node_and_update(a=actions[self.action_step])

        while (not stop) and (self.graph_size < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not(a=actions[self.action_step])
            while to_add_edge and (num_trials
                                   < self.graph_size - 1):
                self.choose_dest_and_update(a=actions[self.action_step])
                num_trials += 1
                to_add_edge = self.add_edge_or_not(a=actions[self.action_step])
            stop = self.add_node_and_update(a=actions[self.action_step])

        return self.get_log_prob(), self.get_prob()

    def forward_inference(self):
        stop = self.add_node_and_update()
        while (not stop) and (self.graph_size < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not()
            while to_add_edge and (num_trials
                                   < self.graph_size - 1):
                self.choose_dest_and_update()
                num_trials += 1
                to_add_edge = self.add_edge_or_not()
            stop = self.add_node_and_update()

        return self.g

    def forward(self, **kwargs):
        # The graph we will work on
        self.g = dgl.DGLGraph()

        # If there are some features for nodes and edges,
        # zero tensors will be set for those of new nodes and edges.
        self.g.set_n_initializer(lambda shape, dtype, ctx:
                                 torch.zeros(shape, device=ctx))
        self.g.set_e_initializer(lambda shape, dtype, ctx:
                                 torch.zeros(shape, device=ctx))

        if self.training:
            return self.forward_train(kwargs.get('actions'))
        else:
            return self.forward_inference()
