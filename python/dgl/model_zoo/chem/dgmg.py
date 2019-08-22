# pylint: disable=C0103, W0622, R1710
"""
Learning Deep Generative Models of Graphs
https://arxiv.org/pdf/1803.03324.pdf
"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Categorical

class GraphEmbed(nn.Module):
    """Compute a molecule representations out of atom representations.

    Parameters
    ----------
    node_hidden_size : int
        Size of atom representation
    """
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
        """
        Parameters
        ----------
        g : DGLGraph
            Current molecule graph

        Returns
        -------
        tensor of dtype float32 and shape (1, self.graph_hidden_size)
            Computed representation for the current molecule graph
        """
        if g.number_of_nodes() == 0:
            # Use a zero tensor for an empty molecule.
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            hvs = g.ndata['hv']
            return (self.node_gating(hvs) *
                    self.node_to_graph(hvs)).sum(0, keepdim=True)

class GraphProp(nn.Module):
    """Perform message passing over a molecule graph and update its atom representations.

    Parameters
    ----------
    num_prop_rounds : int
        Number of message passing rounds for each time
    node_hidden_size : int
        Size of atom representation
    edge_hidden_size : int
        Size of bond representation
    """
    def __init__(self, num_prop_rounds, node_hidden_size, edge_hidden_size):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        self.reduce_funcs = []
        node_update_funcs = []

        for t in range(num_prop_rounds):
            # input being [hv, hu, xuv]
            message_funcs.append(nn.Linear(2 * node_hidden_size + edge_hidden_size,
                                           self.node_activation_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size,
                           node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, send a message concat([h_u, x_uv])

        Parameters
        ----------
        edges : batch of edges

        Returns
        -------
        dict
            Dictionary containing messages for the edge batch,
            with the messages being tensors of shape (B, F1),
            B for the number of edges and F1 for the message size.
        """
        return {'m': torch.cat([edges.src['hv'],
                                edges.data['he']],
                               dim=1)}

    def dgmg_reduce(self, nodes, round):
        """Aggregate messages.

        Parameters
        ----------
        nodes : batch of nodes
        round : int
            Update round

        Returns
        -------
        dict
            Dictionary containing aggregated messages for each node
            in the batch, with the messages being tensors of shape
            (B, F2), B for the number of nodes and F2 for the aggregated
            message size
        """
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        # Make copies of original atom representations to match the
        # number of messages.
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {'a': node_activation}

    def forward(self, g):
        """
        Parameters
        ----------
        g : DGLGraph
        """
        if g.number_of_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                g.update_all(message_func=self.dgmg_msg,
                             reduce_func=self.reduce_funcs[t])
                g.ndata['hv'] = self.node_update_funcs[t](
                    g.ndata['a'], g.ndata['hv'])

class AddNode(nn.Module):
    """Stop or add an atom of a particular type.

    Parameters
    ----------
    env : MoleculeEnv
        Environment for generating molecules
    graph_embed_func : callable taking g as input
        Function for computing molecule representation
    node_hidden_size : int
        Size of atom representation
    dropout : float
        Probability for dropout
    """
    def __init__(self, env, graph_embed_func, node_hidden_size, dropout):
        super(AddNode, self).__init__()

        self.env = env
        n_node_types = len(env.atom_types)

        self.graph_op = {'embed': graph_embed_func}

        self.stop = n_node_types
        self.add_node = nn.Sequential(
            nn.Linear(graph_embed_func.graph_hidden_size, graph_embed_func.graph_hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(graph_embed_func.graph_hidden_size, n_node_types + 1)
        )

        # If to add a node, initialize its hv
        self.node_type_embed = nn.Embedding(n_node_types, node_hidden_size)
        self.initialize_hv = nn.Linear(node_hidden_size + \
                                       graph_embed_func.graph_hidden_size,
                                       node_hidden_size)

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def _initialize_node_repr(self, g, node_type, graph_embed):
        """Initialize atom representation

        Parameters
        ----------
        g : DGLGraph
        node_type : int
            Index for the type of the new atom
        graph_embed : tensor of dtype float32
            Molecule representation
        """
        num_nodes = g.number_of_nodes()
        hv_init = torch.cat([
            self.node_type_embed(torch.LongTensor([node_type])),
            graph_embed], dim=1)
        hv_init = self.dropout(hv_init)
        hv_init = self.initialize_hv(hv_init)
        g.nodes[num_nodes - 1].data['hv'] = hv_init
        g.nodes[num_nodes - 1].data['a'] = self.init_node_activation

    def prepare_log_prob(self, compute_log_prob):
        """Setup for returning log likelihood

        Parameters
        ----------
        compute_log_prob : bool
            Whether to compute log likelihood
        """
        if compute_log_prob:
            self.log_prob = []
        self.compute_log_prob = compute_log_prob

    def forward(self, action=None):
        """
        Parameters
        ----------
        action : None or int
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.

        Returns
        -------
        stop : bool
            Whether we stop adding new atoms
        """
        g = self.env.dgl_graph

        graph_embed = self.graph_op['embed'](g)

        logits = self.add_node(graph_embed).view(1, -1)
        probs = F.softmax(logits, dim=1)

        if action is None:
            action = Categorical(probs).sample().item()
        stop = bool(action == self.stop)

        if not stop:
            self.env.add_atom(action)
            self._initialize_node_repr(g, action, graph_embed)

        if self.compute_log_prob:
            sample_log_prob = F.log_softmax(logits, dim=1)[:, action: action + 1]
            self.log_prob.append(sample_log_prob)

        return stop

class AddEdge(nn.Module):
    """Stop or add a bond of a particular type.

    Parameters
    ----------
    env : MoleculeEnv
        Environment for generating molecules
    graph_embed_func : callable taking g as input
        Function for computing molecule representation
    node_hidden_size : int
        Size of atom representation
    dropout : float
        Probability for dropout
    """
    def __init__(self, env, graph_embed_func, node_hidden_size, dropout):
        super(AddEdge, self).__init__()

        self.env = env
        n_bond_types = len(env.bond_types)

        self.stop = n_bond_types

        self.graph_op = {'embed': graph_embed_func}
        self.add_edge = nn.Sequential(
            nn.Linear(graph_embed_func.graph_hidden_size + node_hidden_size,
                      graph_embed_func.graph_hidden_size + node_hidden_size),
            nn.Dropout(p=dropout),
            nn.Linear(graph_embed_func.graph_hidden_size + node_hidden_size, n_bond_types + 1)
        )

    def prepare_log_prob(self, compute_log_prob):
        """Setup for returning log likelihood

        Parameters
        ----------
        compute_log_prob : bool
            Whether to compute log likelihood
        """
        if compute_log_prob:
            self.log_prob = []
        self.compute_log_prob = compute_log_prob

    def forward(self, action=None):
        """
        Parameters
        ----------
        action : None or int
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.

        Returns
        -------
        stop : bool
            Whether we stop adding new bonds
        action : int
            The type for the new bond
        """
        g = self.env.dgl_graph

        graph_embed = self.graph_op['embed'](g)
        src_embed = g.nodes[g.number_of_nodes() - 1].data['hv']

        logits = self.add_edge(
            torch.cat([graph_embed, src_embed], dim=1))
        probs = F.softmax(logits, dim=1)

        if action is None:
            action = Categorical(probs).sample().item()
        stop = bool(action == self.stop)

        if self.compute_log_prob:
            sample_log_prob = F.log_softmax(logits, dim=1)[:, action: action + 1]
            self.log_prob.append(sample_log_prob)

        return stop, action

class ChooseDestAndUpdate(nn.Module):
    """Choose the atom to connect for the new bond.

    Parameters
    ----------
    env : MoleculeEnv
        Environment for generating molecules
    graph_prop_func : callable taking g as input
        Function for performing message passing
        and updating atom representations
    node_hidden_size : int
        Size of atom representation
    dropout : float
        Probability for dropout
    """
    def __init__(self, env, graph_prop_func, node_hidden_size, dropout):
        super(ChooseDestAndUpdate, self).__init__()

        self.env = env
        n_bond_types = len(self.env.bond_types)
        # To be used for one-hot encoding of bond type
        self.bond_embedding = torch.eye(n_bond_types)

        self.graph_op = {'prop': graph_prop_func}
        self.choose_dest = nn.Sequential(
            nn.Linear(2 * node_hidden_size + n_bond_types, 2 * node_hidden_size + n_bond_types),
            nn.Dropout(p=dropout),
            nn.Linear(2 * node_hidden_size + n_bond_types, 1)
        )

    def _initialize_edge_repr(self, g, src_list, dest_list, edge_embed):
        """Initialize bond representation

        Parameters
        ----------
        g : DGLGraph
        src_list : list of int
            source atoms for new bonds
        dest_list : list of int
            destination atoms for new bonds
        edge_embed : 2D tensor of dtype float32
            Embeddings for the new bonds
        """
        g.edges[src_list, dest_list].data['he'] = edge_embed.expand(len(src_list), -1)

    def prepare_log_prob(self, compute_log_prob):
        """Setup for returning log likelihood

        Parameters
        ----------
        compute_log_prob : bool
            Whether to compute log likelihood
        """
        if compute_log_prob:
            self.log_prob = []
        self.compute_log_prob = compute_log_prob

    def forward(self, bond_type, dest):
        """
        Parameters
        ----------
        bond_type : int
            The type for the new bond
        dest : int or None
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.
        """
        g = self.env.dgl_graph

        src = g.number_of_nodes() - 1
        possible_dests = range(src)

        src_embed_expand = g.nodes[src].data['hv'].expand(src, -1)
        possible_dests_embed = g.nodes[possible_dests].data['hv']
        edge_embed = self.bond_embedding[bond_type: bond_type + 1]

        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed,
                       src_embed_expand,
                       edge_embed.expand(src, -1)], dim=1)).view(1, -1)
        dests_probs = F.softmax(dests_scores, dim=1)

        if dest is None:
            dest = Categorical(dests_probs).sample().item()

        if not g.has_edge_between(src, dest):
            # For undirected graphs, we add edges for both directions
            # so that we can perform graph propagation.
            src_list = [src, dest]
            dest_list = [dest, src]
            self.env.add_bond(src, dest, bond_type)
            self._initialize_edge_repr(g, src_list, dest_list, edge_embed)

            # Perform message passing when new bonds are added.
            self.graph_op['prop'](g)

        if self.compute_log_prob:
            if dests_probs.nelement() > 1:
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, dest: dest + 1])

def weights_init(m):
    '''Function to initialize weights for models

    Code from https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5

    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def dgmg_message_weight_init(m):
    """Weight initialization for graph propagation module

    These are suggested by the author. This should only be used for
    the message passing functions, i.e. fe's in the paper.
    """
    def _weight_init(m):
        if isinstance(m, nn.Linear):
            init.normal_(m.weight.data, std=1./10)
            init.normal_(m.bias.data, std=1./10)
        else:
            raise ValueError('Expected the input to be of type nn.Linear!')

    if isinstance(m, nn.ModuleList):
        for layer in m:
            layer.apply(_weight_init)
    else:
        m.apply(_weight_init)

class DGMG(nn.Module):
    """DGMG model

    Users only need to initialize an instance of this class.

    Parameters
    ----------
    env : MoleculeEnv
        Environment for generating molecules
    node_hidden_size : int
        Size of atom representation
    num_prop_rounds : int
        Number of message passing rounds for each time
    dropout : float
        Probability for dropout
    """
    def __init__(self, env, node_hidden_size, num_prop_rounds, dropout):
        super(DGMG, self).__init__()

        self.env = env

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size)

        # Graph propagation module
        # For one-hot encoding, edge_hidden_size is just the number of bond types
        self.graph_prop = GraphProp(num_prop_rounds, node_hidden_size, len(self.env.bond_types))

        # Actions
        self.add_node_agent = AddNode(
            env, self.graph_embed, node_hidden_size, dropout)
        self.add_edge_agent = AddEdge(
            env, self.graph_embed, node_hidden_size, dropout)
        self.choose_dest_agent = ChooseDestAndUpdate(
            env, self.graph_prop, node_hidden_size, dropout)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        """Initialize model weights"""
        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)

        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

    @property
    def action_step(self):
        """Get current action step and increment it by 1.

        Returns
        -------
        old_step_count : int
            Current action step
        """
        old_step_count = self.step_count
        self.step_count += 1

        return old_step_count

    def prepare_log_prob(self, compute_log_prob):
        """Setup for returning log likelihood

        Parameters
        ----------
        compute_log_prob : bool
            Whether to compute log likelihood
        """
        self.compute_log_prob = compute_log_prob
        self.add_node_agent.prepare_log_prob(compute_log_prob)
        self.add_edge_agent.prepare_log_prob(compute_log_prob)
        self.choose_dest_agent.prepare_log_prob(compute_log_prob)

    def add_node_and_update(self, a=None):
        """Decide if to add a new atom.
        If a new atom should be added, update the graph.

        Parameters
        ----------
        a : None or int
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.
        """
        return self.add_node_agent(a)

    def add_edge_or_not(self, a=None):
        """Decide if to add a new bond.

        Parameters
        ----------
        a : None or int
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.
        """
        return self.add_edge_agent(a)

    def choose_dest_and_update(self, bond_type, a=None):
        """Choose destination and connect it to the latest atom.
        Add edges for both directions and update the graph.

        Parameters
        ----------
        bond_type : int
            The type of the new bond to add
        a : None or int
            If None, a new action will be sampled. If not None,
            teacher forcing will be used to enforce the decision of the
            corresponding action.
        """
        self.choose_dest_agent(bond_type, a)

    def get_log_prob(self):
        """Compute the log likelihood for the decision sequence,
        typically corresponding to the generation of a molecule.

        Returns
        -------
        torch.tensor consisting of a float only
        """
        return torch.cat(self.add_node_agent.log_prob).sum()\
               + torch.cat(self.add_edge_agent.log_prob).sum()\
               + torch.cat(self.choose_dest_agent.log_prob).sum()

    def teacher_forcing(self, actions):
        """Generate a molecule according to a sequence of actions.

        Parameters
        ----------
        actions : list of 2-tuples of int
            actions[t] gives (i, j), the action to execute by DGMG at timestep t.
            - If i = 0, j specifies either the type of the atom to add or termination
            - If i = 1, j specifies either the type of the bond to add or termination
            - If i = 2, j specifies the destination atom id for the bond to add.
              With the formulation of DGMG, j must be created before the decision.
        """
        stop_node = self.add_node_and_update(a=actions[self.action_step][1])
        while not stop_node:
            # A new atom was just added.
            stop_edge, bond_type = self.add_edge_or_not(a=actions[self.action_step][1])
            while not stop_edge:
                # A new bond is to be added.
                self.choose_dest_and_update(bond_type, a=actions[self.action_step][1])
                stop_edge, bond_type = self.add_edge_or_not(a=actions[self.action_step][1])
            stop_node = self.add_node_and_update(a=actions[self.action_step][1])

    def rollout(self):
        """Sample a molecule from the distribution learned by DGMG."""
        stop_node = self.add_node_and_update()
        while not stop_node:
            stop_edge, bond_type = self.add_edge_or_not()
            if self.env.num_atoms() == 1:
                stop_edge = True
            while not stop_edge:
                self.choose_dest_and_update(bond_type)
                stop_edge, bond_type = self.add_edge_or_not()
            stop_node = self.add_node_and_update()

    def forward(self, actions=None, rdkit_mol=False, compute_log_prob=False):
        """
        Parameters
        ----------
        actions : list of 2-tuples or None.
            If actions are not None, generate a molecule according to actions.
            Otherwise, a molecule will be generated based on sampled actions.
        rdkit_mol : bool
            Whether to maintain a Chem.rdchem.Mol object. This brings extra
            computational cost, but is necessary if we are interested in
            learning the generated molecule.
        compute_log_prob : bool
            Whether to compute log likelihood

        Returns
        -------
        torch.tensor consisting of a float only, optional
            The log likelihood for the actions taken
        str, optional
            The generated molecule in the form of SMILES
        """
        # Initialize an empty molecule
        self.env.reset(rdkit_mol=rdkit_mol)
        self.prepare_log_prob(compute_log_prob)

        if actions is not None:
            # A sequence of decisions is given, use teacher forcing
            self.step_count = 0
            self.teacher_forcing(actions)
        else:
            # Sample a molecule from the distribution learned by DGMG
            self.rollout()

        if compute_log_prob and rdkit_mol:
            return self.get_log_prob(), self.env.get_current_smiles()

        if compute_log_prob:
            return self.get_log_prob()

        if rdkit_mol:
            return self.env.get_current_smiles()
