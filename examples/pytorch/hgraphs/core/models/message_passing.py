"""
Defines models for propagating information between nodes
within and between different levels of the hierarchical graph.

Abbreviations:
msg := message
hgraph := hierarchical graph
"""

import torch, dgl

class HierMessagePassingNet(torch.nn.Module):
    def __init__(self, rep_size, rep_activation, neighbor_hops, interlevel_dropout,
                 mpn_model, **mpn_model_kwargs):
        """
        Args:
        rep_size -
        rep_activation -
        neighbor_hops -
        interlevel_dropout - Dropout to use when passing embeddings up the hierarchy.
        mpn_model - Model used for passing message into and within a level of hgraph.
        mpn_model_kwargs - Passed into mpn_model.
        """
        super().__init__()

        self.levels = [
            ("atom", "bond", "atom"),
            ("attachment_config", "ancestry", "attachment_config"),
            ("motif", "ancestry", "motif")
        ]
        
        self.msg_passing_nets = torch.nn.ModuleList(
            mpn_model(**mpn_model_kwargs,
                      rep_size = rep_size,
                      rep_activation = rep_activation,
                      neighbor_hops = neighbor_hops,
                      interlevel_dropout = interlevel_dropout)
            for level in self.levels
        )

    def forward(self, hgraph):
        for i, (level, msg_passing_net) in enumerate(zip(self.levels, self.msg_passing_nets)):
            prev_level = self.levels[i - 1] if i != 0 else None
            msg_passing_net(hgraph, level, prev_level)

class MessagePassingNet(torch.nn.Module):
    def __init__(self, rep_size, rep_activation, neighbor_hops, interlevel_dropout):
        super().__init__()
        
        self.make_interlevel_msgs = dgl.function.copy_src("rep", "prev_level_rep")
        self.reduce_interlevel_msgs = dgl.function.sum("prev_level_rep", "sum_prev_level_reps")
        self.update_nodes_interlevel_msgs = NodeUpdater(rep_size, rep_size, rep_activation, interlevel_dropout)

        self.rep_size = rep_size
        self.rep_activation = rep_activation
        self.neighbor_hops = neighbor_hops
        self.interlevel_dropout = interlevel_dropout
        
    def forward(self, hgraph, level, prev_level = None):
        raise NotImplementedError()

    def interlevel_forward(self, hgraph, node_type, prev_level):
        """
        Propagate information from nodes in the last level to adjacent nodes
        in this level.
        """

        if prev_level is not None:
            if prev_level[0] != prev_level[2]:
                raise TypeError("The source and destination nodes in the previous level should have the same type!")
            prev_level_node_type = prev_level[0]
            interlevel_edge_type = (prev_level_node_type, "of", node_type)
            
            hgraph.update_all(self.make_interlevel_msgs, 
                              self.reduce_interlevel_msgs,
                              etype = interlevel_edge_type)
            
            sum_prev_level_reps = hgraph.nodes[node_type].data["sum_prev_level_reps"]
            self.update_nodes_interlevel_msgs(hgraph, node_type, sum_prev_level_reps)

class GCN(MessagePassingNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gcns = torch.nn.ModuleList([
            dgl.nn.GraphConv(self.rep_size, self.rep_size, "both",
                             True, True, torch.nn.functional.relu)
            for i in range(self.neighbor_hops)
        ])

    def forward(self, hgraph, level, prev_level = None):
        if level[0] != level[2]:
            raise TypeError("The source and destination nodes in one level should have the same type!")
        node_type, edge_type = level[0], level

        self.interlevel_forward(hgraph, node_type, prev_level)
        self.intralevel_forward(hgraph, node_type, edge_type)
        
        return hgraph
    
    def intralevel_forward(self, hgraph, node_type, edge_type):
        level_graph = hgraph.edge_type_subgraph([edge_type])
        level_graph = dgl.add_self_loop(level_graph)
        for gcn in self.gcns:
            hgraph.nodes[node_type].data["rep"] = gcn(level_graph, hgraph.nodes[node_type].data["rep"])

class RNN_MPN(MessagePassingNet):
    def __init__(self, hidden_size, dropout = 0.0, rnn_model = torch.nn.LSTM, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.make_intralevel_msgs = RNNMessageMaker(self.rep_size, hidden_size, dropout, rnn_model)
        self.reduce_intralevel_msgs = dgl.function.sum("hidden_and_cell", "sum_incoming_hidden_and_cell")
        self.update_nodes_intralevel_msgs = NodeUpdater(hidden_size, self.rep_size, self.rep_activation, dropout)

        self.hidden_size = hidden_size
        self.dropout = dropout
        
    def forward(self, hgraph, level, prev_level = None):
        if level[0] != level[2]:
            raise TypeError("The source and destination nodes in one level should have the same type!")
        node_type, edge_type = level[0], level
        hgraph.nodes[node_type].data["sum_incoming_hidden_and_cell"] = (
            torch.randn(hgraph.number_of_nodes(node_type), 2, self.hidden_size, device = hgraph.device)
        )

        self.interlevel_forward(hgraph, node_type, prev_level)

        #Running empty graph features through RNN in MPN breaks CUDA.
        if hgraph.number_of_edges(edge_type) > 0:
          self.intralevel_forward(hgraph, node_type, edge_type)
        
        return hgraph

    def intralevel_forward(self, hgraph, node_type, edge_type):
        for hop in range(self.neighbor_hops):
            hgraph.update_all(self.make_intralevel_msgs, self.reduce_intralevel_msgs, etype = edge_type)

        msg_hiddens = hgraph.nodes[node_type].data["sum_incoming_hidden_and_cell"][:, 0]
        self.update_nodes_intralevel_msgs(hgraph, node_type, msg_hiddens)

class RNNMessageMaker(torch.nn.Module):
    def __init__(self, rep_size, hidden_size, dropout, rnn_model = torch.nn.LSTM):
        super().__init__()
        
        node_rep_size = edge_rep_size = rep_size

        self.rnn = rnn_model(node_rep_size + edge_rep_size, hidden_size, dropout = dropout)

    def forward(self, edges):
        node_reps, edge_reps = edges.src["rep"], edges.data["rep"]
        inputs = torch.cat([node_reps, edge_reps], 1)
        inputs = inputs.unsqueeze(0) #Since we're processing only 1 seq element at a time.
        rnn_state = edges.src["sum_incoming_hidden_and_cell"]
        hiddens, cells = rnn_state[:, 0].contiguous(), rnn_state[:, 1].contiguous()
        hiddens, cells = hiddens.unsqueeze(0), cells.unsqueeze(0) #Since it's only 1 layer+direction.
        outputs, (updated_hiddens, updated_cells) = self.rnn(inputs, (hiddens, cells))
        updated_hiddens, updated_cells = updated_hiddens.squeeze(0), updated_cells.squeeze(0)
        updated_rnn_state = torch.stack([updated_hiddens, updated_cells], 1)
        new_features = { "hidden_and_cell": updated_rnn_state }

        return new_features

class NodeUpdater(torch.nn.Module):
    """
    Update node representations by passing their current representations
    concatenated with a vector of additional info to an MLP and using the
    output as the new node representation.
    """

    def __init__(self, add_info_size, rep_size, rep_activation, dropout):
        super().__init__()

        concat_size = rep_size + add_info_size
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(concat_size, rep_size),
            rep_activation(),
            torch.nn.Dropout(dropout),
        )

    def forward(self, hgraph, node_type, add_info):
        node_reps = hgraph.nodes[node_type].data["rep"]
        concat = torch.cat([node_reps, add_info], 1)
        node_reps = self.MLP(concat)
        hgraph.nodes[node_type].data["rep"] = node_reps

        return hgraph
