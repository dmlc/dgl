"""Weisfeiler-Lehman Network (WLN) for ranking candidate products"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
import torch.nn as nn

from dgl.nn.pytorch import SumPooling

from ..gnn.wln import WLN

__all__ = ['WLNReactionRanking']

# pylint: disable=W0221, E1101
class WLNReactionRanking(nn.Module):
    r"""Weisfeiler-Lehman Network (WLN) for Candidate Product Ranking

    The model is introduced in `Predicting Organic Reaction Outcomes with
    Weisfeiler-Lehman Network <https://arxiv.org/abs/1709.04555>`__ and then
    further improved in `A graph-convolutional neural network model for the
    prediction of chemical reactivity
    <https://pubs.rsc.org/en/content/articlelanding/2019/sc/c8sc04228d#!divAbstract>`__

    The model updates representations of nodes in candidate products with WLN and predicts
    the score for candidate products to be the real product.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_hidden_feats=500,
                 num_encode_gnn_layers=3):
        super(WLNReactionRanking, self).__init__()

        self.gnn = WLN(node_in_feats=node_in_feats,
                       edge_in_feats=edge_in_feats,
                       node_out_feats=node_hidden_feats,
                       n_layers=num_encode_gnn_layers,
                       set_comparison=False)
        self.diff_gnn = WLN(node_in_feats=node_hidden_feats,
                            edge_in_feats=edge_in_feats,
                            node_out_feats=node_hidden_feats,
                            n_layers=1,
                            project_in_feats=False,
                            set_comparison=False)
        self.readout = SumPooling()
        self.predict = nn.Sequential(
            nn.Linear(node_hidden_feats, node_hidden_feats),
            nn.ReLU(),
            nn.Linear(node_hidden_feats, 1)
        )

    def forward(self, batch_mol_graphs, node_feats, edge_feats, candidate_scores):
        r"""Predicts the score for candidate products to be the true product

        Parameters
        ----------
        batch_mol_graphs : DGLGraph
            A batch of B molecular graphs, where the first graph is the reactants and
            the rest graphs are candidate products.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.
        candidate_scores : float32 tensor of shape (B - 1, 1)
            Scores for candidate products based on the model for reaction center prediction
        """
        # Update representations for nodes in both reactants and candidate products
        node_feats = self.gnn(batch_mol_graphs, node_feats, edge_feats)
        num_nodes_per_graph = batch_mol_graphs.batch_num_nodes[0]
        # (N, node_out_feats)
        reactant_node_feats = node_feats[:num_nodes_per_graph, :]
        old_feats_shape = reactant_node_feats.shape
        num_candidate_products = batch_mol_graphs.batch_size - 1
        # (1, N, node_out_feats)
        expanded_reactant_node_feats = reactant_node_feats.reshape((1,) + old_feats_shape)
        # (B, N, node_out_feats)
        expanded_reactant_node_feats = expanded_reactant_node_feats.expand(
            (num_candidate_products,) + old_feats_shape)

        # (B, N, node_out_feats)
        candidate_product_node_feats = node_feats[num_nodes_per_graph:, :].reshape(
            (batch_mol_graphs.batch_size - 1,) + old_feats_shape)

        # Get the node representation difference between candidate products and reactants
        candidate_node_feats_difference = candidate_product_node_feats - \
                                          expanded_reactant_node_feats
        diff_node_feats = torch.cat([
            reactant_node_feats,
            candidate_node_feats_difference.reshape(-1, reactant_node_feats.shape[-1])], dim=0)

        # One more GNN layer for message passing with the node representation difference
        diff_node_feats = self.diff_gnn(batch_mol_graphs, diff_node_feats, edge_feats)
        graph_feats = self.readout(batch_mol_graphs, diff_node_feats)
        candidate_product_feats = graph_feats[1:, :]

        return self.predict(candidate_product_feats) + candidate_scores
