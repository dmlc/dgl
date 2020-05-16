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

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_hidden_feats : int
        Size for the hidden node representations. Default to 500.
    num_encode_gnn_layers : int
        Number of WLN layers for updating node representations.
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

    def forward(self, reactant_graph, reactant_node_feats, reactant_edge_feats,
                product_graphs, product_node_feats, product_edge_feats,
                candidate_scores, batch_num_candidate_products):
        r"""Predicts the score for candidate products to be the true product

        Parameters
        ----------
        reactant_graph : DGLGraph
            DGLGraph for a batch of reactants.
        reactant_node_feats : float32 tensor of shape (V1, node_in_feats)
            Input node features for the reactants. V1 for the number of nodes.
        reactant_edge_feats : float32 tensor of shape (E1, edge_in_feats)
            Input edge features for the reactants. E1 for the number of edges in
            reactant_graph.
        product_graphs : DGLGraph
            DGLGraph for the candidate products in a batch of reactions.
        product_node_feats : float32 tensor of shape (V2, node_in_feats)
            Input node features for the candidate products. V2 for the number of nodes.
        product_edge_feats : float32 tensor of shape (E2, edge_in_feats)
            Input edge features for the candidate products. E2 for the number of edges
            in the graphs for candidate products.
        candidate_scores : float32 tensor of shape (B, 1)
            Scores for candidate products based on the model for reaction center prediction
        batch_num_candidate_products : list of int
            Number of candidate products for the reactions in the batch

        Returns
        -------
        float32 tensor of shape (B, 1)
            Predicted scores for candidate products
        """
        # Update representations for nodes in both reactants and candidate products
        batch_reactant_node_feats = self.gnn(
            reactant_graph, reactant_node_feats, reactant_edge_feats)
        batch_product_node_feats = self.gnn(
            product_graphs, product_node_feats, product_edge_feats)

        # Iterate over the reactions in the batch
        reactant_node_start = 0
        product_graph_start = 0
        product_node_start = 0
        batch_diff_node_feats = []

        for i, num_candidate_products in enumerate(batch_num_candidate_products):
            reactant_node_end = reactant_node_start + reactant_graph.batch_num_nodes[i]
            product_graph_end = product_graph_start + num_candidate_products
            product_node_end = product_node_start + sum(
                product_graphs.batch_num_nodes[product_graph_start: product_graph_end])

            # (N, node_out_feats)
            reactant_node_feats = batch_reactant_node_feats[reactant_node_start:
                                                            reactant_node_end, :]
            product_node_feats = batch_product_node_feats[product_node_start: product_node_end, :]

            old_feats_shape = reactant_node_feats.shape
            # (1, N, node_out_feats)
            expanded_reactant_node_feats = reactant_node_feats.reshape((1,) + old_feats_shape)
            # (B, N, node_out_feats)
            expanded_reactant_node_feats = expanded_reactant_node_feats.expand(
                (num_candidate_products,) + old_feats_shape)
            # (B, N, node_out_feats)
            candidate_product_node_feats = product_node_feats.reshape(
                (num_candidate_products,) + old_feats_shape)

            # Get the node representation difference between candidate products and reactants
            diff_node_feats = candidate_product_node_feats - expanded_reactant_node_feats
            diff_node_feats = diff_node_feats.reshape(-1, diff_node_feats.shape[-1])
            batch_diff_node_feats.append(diff_node_feats)

            reactant_node_start = reactant_node_end
            product_graph_start = product_graph_end
            product_node_start = product_node_end

        batch_diff_node_feats = torch.cat(batch_diff_node_feats, dim=0)
        # One more GNN layer for message passing with the node representation difference
        diff_node_feats = self.diff_gnn(product_graphs, batch_diff_node_feats, product_edge_feats)
        candidate_product_feats = self.readout(product_graphs, diff_node_feats)

        return self.predict(candidate_product_feats) + candidate_scores
