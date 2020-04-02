"""Weisfeiler-Lehman Network (WLN) for Reaction Center Prediction."""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl.function as fn
import torch
import torch.nn as nn

from ..gnn.wln import WLNLinear, WLN

__all__ = ['WLNReactionCenter']

# pylint: disable=W0221, E1101
class WLNContext(nn.Module):
    """Attention-based context computation for each node.

    A context vector is computed by taking a weighted sum of node representations,
    with weights computed from an attention module.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_pair_in_feats : int
        Size for the input features of node pairs.
    """
    def __init__(self, node_in_feats, node_pair_in_feats):
        super(WLNContext, self).__init__()

        self.project_feature_sum = WLNLinear(node_in_feats, node_in_feats, bias=False)
        self.project_node_pair_feature = WLNLinear(node_pair_in_feats, node_in_feats)
        self.compute_attention = nn.Sequential(
            nn.ReLU(),
            WLNLinear(node_in_feats, 1),
            nn.Sigmoid()
        )

    def forward(self, batch_complete_graphs, node_feats, feat_sum, node_pair_feat):
        """Compute context vectors for each node.

        Parameters
        ----------
        batch_complete_graphs : DGLGraph
            A batch of fully connected graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        feat_sum : float32 tensor of shape (E_full, node_in_feats)
            Sum of node_feats between each pair of nodes. E_full for the number of
            edges in the batch of complete graphs.
        node_pair_feat : float32 tensor of shape (E_full, node_pair_in_feats)
            Input features for each pair of nodes. E_full for the number of edges in
            the batch of complete graphs.

        Returns
        -------
        node_contexts : float32 tensor of shape (V, node_in_feats)
            Context vectors for nodes.
        """
        with batch_complete_graphs.local_scope():
            batch_complete_graphs.ndata['hv'] = node_feats
            batch_complete_graphs.edata['a'] = self.compute_attention(
                self.project_feature_sum(feat_sum) + \
                self.project_node_pair_feature(node_pair_feat)
            )
            batch_complete_graphs.update_all(
                fn.src_mul_edge('hv', 'a', 'm'), fn.sum('m', 'context'))
            node_contexts = batch_complete_graphs.ndata.pop('context')

        return node_contexts

class WLNReactionCenter(nn.Module):
    r"""Weisfeiler-Lehman Network (WLN) for Reaction Center Prediction.

    The model is introduced in `Predicting Organic Reaction Outcomes with
    Weisfeiler-Lehman Network <https://arxiv.org/abs/1709.04555>`__.

    The model uses WLN to update atom representations and then predicts the
    score for each pair of atoms to form a bond.

    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    edge_in_feats : int
        Size for the input edge features.
    node_out_feats : int
        Size for the output node representations. Default to 300.
    node_pair_in_feats : int
        Size for the input features of node pairs.
    n_layers : int
        Number of times for message passing. Note that same parameters
        are shared across n_layers message passing. Default to 3.
    n_tasks : int
        Number of tasks for prediction.
    """
    def __init__(self,
                 node_in_feats,
                 edge_in_feats,
                 node_pair_in_feats,
                 node_out_feats=300,
                 n_layers=3,
                 n_tasks=5):
        super(WLNReactionCenter, self).__init__()

        self.gnn = WLN(node_in_feats=node_in_feats,
                       edge_in_feats=edge_in_feats,
                       node_out_feats=node_out_feats,
                       n_layers=n_layers)
        self.context_module = WLNContext(node_in_feats=node_out_feats,
                                         node_pair_in_feats=node_pair_in_feats)
        self.project_feature_sum = WLNLinear(node_out_feats, node_out_feats, bias=False)
        self.project_node_pair_feature = WLNLinear(node_pair_in_feats, node_out_feats, bias=False)
        self.project_context_sum = WLNLinear(node_out_feats, node_out_feats)
        self.predict = nn.Sequential(
            nn.ReLU(),
            WLNLinear(node_out_feats, n_tasks)
        )

    def forward(self, batch_mol_graphs, batch_complete_graphs,
                node_feats, edge_feats, node_pair_feats):
        r"""Predict score for each pair of nodes.

        Parameters
        ----------
        batch_mol_graphs : DGLGraph
            A batch of molecular graphs.
        batch_complete_graphs : DGLGraph
            A batch of fully connected graphs.
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges.
        node_pair_feats : float32 tensor of shape (E_full, node_pair_in_feats)
            Input features for each pair of nodes. E_full for the number of edges in
            the batch of complete graphs.

        Returns
        -------
        scores : float32 tensor of shape (E_full, 5)
            Predicted scores for each pair of atoms to perform one of the following
            5 actions in reaction:

            * The bond between them gets broken
            * Forming a single bond
            * Forming a double bond
            * Forming a triple bond
            * Forming an aromatic bond
        biased_scores : float32 tensor of shape (E_full, 5)
            Comparing to scores, a bias is added if the pair is for a same atom.
        """
        node_feats = self.gnn(batch_mol_graphs, node_feats, edge_feats)

        # Compute context vectors for all atoms, which are weighted sum of atom
        # representations in all reactants.
        with batch_complete_graphs.local_scope():
            batch_complete_graphs.ndata['hv'] = node_feats
            batch_complete_graphs.apply_edges(fn.u_add_v('hv', 'hv', 'feature_sum'))
            feat_sum = batch_complete_graphs.edata.pop('feature_sum')
        node_contexts = self.context_module(batch_complete_graphs, node_feats,
                                            feat_sum, node_pair_feats)

        # Predict score
        with batch_complete_graphs.local_scope():
            batch_complete_graphs.ndata['context'] = node_contexts
            batch_complete_graphs.apply_edges(fn.u_add_v('context', 'context', 'context_sum'))
            scores = self.predict(
                self.project_feature_sum(feat_sum) + \
                self.project_node_pair_feature(node_pair_feats) + \
                self.project_context_sum(batch_complete_graphs.edata['context_sum'])
            )

        # Masking self loops
        nodes = batch_complete_graphs.nodes()
        e_ids = batch_complete_graphs.edge_ids(nodes, nodes)
        bias = torch.zeros(scores.shape[0], 5).to(scores.device)
        bias[e_ids, :] = 1e4
        biased_scores = scores - bias

        return scores, biased_scores
