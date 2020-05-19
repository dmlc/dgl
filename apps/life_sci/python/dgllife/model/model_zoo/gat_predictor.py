"""GAT-based model for regression and classification on graphs."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch.nn as nn

from .mlp_predictor import MLPPredictor
from ..gnn.gat import GAT
from ..readout.weighted_sum_and_max import WeightedSumAndMax

# pylint: disable=W0221
class GATPredictor(nn.Module):
    r"""GAT-based model for regression and classification on graphs.

    GAT is introduced in `Graph Attention Networks <https://arxiv.org/abs/1710.10903>`__.
    This model is based on GAT and can be used for regression and classification on graphs.

    After updating node representations, we perform a weighted sum with learnable
    weights and max pooling on them and concatenate the output of the two operations,
    which is then fed into an MLP for final prediction.

    For classification tasks, the output will be logits, i.e.
    values before sigmoid or softmax.

    Parameters
    ----------
    in_feats : int
        Number of input node features
    hidden_feats : list of int
        ``hidden_feats[i]`` gives the output size of an attention head in the i-th GAT layer.
        ``len(hidden_feats)`` equals the number of GAT layers. By default, we use ``[32, 32]``.
    num_heads : list of int
        ``num_heads[i]`` gives the number of attention heads in the i-th GAT layer.
        ``len(num_heads)`` equals the number of GAT layers. By default, we use 4 attention heads
        for each GAT layer.
    feat_drops : list of float
        ``feat_drops[i]`` gives the dropout applied to the input features in the i-th GAT layer.
        ``len(feat_drops)`` equals the number of GAT layers. By default, this will be zero for
        all GAT layers.
    attn_drops : list of float
        ``attn_drops[i]`` gives the dropout applied to attention values of edges in the i-th GAT
        layer. ``len(attn_drops)`` equals the number of GAT layers. By default, this will be zero
        for all GAT layers.
    alphas : list of float
        Hyperparameters in LeakyReLU, which are the slopes for negative values. ``alphas[i]``
        gives the slope for negative value in the i-th GAT layer. ``len(alphas)`` equals the
        number of GAT layers. By default, this will be 0.2 for all GAT layers.
    residuals : list of bool
        ``residual[i]`` decides if residual connection is to be used for the i-th GAT layer.
        ``len(residual)`` equals the number of GAT layers. By default, residual connection
        is performed for each GAT layer.
    agg_modes : list of str
        The way to aggregate multi-head attention results for each GAT layer, which can be either
        'flatten' for concatenating all-head results or 'mean' for averaging all-head results.
        ``agg_modes[i]`` gives the way to aggregate multi-head attention results for the i-th
        GAT layer. ``len(agg_modes)`` equals the number of GAT layers. By default, we flatten
        multi-head results for intermediate GAT layers and compute mean of multi-head results
        for the last GAT layer.
    activations : list of activation function or None
        ``activations[i]`` gives the activation function applied to the aggregated multi-head
        results for the i-th GAT layer. ``len(activations)`` equals the number of GAT layers.
        By default, ELU is applied for intermediate GAT layers and no activation is applied
        for the last GAT layer.
    classifier_hidden_feats : int
        Size of hidden graph representations in the classifier. Default to 128.
    classifier_dropout : float
        The probability for dropout in the classifier. Default to 0.
    n_tasks : int
        Number of tasks, which is also the output size. Default to 1.
    """
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None,
                 classifier_hidden_feats=128, classifier_dropout=0., n_tasks=1):
        super(GATPredictor, self).__init__()

        self.gnn = GAT(in_feats=in_feats,
                       hidden_feats=hidden_feats,
                       num_heads=num_heads,
                       feat_drops=feat_drops,
                       attn_drops=attn_drops,
                       alphas=alphas,
                       residuals=residuals,
                       agg_modes=agg_modes,
                       activations=activations)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, classifier_hidden_feats,
                                    n_tasks, classifier_dropout)

    def forward(self, bg, feats):
        """Graph-level regression/soft classification.

        Parameters
        ----------
        bg : DGLGraph
            DGLGraph for a batch of graphs.
        feats : FloatTensor of shape (N, M1)
            * N is the total number of nodes in the batch of graphs
            * M1 is the input node feature size, which must match
              in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            * Predictions on graphs
            * B for the number of graphs in the batch
        """
        node_feats = self.gnn(bg, feats)
        graph_feats = self.readout(bg, node_feats)
        return self.predict(graph_feats)
