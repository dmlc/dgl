# pylint: disable=C0111, C0103, C0200
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GCNLayer, GATLayer
from ...readout import max_nodes
from ...nn.pytorch import WeightAndSum
from ...contrib.deprecation import deprecated

class MLPBinaryClassifier(nn.Module):
    """MLP for soft binary classification over multiple tasks from molecule representations.

    Parameters
    ----------
    in_feats : int
        Number of input molecular graph features
    hidden_feats : int
        Number of molecular graph features in hidden layers
    n_tasks : int
        Number of tasks, also output size
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
        super(MLPBinaryClassifier, self).__init__()

        self.predict = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feats, hidden_feats),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_feats),
            nn.Linear(hidden_feats, n_tasks)
        )

    def forward(self, h):
        """Perform soft binary classification over multiple tasks

        Parameters
        ----------
        h : FloatTensor of shape (B, M3)
            * B is the number of molecules in a batch
            * M3 is the input molecule feature size, must match in_feats in initialization

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
        """
        return self.predict(h)

class BaseGNNClassifier(nn.Module):
    """GCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.

    Parameters
    ----------
    gnn_out_feats : int
        Number of atom representation features after using GNN
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    def __init__(self, gnn_out_feats, n_tasks, classifier_hidden_feats=128, dropout=0.):
        super(BaseGNNClassifier, self).__init__()
        self.gnn_layers = nn.ModuleList()

        self.weighted_sum_readout = WeightAndSum(gnn_out_feats)
        self.g_feats = 2 * gnn_out_feats
        self.soft_classifier = MLPBinaryClassifier(
            self.g_feats, classifier_hidden_feats, n_tasks, dropout)

    def forward(self, g, feats):
        """Multi-task prediction for a batch of molecules

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, M0)
            Initial features for all atoms in the batch of molecules

        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Soft prediction for all tasks on the batch of molecules
        """
        # Update atom features with GNNs
        for gnn in self.gnn_layers:
            feats = gnn(g, feats)

        # Compute molecule features from atom features
        h_g_sum = self.weighted_sum_readout(g, feats)

        with g.local_scope():
            g.ndata['h'] = feats
            h_g_max = max_nodes(g, 'h')

        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        # Multi-task prediction
        return self.soft_classifier(h_g)

class GCNClassifier(BaseGNNClassifier):
    """GCN based predictor for multitask prediction on molecular graphs
    We assume each task requires to perform a binary classification.

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    gcn_hidden_feats : list of int
        gcn_hidden_feats[i] gives the number of output atom features
        in the i+1-th gcn layer
    n_tasks : int
        Number of prediction tasks
    classifier_hidden_feats : int
        Number of molecular graph features in hidden layers of the MLP Classifier
    dropout : float
        The probability for dropout. Default to be 0., i.e. no
        dropout is performed.
    """
    @deprecated('Import GCNPredictor from dgllife.model instead.', 'class')
    def __init__(self, in_feats, gcn_hidden_feats, n_tasks,
                 classifier_hidden_feats=128, dropout=0.):
        super(GCNClassifier, self).__init__(gnn_out_feats=gcn_hidden_feats[-1],
                                            n_tasks=n_tasks,
                                            classifier_hidden_feats=classifier_hidden_feats,
                                            dropout=dropout)

        for i in range(len(gcn_hidden_feats)):
            out_feats = gcn_hidden_feats[i]
            self.gnn_layers.append(GCNLayer(in_feats, out_feats))
            in_feats = out_feats

class GATClassifier(BaseGNNClassifier):
    """GAT based predictor for multitask prediction on molecular graphs.
    We assume each task requires to perform a binary classification.

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    """
    @deprecated('Import GATPredictor from dgllife.model instead.', 'class')
    def __init__(self, in_feats, gat_hidden_feats, num_heads,
                 n_tasks, classifier_hidden_feats=128, dropout=0):
        super(GATClassifier, self).__init__(gnn_out_feats=gat_hidden_feats[-1],
                                            n_tasks=n_tasks,
                                            classifier_hidden_feats=classifier_hidden_feats,
                                            dropout=dropout)
        assert len(gat_hidden_feats) == len(num_heads), \
            'Got gat_hidden_feats with length {:d} and num_heads with length {:d}, ' \
            'expect them to be the same.'.format(len(gat_hidden_feats), len(num_heads))
        num_layers = len(num_heads)
        for l in range(num_layers):
            if l > 0:
                in_feats = gat_hidden_feats[l - 1] * num_heads[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, gat_hidden_feats[l], num_heads[l],
                                            feat_drop=dropout, attn_drop=dropout,
                                            agg_mode=agg_mode, activation=agg_act))
