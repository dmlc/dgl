# pylint: disable=C0111, C0103, C0200
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GraphConv

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, activation=F.relu,
                 residual=True, batchnorm=True, dropout=0.):
        """Single layer GCN for updating node features
        Parameters
        ----------
        in_feats : int
            Number of input atom features
        out_feats : int
            Number of output atom features
        activation : activation function
            Default to be ReLU
        residual : bool
            Whether to use residual connection, default to be True
        batchnorm : bool
            Whether to use batch normalization on the output,
            default to be True
        dropout : float
            The probability for dropout. Default to be 0., i.e. no
            dropout is performed.
        """
        super(GCNLayer, self).__init__()

        self.activation = activation
        self.graph_conv = GraphConv(in_feats=in_feats, out_feats=out_feats,
                                    norm=False, activation=activation)
        self.dropout = nn.Dropout(dropout)

        self.residual = residual
        if residual:
            self.res_connection = nn.Linear(in_feats, out_feats)

        self.bn = batchnorm
        if batchnorm:
            self.bn_layer = nn.BatchNorm1d(out_feats)

    def forward(self, feats, bg):
        """Update atom representations
        Parameters
        ----------
        feats : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel
        Returns
        -------
        new_feats : FloatTensor of shape (N, M2)
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        new_feats = self.graph_conv(feats, bg)
        if self.residual:
            res_feats = self.activation(self.res_connection(feats))
            new_feats = new_feats + res_feats
        new_feats = self.dropout(new_feats)

        if self.bn:
            new_feats = self.bn_layer(new_feats)

        return new_feats

class MLPBinaryClassifier(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_tasks, dropout=0.):
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

class GCNClassifier(nn.Module):
    def __init__(self, in_feats, gcn_hidden_feats, n_tasks, classifier_hidden_feats=128,
                 dropout=0., atom_data_field='h', atom_weight_field='w'):
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
        atom_data_field : str
            Name for storing atom features in DGLGraphs
        atom_weight_field : str
            Name for storing atom weights in DGLGraphs
        """
        super(GCNClassifier, self).__init__()
        self.atom_data_field = atom_data_field

        self.gcn_layers = nn.ModuleList()
        for i in range(len(gcn_hidden_feats)):
            out_feats = gcn_hidden_feats[i]
            self.gcn_layers.append(GCNLayer(in_feats, out_feats))
            in_feats = out_feats

        self.atom_weight_field = atom_weight_field
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

        self.g_feats = 2 * in_feats
        self.soft_classifier = MLPBinaryClassifier(
            self.g_feats, classifier_hidden_feats, n_tasks, dropout)

    def forward(self, feats, bg):
        """Multi-task prediction for a batch of molecules
        Parameters
        ----------
        feats : FloatTensor of shape (N, M0)
            Initial features for all atoms in the batch of molecules
        bg : BatchedDGLGraph
            B Batched DGLGraphs for processing multiple molecules in parallel
        Returns
        -------
        FloatTensor of shape (B, n_tasks)
            Soft prediction for all tasks on the batch of molecules
        """
        # Update atom features
        for gcn in self.gcn_layers:
            feats = gcn(feats, bg)

        # Compute molecule features from atom features
        bg.ndata[self.atom_data_field] = feats
        bg.ndata[self.atom_weight_field] = self.atom_weighting(feats)
        h_g_sum = dgl.sum_nodes(
            bg, self.atom_data_field, self.atom_weight_field)
        h_g_max = dgl.max_nodes(bg, self.atom_data_field)
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)

        # Multi-task prediction
        return self.soft_classifier(h_g)
