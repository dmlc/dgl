# pylint: disable=C0103, E1101
"""GNN layers for updating atom representations"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import GraphConv, edge_softmax

class GCNLayer(nn.Module):
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
    def __init__(self, in_feats, out_feats, activation=F.relu,
                 residual=True, batchnorm=True, dropout=0.):
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

class GATConv(nn.Module):
    """
    Apply multi-head graph attention over an input signal.

    # Todo (Zihao): make this class part of nn module.

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features for each attention head
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, slope for negative values
    residual : bool
        Whether to perform skip connection, default to be False
    activation : activation function
        Activation function applied to multi-head results, default to be None
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha, residual=False, activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop) if feat_drop else lambda x: x
        self.attn_drop = nn.Dropout(attn_drop) if attn_drop else lambda x: x
        self.leaky_relu = nn.LeakyReLU(alpha)
        self._residual = residual
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.register_buffer('res_fc', None)

        self.activation = activation
        self._reset_parameters()

    def _reset_parameters(self):
        """Parameter initialization"""
        nn.init.xavier_normal_(self.fc.weight, gain=1.414)
        nn.init.xavier_normal_(self.attn_l, gain=1.414)
        nn.init.xavier_normal_(self.attn_r, gain=1.414)
        if self._residual and self.res_fc is not None:
            nn.init.xavier_normal_(self.res_fc.weight, gain=1.414)


    def forward(self, feat, graph):
        """Compute multi-head atom representations

        Parameters
        ----------
        feat : FloatTensor of shape (N, M1)
            * N is the total number of atoms in the batched graph
            * M1 is the input atom feature size, must match in_feats in initialization
        bg : BatchedDGLGraph
            Batched DGLGraphs for processing multiple molecules in parallel

        Returns
        -------
        rst : FloatTensor of shape (N, H, M2)
            * N is the total number of atoms in the batched graph
            * H is the number of attention heads
            * M2 is the output atom feature size, must match out_feats in initialization
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})

        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']

        # residual
        if self._residual:
            if self.res_fc is not None:
                resval = self.res_fc(h).view(-1, self._num_heads, self._out_feats)
            else:
                resval = h.unsqueeze(1)
            rst = rst + resval

        # activation
        if self.activation is not None:
            rst = self.activation(rst)

        return rst

class GATLayer(nn.Module):
    """Single layer GAT for updating node features

    Parameters
    ----------
    in_feats : int
        Number of input atom features
    out_feats : int
        Number of output atom features for each attention head
    num_heads : int
        Number of attention heads
    feat_drop : float
        Dropout applied to the input features
    attn_drop : float
        Dropout applied to attention values of edges
    alpha : float
        Hyperparameter in LeakyReLU, slope for negative values. Default to be 0.2
    residual : bool
        Whether to perform skip connection, default to be False
    agg_mode : str
        The way to aggregate multi-head attention results, can be either
        'flatten' for concatenating all head results or 'mean' for averaging
        all head results
    activation : activation function or None
        Activation function applied to aggregated multi-head results, default to be None.
    """
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()
        self.gnn = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                           feat_drop=feat_drop, attn_drop=attn_drop,
                           alpha=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

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
            * M2 is the output atom feature size. If self.agg_mode == 'flatten', this would
              be out_feats * num_heads, else it would be just out_feats.
        """
        new_feats = self.gnn(feats, bg)
        if self.agg_mode == 'flatten':
            new_feats = new_feats.flatten(1)
        else:
            new_feats = new_feats.mean(1)

        if self.activation is not None:
            new_feats = self.activation(new_feats)

        return new_feats
