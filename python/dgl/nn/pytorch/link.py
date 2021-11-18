"""Torch modules for link prediction."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch.nn as nn
import torch.nn.functional as F

from ... import function as fn

__all__ = ['HadamardPredictor']

class HadamardPredictor(nn.Module):
    r"""

    Description
    -----------
    Link predictor based on Hadamard product (element-wise product) introduced in
    `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`. It applies the Hadamard product to pairs
    of node representations and passes the results to an MLP for the final prediction.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    hidden_feats : int
        Hidden feature size.
    out_feats : int
        Output feature size.
    num_layers : int, optional
        Number of linear layers, which should be at least 2.
    dropout : float, optional
        Dropout to apply in MLP.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function in MLP.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import HadamardPredictor

    Case1: score the edges of a graph

    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> in_feats = 2
    >>> feat = torch.randn((g.num_nodes(), in_feats))
    >>> predictor = HadamardPredictor(in_feats=in_feats, hidden_feats=3, out_feats=1)
    >>> predictor(feat, g).shape
    torch.Size([3, 1])

    Case2: score arbitrary node pairs

    >>> num_pairs = 3
    >>> feat_i = torch.randn((num_pairs, in_feats))
    >>> feat_j = torch.randn((num_pairs, in_feats))
    >>> predictor = HadamardPredictor(in_feats=in_feats, hidden_feats=3, out_feats=1)
    >>> predictor((feat_i, feat_j)).shape
    torch.Size([3, 1])
    """
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_layers=2,
                 dropout=0.,
                 activation=F.relu):
        super(HadamardPredictor, self).__init__()

        assert num_layers >= 2, 'Expect num_layers to be at least 2.'
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_feats, hidden_feats))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_feats, hidden_feats))
        self.layers.append(nn.Linear(hidden_feats, out_feats))

        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feat, graph=None):
        r"""

        Scores the edges of a graph or arbitrary node pairs.

        Parameters
        ----------
        feat : torch.Tensor or pair of torch.Tensor

            - If a torch.Tensor is given, it is the node feature of shape :math:`(N, D_{in})`,
              where :math:`N` is the number of nodes and :math:`D_{in}` is the input feature size.
            - If a pair of torch.Tensor is given, the pair must contain two tensors of shape
              :math:`(N, D_{in})`.

        graph : DGLGraph, optional
            This argument is only required if :attr:`feat` is a single torch.Tensor. In this case,
            the function will score the edges of the graph. The graph should have the same number
            of nodes as ``feat.shape[0]``.

        Returns
        -------
        torch.Tensor
            The unnormalized scores of edges/node pairs, which is of shape :math:`(E, D_{out})`,
            where :math:`E` is the number of edges/node pairs and :math:`D_{out}` is the output
            feature size.
        """
        if isinstance(feat, tuple):
            h_i, h_j = feat
            h = h_i * h_j
        else:
            with graph.local_scope():
                graph.ndata['h'] = feat
                graph.apply_edges(fn.u_mul_v('h', 'h', 'h'))
                h = graph.edata['h']

        for layer in self.layers[:-1]:
            h = layer(h)
            if self.activation is not None:
                h = self.activation(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return self.layers[-1](h)
