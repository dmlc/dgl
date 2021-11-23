"""Torch modules for link prediction."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch
import torch.nn as nn

from ... import function as fn

__all__ = ['ElementwisePredictor', 'ConcatPredictor']

class ElementwisePredictor(nn.Module):
    r"""

    Description
    -----------
    Link predictor based on element-wise product (Hadamard product) introduced in
    `Open Graph Benchmark: Datasets for Machine Learning on Graphs
    <https://arxiv.org/abs/2005.00687>`. It applies the element-wise product to pairs
    of node representations and optionally passes the results to a predictor like
    MLP for the final prediction.

    Parameters
    ----------
    predictor : callable, optional
        If not None, apply this predictor after taking the element-wise product. For example,
        one can use an MLP. The input size of the predictor should be the same as the size of
        node representations.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> from dgl.nn import ElementwisePredictor

    Define an MLP class for predictor.

    >>> class MLP(nn.Module):
    ...     def __init__(self, in_feats, hidden_feats, out_feats):
    ...         super(MLP, self).__init__()
    ...         self.layer1 = nn.Linear(in_feats, hidden_feats)
    ...         self.layer2 = nn.Linear(hidden_feats, out_feats)
    ...
    ...     def reset_parameters(self):
    ...         self.layer1.reset_parameters()
    ...         self.layer2.reset_parameters()
    ...
    ...     def forward(self, h):
    ...         h = self.layer1(h)
    ...         h = F.relu(h)
    ...         return self.layer2(h)

    Case1: apply to the edges of a graph

    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> in_feats = 2
    >>> feat = torch.randn((g.num_nodes(), in_feats))

    Use ElementwisePredictor without an MLP

    >>> link_pred = ElementwisePredictor()
    >>> link_pred(feat, g).shape
    torch.Size([3, 2])

    Use ElementwisePredictor with an MLP

    >>> predictor = MLP(in_feats=in_feats, hidden_feats=3, out_feats=1)
    >>> link_pred = ElementwisePredictor(predictor)
    >>> link_pred(feat, g).shape
    torch.Size([3, 1])

    Case2: apply to arbitrary node pairs

    >>> num_pairs = 3
    >>> feat_i = torch.randn((num_pairs, in_feats))
    >>> feat_j = torch.randn((num_pairs, in_feats))
    >>> link_pred = ElementwisePredictor()
    >>> link_pred((feat_i, feat_j)).shape
    torch.Size([3, 2])
    """
    def __init__(self,
                 predictor=None):
        super(ElementwisePredictor, self).__init__()

        self.predictor = predictor

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        if hasattr(self.predictor, 'reset_parameters'):
            self.predictor.reset_parameters()

    def forward(self, feat, graph=None):
        r"""

        Computes element-wise product of pairs of node representations for the edges of a graph
        or arbitrary node pairs. It then passes the results to a predictor if specified.

        Parameters
        ----------
        feat : torch.Tensor or pair of torch.Tensor

            - If a torch.Tensor is given, it is the node feature of shape :math:`(N, D_{in})`,
              where :math:`N` is the number of nodes and :math:`D_{in}` is the input feature size.
            - If a pair of torch.Tensor is given, the pair must contain two tensors of shape
              :math:`(E, D_{in})`, where :math:`E` is the number of node pairs.

        graph : DGLGraph, optional
            This argument is only required if :attr:`feat` is a single torch.Tensor. In this case,
            the function will score the edges of the graph. The graph should have the same number
            of nodes as ``feat.shape[0]``.

        Returns
        -------
        torch.Tensor
            The tensor is of shape :math:`(E, D_{out})`, where :math:`E` is the number of
            edges/node pairs.
        """
        if isinstance(feat, tuple):
            h_i, h_j = feat
            h = h_i * h_j
        else:
            with graph.local_scope():
                graph.ndata['h'] = feat
                graph.apply_edges(fn.u_mul_v('h', 'h', 'h'))
                h = graph.edata['h']

        if self.predictor is not None:
            h = self.predictor(h)
        return h

class ConcatPredictor(nn.Module):
    r"""

    Description
    -----------
    Link predictor based on the concatenation of source node representations,
    destination node representations, and optionally relation representations.
    It then passes the results to a predictor if specified.

    With this class, one can implement MLP-based triple encoder introduced in
    `Knowledge vault: A web-scale approach to probabilistic knowledge fusion
    <https://www.cs.ubc.ca/~murphyk/Papers/kv-kdd14.pdf>`.

    Parameters
    ----------
    predictor : callable, optional
        If not None, apply this predictor after taking the concatenation. For example,
        one can use an MLP. The input size of the predictor should be the sum of the size
        of source node representation, destination node representation, and optionally
        the relation representation.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> import torch.nn as nn
    >>> import torch.nn.functional as F
    >>> from dgl.nn import ConcatPredictor

    Define an MLP class for predictor.

    >>> class MLP(nn.Module):
    ...     def __init__(self, in_feats, hidden_feats, out_feats):
    ...         super(MLP, self).__init__()
    ...         self.layer1 = nn.Linear(in_feats, hidden_feats)
    ...         self.layer2 = nn.Linear(hidden_feats, out_feats)
    ...
    ...     def reset_parameters(self):
    ...         self.layer1.reset_parameters()
    ...         self.layer2.reset_parameters()
    ...
    ...     def forward(self, h):
    ...         h = self.layer1(h)
    ...         h = F.relu(h)
    ...         return self.layer2(h)

    Case1: apply to pairs of node representations

    >>> src_feat_size = 2
    >>> dst_feat_size = 4
    >>> num_edges = 5
    >>> src_feats = torch.randn((num_edges, src_feat_size))
    >>> dst_feats = torch.randn((num_edges, dst_feat_size))

    Use ConcatPredictor without an MLP

    >>> link_pred = ConcatPredictor()
    >>> link_pred(src_feats, dst_feats).shape
    torch.Size([5, 6])

    Use ConcatPredictor with an MLP

    >>> predictor = MLP(in_feats=src_feat_size + dst_feat_size, hidden_feats=3, out_feats=1)
    >>> link_pred = ConcatPredictor(predictor)
    >>> link_pred(src_feats, dst_feats).shape
    torch.Size([5, 1])

    Case2: apply to source node, destination node, and relation representations

    >>> rel_feat_size = 3
    >>> rel_feats = torch.randn((num_edges, rel_feat_size))
    >>> in_feat_size = src_feat_size + dst_feat_size + rel_feat_size
    >>> predictor = MLP(in_feats=in_feat_size, hidden_feats=3, out_feats=1)
    >>> link_pred = ConcatPredictor(predictor)
    >>> link_pred(src_feats, dst_feats, rel_feats).shape
    torch.Size([5, 1])
    """
    def __init__(self,
                 predictor=None):
        super(ConcatPredictor, self).__init__()

        self.predictor = predictor

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        if hasattr(self.predictor, 'reset_parameters'):
            self.predictor.reset_parameters()

    def forward(self, src_feats, dst_feats, rel_feats=None):
        r"""

        Concatenates source node representation, destination node representation, and
        optionally relation representation for arbitrary node pairs or triples. It
        then passes the results to a predictor if specified.

        Parameters
        ----------
        src_feats : torch.Tensor
            Source node representations, which is of shape :math:`(E, D1)`, where :math:`E`
            is the number of node pairs.
        dst_feats : torch.Tensor
            Destination node representations, which is of shape :math:`(E, D2)`.
        rel_feats : torch.Tensor, optional
            Relation representations, which is of shape :math:`(E, D3)`.

        Returns
        -------
        torch.Tensor
            The tensor is of shape :math:`(E, D_{out})`, where :math:`D_{out}` is D1 + D2 + D3
            if the predictor is not speicifed and the predictor's output size otherwise.
        """
        if rel_feats is None:
            h = torch.cat([src_feats, dst_feats], dim=-1)
        else:
            h = torch.cat([src_feats, rel_feats, dst_feats], dim=-1)

        if self.predictor is not None:
            h = self.predictor(h)
        return h
