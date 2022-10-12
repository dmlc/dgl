"""Predictor for edges in homogeneous graphs."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgePredictor(nn.Module):
    r"""Predictor/score function for pairs of node representations

    Given a pair of node representations, :math:`h_i` and :math:`h_j`, it combines them with

    **dot product**

    .. math::

        h_i^{T} h_j

    or **cosine similarity**

    .. math::

        \frac{h_i^{T} h_j}{{\| h_i \|}_2 \cdot {\| h_j \|}_2}

    or **elementwise product**

    .. math::

        h_i \odot h_j

    or **concatenation**

    .. math::

        h_i \Vert h_j

    Optionally, it passes the combined results to a linear layer for the final prediction.

    Parameters
    ----------
    op : str
        The operation to apply. It can be 'dot', 'cos', 'ele', or 'cat',
        corresponding to the equations above in order.
    in_feats : int, optional
        The input feature size of :math:`h_i` and :math:`h_j`. It is required
        only if a linear layer is to be applied.
    out_feats : int, optional
        The output feature size. It is reuiqred only if a linear layer is to be applied.
    bias : bool, optional
        Whether to use bias for the linear layer if it applies.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EdgePredictor
    >>> num_nodes = 2
    >>> num_edges = 3
    >>> in_feats = 4
    >>> g = dgl.rand_graph(num_nodes=num_nodes, num_edges=num_edges)
    >>> h = th.randn(num_nodes, in_feats)
    >>> src, dst = g.edges()
    >>> h_src = h[src]
    >>> h_dst = h[dst]

    Case1: dot product

    >>> predictor = EdgePredictor('dot')
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 1])
    >>> predictor = EdgePredictor('dot', in_feats, out_feats=3)
    >>> predictor.reset_parameters()
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 3])

    Case2: cosine similarity

    >>> predictor = EdgePredictor('cos')
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 1])
    >>> predictor = EdgePredictor('cos', in_feats, out_feats=3)
    >>> predictor.reset_parameters()
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 3])

    Case3: elementwise product

    >>> predictor = EdgePredictor('ele')
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 4])
    >>> predictor = EdgePredictor('ele', in_feats, out_feats=3)
    >>> predictor.reset_parameters()
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 3])

    Case4: concatenation

    >>> predictor = EdgePredictor('cat')
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 8])
    >>> predictor = EdgePredictor('cat', in_feats, out_feats=3)
    >>> predictor.reset_parameters()
    >>> predictor(h_src, h_dst).shape
    torch.Size([3, 3])
    """

    def __init__(self, op, in_feats=None, out_feats=None, bias=False):
        super(EdgePredictor, self).__init__()

        assert op in [
            "dot",
            "cos",
            "ele",
            "cat",
        ], "Expect op to be in ['dot', 'cos', 'ele', 'cat'], got {}".format(op)
        self.op = op
        if (in_feats is not None) and (out_feats is not None):
            if op in ["dot", "cos"]:
                in_feats = 1
            elif op == "cat":
                in_feats = 2 * in_feats
            self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        else:
            self.linear = None

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        if self.linear is not None:
            self.linear.reset_parameters()

    def forward(self, h_src, h_dst):
        r"""

        Description
        -----------
        Predict for pairs of node representations.

        Parameters
        ----------
        h_src : torch.Tensor
            Source node features. The tensor is of shape :math:`(E, D_{in})`,
            where :math:`E` is the number of edges/node pairs, and :math:`D_{in}`
            is the input feature size.
        h_dst : torch.Tensor
            Destination node features. The tensor is of shape :math:`(E, D_{in})`,
            where :math:`E` is the number of edges/node pairs, and :math:`D_{in}`
            is the input feature size.

        Returns
        -------
        torch.Tensor
            The output features.
        """
        if self.op == "dot":
            N, D = h_src.shape
            h = torch.bmm(h_src.view(N, 1, D), h_dst.view(N, D, 1)).squeeze(-1)
        elif self.op == "cos":
            h = F.cosine_similarity(h_src, h_dst).unsqueeze(-1)
        elif self.op == "ele":
            h = h_src * h_dst
        else:
            h = torch.cat([h_src, h_dst], dim=-1)

        if self.linear is not None:
            h = self.linear(h)

        return h
