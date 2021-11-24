"""Torch Module for Signed Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
import torch.nn.functional as F

from .... import function as fn


class SignConv(nn.Module):
    r"""

    Description
    -----------
    Signed Graph Convolutional layer from paper `Signed Graph
    Convolutional Network <https://arxiv.org/abs/1808.06354>`__

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{neg})}
        \left[ \frac{1}{|\mathcal{N}^{-}(v)|} \sum_{w \in \mathcal{N}^{-}(v)}
        \mathbf{x}_w , \mathbf{x}_v \right]

    if :obj:`first_aggr` is set to :obj:`True`, and

    .. math::
        \mathbf{x}_v^{(\textrm{pos})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{pos})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{neg})},
        \mathbf{x}_v^{(\textrm{pos})} \right]

        \mathbf{x}_v^{(\textrm{neg})} &= \mathbf{\Theta}^{(\textrm{pos})}
        \left[ \frac{1}{|\mathcal{N}^{+}(v)|} \sum_{w \in \mathcal{N}^{+}(v)}
        \mathbf{x}_w^{(\textrm{neg})}, \frac{1}{|\mathcal{N}^{-}(v)|}
        \sum_{w \in \mathcal{N}^{-}(v)} \mathbf{x}_w^{(\textrm{pos})},
        \mathbf{x}_v^{(\textrm{neg})} \right]


    Follows codes from original author's repo at `https://github.com/benedekrozemberczki/SGCN/`

    Parameters
    ----------
    in_feats : int
        Input feature size. i.e, the number of dimensions of :math:`X`.
    out_feats : int
        Output feature size.  i.e, the number of dimensions of :math:`H^{K}`.
    first_aggr: bool
        Whether this is used for first layer aggregation
    bias: bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    norm: bool, optional
        If True, using mean operator over neighbor's feature instead of sum.
    norm_embed: bool, optional
        If True, normalize the output vector with L2 norm.

    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import SignConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> conv = SignConv(10, 2, True) # First layer aggregation
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4238,  0.9058],
            [-0.4238,  0.9058],
            [-0.4238,  0.9058],
            [-0.4238,  0.9058],
            [-0.4238,  0.9058],
            [-0.8234,  0.5674]], grad_fn=<DivBackward0>)
    >>> conv2 = SignConv(10, 2, False) # Non-first layer aggregation
    >>> edge_sign = th.randn(g.num_edges()) # Use random edge weight as sign
    >>> res = conv2(g, feat, edge_sign)
    >>> res
    tensor([[ 0.8376, -0.5463],
        [ 0.8376, -0.5463],
        [ 0.9930,  0.1184],
        [ 0.9930,  0.1184],
        [ 0.9930,  0.1184],
        [ 0.9817,  0.1904]], grad_fn=<DivBackward0>)
    """

    def __init__(self, in_feats,
                 out_feats, first_aggr,
                 norm=True,
                 norm_embed=True,
                 bias=True):
        super(SignConv, self).__init__()

        self.in_channels = in_feats
        self.out_channels = out_feats
        self.first_aggr = first_aggr
        self.norm = norm
        self.norm_embed = norm_embed
        if first_aggr:
            self.linear_layer = nn.Linear(in_feats * 2, out_feats, bias=bias)
        else:
            self.linear_layer = nn.Linear(in_feats * 3, out_feats, bias=bias)

    def forward(self, graph, feature, edge_sign=None):
        r"""

        Description
        -----------
        Compute signed graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.
        edge_sign: torch.Tensor, optional
            Required if first_aggr is set to False.
            Indicate the sign of the edges defined in the equation. Positive entries
            are considered as positive edges. Negative entries are considered
            as negative edges.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where :math:`D_{out}`
            is size of output feature.
        """
        if self.first_aggr:
            assert edge_sign is None, "Cannot use edge_sign if first_aggr is set to True"
            return self.forward_base(graph, feature)
        else:
            assert edge_sign is not None, "Need to provide edge_sign if first_aggr is set to False"
            return self.forward_deep(graph, feature, edge_sign)

    def forward_base(self, graph, feature):
        with graph.local_scope():
            graph.ndata["feat"] = feature            
            if self.norm:
                graph.update_all(fn.copy_u("feat", "m"), fn.mean("m", "out"))
            else:
                graph.update_all(fn.copy_u("feat", "m"), fn.sum("m", "out"))

            out = th.cat([graph.ndata["out"], feature], 1)
            out = self.linear_layer(out)

            if self.norm_embed:
                out = F.normalize(out, p=2, dim=-1)

        return out

    def forward_deep(self, graph, feature, edge_sign):
        with graph.local_scope():
            graph.ndata["feat"] = feature
            graph.edata["pos_e"] = (edge_sign >= 0).float()
            graph.edata["neg_e"] = (edge_sign < 0).float()
            if self.norm:
                reducer = fn.mean
            else:
                reducer = fn.sum
            graph.update_all(fn.u_mul_e("feat", "pos_e", "m"),
                             reducer("m", "pos_out"))
            graph.update_all(fn.u_mul_e("feat", "neg_e", "m"),
                             reducer("m", "neg_out"))

            pos_out = graph.ndata["pos_out"]
            neg_out = graph.ndata["neg_out"]

        out = th.cat([pos_out, neg_out, feature], 1)
        out = self.linear_layer(out)

        if self.norm_embed:
            out = F.normalize(out, p=2, dim=-1)

        return out
