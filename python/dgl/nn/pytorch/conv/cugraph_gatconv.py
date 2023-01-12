"""Torch Module for graph attention network layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments

import torch
from torch import nn

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import mha_gat_n2n as GATConvAgg
except ModuleNotFoundError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class CuGraphGATConv(nn.Module):
    r"""Graph attention layer from `Graph Attention Network
    <https://arxiv.org/pdf/1710.10903.pdf>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.GATConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.
    max_in_degree : int, optional
        Maximum number of sampled neighbors of a destination node,
        i.e. maximum in degree of destination nodes. If ``None``, it will be
        calculated on the fly during :meth:`forward`.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import CuGraphGATConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> g = dgl.add_self_loop(g)
    >>> feat = th.ones(6, 10).to(device)
    >>> conv = CuGraphGATConv(10, 2, num_heads=3).to(device)
    >>> res = conv(g, feat)
    >>> res
    tensor([[[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]],
            [[ 0.2340,  1.9226],
            [ 1.6477, -1.9986],
            [ 1.1138, -1.9302]]], device='cuda:0', grad_fn=<ViewBackward0>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        negative_slope=0.2,
        bias=True,
        max_in_degree=None,
    ):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "dgl.nn.CuGraphGATConv requires pylibcugraphops >= 23.02 "
                "to be installed."
            )
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.max_in_degree = max_in_degree

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_weights = nn.Parameter(
            torch.Tensor(2 * num_heads * out_feats)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""

        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(
            self.attn_weights.view(2, self.num_heads, self.out_feats), gain=gain
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, feat):
        r"""Forward computation.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            Input features of shape :math:`(N, D_{in})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where
            :math:`H` is the number of heads, and :math:`D_{out}` is size of
            output feature.
        """

        offsets, indices, _ = g.adj_sparse("csc")

        if g.is_block:
            max_in_degree = self.max_in_degree
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()
            _graph = make_mfg_csr(
                g.dstnodes(), g.srcnodes(), offsets, indices, max_in_degree
            )
        else:
            _graph = make_fg_csr(offsets, indices)

        feat_transformed = self.fc(feat)
        out = GATConvAgg(
            feat_transformed,
            self.attn_weights,
            _graph,
            self.num_heads,
            "LeakyReLU",
            self.negative_slope,
            add_own_node=False,
            concat_heads=True,
        ).view(-1, self.num_heads, self.out_feats)

        if self.bias is not None:
            out = out + self.bias

        return out
