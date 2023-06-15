"""Torch Module for graph attention network layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments

import torch
from torch import nn

from .cugraph_base import CuGraphBaseConv

try:
    from pylibcugraphops.pytorch import SampledCSC, StaticCSC
    from pylibcugraphops.pytorch.operators import mha_gat_n2n as GATConvAgg

    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphGATConv(CuGraphBaseConv):
    r"""Graph attention layer from `Graph Attention Networks
    <https://arxiv.org/pdf/1710.10903.pdf>`__, with the sparse aggregation
    accelerated by cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.GATConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops=23.04`.
    :code:`pylibcugraphops` 23.04 requires python 3.8.x or 3.10.x.

    .. note::
        This is an **experimental** feature.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature. Defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope. Defaults: ``0.2``.
    residual : bool, optional
        If True, use residual connection. Defaults: ``False``.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    bias : bool, optional
        If True, learns a bias term. Defaults: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import CuGraphGATConv
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> g = dgl.add_self_loop(g)
    >>> feat = torch.ones(6, 10).to(device)
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
    MAX_IN_DEGREE_MFG = 200

    def __init__(
        self,
        in_feats,
        out_feats,
        num_heads,
        feat_drop=0.0,
        negative_slope=0.2,
        residual=False,
        activation=None,
        bias=True,
    ):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(
                f"{self.__class__.__name__} requires pylibcugraphops=23.04. "
                f"Install via `conda install -c nvidia 'pylibcugraphops=23.04'`."
                f"pylibcugraphops requires Python 3.8 or 3.10."
            )
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_heads = num_heads
        self.feat_drop = nn.Dropout(feat_drop)
        self.negative_slope = negative_slope
        self.activation = activation

        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_weights = nn.Parameter(
            torch.Tensor(2 * num_heads * out_feats)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * out_feats))
        else:
            self.register_buffer("bias", None)

        if residual:
            if in_feats == out_feats * num_heads:
                self.res_fc = nn.Identity()
            else:
                self.res_fc = nn.Linear(
                    in_feats, out_feats * num_heads, bias=False
                )
        else:
            self.register_buffer("res_fc", None)

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

        if isinstance(self.res_fc, nn.Linear):
            self.res_fc.reset_parameters()

    def forward(self, g, feat, max_in_degree=None):
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input features of shape :math:`(N, D_{in})`.
        max_in_degree : int
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where
            :math:`H` is the number of heads, and :math:`D_{out}` is size of
            output feature.
        """
        offsets, indices, _ = g.adj_tensors("csc")

        if g.is_block:
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()

            if max_in_degree < self.MAX_IN_DEGREE_MFG:
                _graph = SampledCSC(
                    offsets,
                    indices,
                    max_in_degree,
                    g.num_src_nodes(),
                )
            else:
                offsets_fg = self.pad_offsets(offsets, g.num_src_nodes() + 1)
                _graph = StaticCSC(offsets_fg, indices)
        else:
            _graph = StaticCSC(offsets, indices)

        feat = self.feat_drop(feat)
        feat_transformed = self.fc(feat)
        out = GATConvAgg(
            feat_transformed,
            self.attn_weights,
            _graph,
            self.num_heads,
            "LeakyReLU",
            self.negative_slope,
            concat_heads=True,
        )[: g.num_dst_nodes()].view(-1, self.num_heads, self.out_feats)

        feat_dst = feat[: g.num_dst_nodes()]
        if self.res_fc is not None:
            out = out + self.res_fc(feat_dst).view(
                -1, self.num_heads, self.out_feats
            )

        if self.bias is not None:
            out = out + self.bias.view(-1, self.num_heads, self.out_feats)

        if self.activation is not None:
            out = self.activation(out)

        return out
