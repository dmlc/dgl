"""Torch Module for Relational graph convolution layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments
import math

import torch as th
from torch import nn

try:
    from pylibcugraphops import make_mfg_csr_hg
    from pylibcugraphops.operators import (
        agg_hg_basis_mfg_n2n_post_bwd as agg_bwd,
    )
    from pylibcugraphops.operators import (
        agg_hg_basis_mfg_n2n_post_fwd as agg_fwd,
    )
except ImportError:
    has_pylibcugraphops = False

    def make_mfg_csr_hg(*args):
        r"""A dummy function to help raise error in RelGraphConvAgg when
        pylibcugraphops is not found."""

        raise NotImplementedError(
            "RelGraphConvAgg requires pylibcugraphops to be installed."
        )

else:
    has_pylibcugraphops = True


class RelGraphConvAgg(th.autograd.Function):
    r"""Custom autograd function for R-GCN aggregation layer that uses the
    aggregation functions in cugraph-ops."""

    @staticmethod
    def forward(ctx, g, num_rels, edge_types, max_in_degree, feat, coeff):
        r"""Compute the forward pass of R-GCN aggregation layer.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFunction
            Context object used to stash information for backward computation.
        g : DGLGraph
            The graph.
        num_rels : int
            Number of relations.
        edge_types : torch.Tensor
            A 1D tensor of edge types.
        max_in_degree : int
            Maximum number of sampled neighbors of a destination node.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: (num_src_nodes, in_feat).
        coeff : torch.Tensor
            A 2D tensor of the coefficient matrix used in basis-decomposition
            regularization. Shape: (num_rels, num_bases). It should be set to
            ``None`` when no regularization is applied.

        Returns
        -------
        agg_output : torch.Tensor
            A 2D tensor of aggregation output. Shape: (num_dst_nodes,
            num_rels * in_feat) when ``coeff=None``; Shape: (num_dst_nodes,
            num_bases * in_feat) otherwise.
        """

        in_feat = feat.shape[-1]
        indptr, indices, edge_ids = g.adj_sparse("csc")
        # Edge_ids is in a mixed order, need to permutate incoming etypes.
        ctx.edge_types_perm = edge_types[edge_ids.long()].int()

        mfg = make_mfg_csr_hg(
            g.dstnodes(),
            g.srcnodes(),
            indptr,
            indices,
            max_in_degree,
            n_node_types=0,
            n_edge_types=num_rels,
            out_node_types=None,
            in_node_types=None,
            edge_types=ctx.edge_types_perm,
        )
        ctx.mfg = mfg

        if coeff is None:
            leading_dimension = num_rels * in_feat
        else:
            num_bases = coeff.shape[-1]
            leading_dimension = num_bases * in_feat

        agg_output = th.empty(
            g.num_dst_nodes(),
            leading_dimension,
            dtype=th.float32,
            device=feat.device,
        )

        if coeff is None:
            agg_fwd(agg_output, feat.detach(), None, mfg)
        else:
            agg_fwd(agg_output, feat.detach(), coeff.detach(), mfg)

        ctx.save_for_backward(feat, coeff)
        return agg_output

    @staticmethod
    def backward(ctx, grad_output):
        r"""Compute the backward pass of R-GCN aggregation layer.

        Parameters
        ----------
        ctx : torch.autograd.function.BackwardCFunction
            Context object used to stash information for backward computation.
        grad_output : torch.Tensor
            A 2D tensor of the gradient of loss function w.r.t output.
        """
        feat, coeff = ctx.saved_tensors

        grad_feat = th.empty_like(feat)
        grad_coeff = None if coeff is None else th.empty_like(coeff)

        if coeff is None:
            agg_bwd(grad_feat, None, grad_output, feat.detach(), None, ctx.mfg)
        else:
            agg_bwd(
                grad_feat,
                grad_coeff,
                grad_output,
                feat.detach(),
                coeff.detach(),
                ctx.mfg,
            )

        return None, None, None, None, grad_feat, grad_coeff


class CuGraphRelGraphConv(nn.Module):
    r"""An accelerated relational graph convolution layer from `Modeling
    Relational Data with Graph Convolutional Networks
    <https://arxiv.org/abs/1703.06103>`__ that leverages the highly-optimized
    aggregation primitives in cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.RelGraphConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=22.12`.

    .. note::
        This is an **experimental** feature.
        Compared with :class:`dgl.nn.pytorch.conv.RelGraphConv`, this model:

        * Only works on cuda devices.
        * Only supports basis-decomposition regularization.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str, optional
        Which weight regularizer to use ("basis" or ``None``):
         - "basis" is for basis-decomposition.
         - ``None`` applies no regularization.
        Default: ``None``.
    num_bases : int, optional
        Number of bases. It comes into effect when a regularizer is applied.
        Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``.
    layer_norm : bool, optional
        True to add layer norm. Default: ``False``.
    max_in_degree : int, optional
        Maximum number of sampled neighbors of a destination node,
        i.e. maximum in degree of destination nodes. If ``None``, it will be
        calculated on the fly during :meth:`forward`.

    Examples
    --------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import CuGraphRelGraphConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> feat = th.ones(6, 10).to(device)
    >>> conv = CuGraphRelGraphConv(
    ...     10, 2, 3, regularizer='basis', num_bases=2).to(device)
    >>> etype = th.tensor([0,1,2,0,1,2]).to(device)
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[-1.7774, -2.0184],
            [-1.4335, -2.3758],
            [-1.7774, -2.0184],
            [-0.4698, -3.0876],
            [-1.4335, -2.3758],
            [-1.4331, -2.3295]], device='cuda:0', grad_fn=<AddBackward0>)
    """

    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        activation=None,
        self_loop=True,
        dropout=0.0,
        layer_norm=False,
        max_in_degree=None,
    ):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "dgl.nn.CuGraphRelGraphConv requires pylibcugraphops "
                "to be installed."
            )
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.max_in_degree = max_in_degree

        # regularizer
        if regularizer is None:
            self.W = nn.Parameter(th.Tensor(num_rels, in_feat, out_feat))
            self.coeff = None
        elif regularizer == "basis":
            if num_bases is None:
                raise ValueError(
                    'Missing "num_bases" for basis regularization.'
                )
            self.W = nn.Parameter(th.Tensor(num_bases, in_feat, out_feat))
            self.coeff = nn.Parameter(th.Tensor(num_rels, num_bases))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f"Supported regularizer options: 'basis' or None, but got "
                f"{regularizer}."
            )
        self.regularizer = regularizer

        # Initialize weights.
        with th.no_grad():
            if self.regularizer is None:
                nn.init.uniform_(
                    self.W,
                    -1 / math.sqrt(self.in_feat),
                    1 / math.sqrt(self.in_feat),
                )
            else:
                nn.init.uniform_(
                    self.W,
                    -1 / math.sqrt(self.in_feat),
                    1 / math.sqrt(self.in_feat),
                )
                nn.init.xavier_uniform_(
                    self.coeff, gain=nn.init.calculate_gain("relu")
                )

        # others
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(
                out_feat, elementwise_affine=True
            )

        # weight for self_loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(
                self.loop_weight, gain=nn.init.calculate_gain("relu")
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, feat, etypes, norm=None):
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor
            A 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
            Note that cugraph-ops only accepts edge type tensors in int32,
            so any input of other integer types will be casted into int32,
            thus introducing some overhead. Pass in int32 tensors directly
            for best performance.
        norm : torch.Tensor, optional
            A 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        _device = next(self.parameters()).device
        if _device.type != "cuda":
            raise RuntimeError(
                f"dgl.nn.CuGraphRelGraphConv requires the model to be on "
                f"device 'cuda', but got '{_device.type}'."
            )
        if _device != g.device:
            raise RuntimeError(
                f"Expected model and graph on the same device, "
                f"but got '{_device}' and '{g.device}'."
            )
        if _device != etypes.device:
            raise RuntimeError(
                f"Expected model and etypes on the same device, "
                f"but got '{_device}' and '{etypes.device}'."
            )
        if _device != feat.device:
            raise RuntimeError(
                f"Expected model and feature tensor on the same device, "
                f"but got '{_device}' and '{feat.device}'."
            )
        # Compute max_in_degree.
        max_in_degree = self.max_in_degree
        if max_in_degree is None:
            max_in_degree = g.in_degrees().max().item()

        with g.local_scope():
            g.srcdata["h"] = feat
            if norm is not None:
                g.edata["norm"] = norm
            # Message passing.
            h = RelGraphConvAgg.apply(
                g, self.num_rels, etypes, max_in_degree, feat, self.coeff
            )
            h = h @ self.W.view(-1, self.out_feat)
            # Apply bias and activation.
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[: g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
