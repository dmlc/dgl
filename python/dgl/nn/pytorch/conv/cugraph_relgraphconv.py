"""Torch Module for Relational graph convolution layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments
import math

import torch
from torch import nn

from .cugraph_base import CuGraphBaseConv

try:
    from pylibcugraphops.pytorch import SampledHeteroCSC, StaticHeteroCSC
    from pylibcugraphops.pytorch.operators import (
        agg_hg_basis_n2n_post as RelGraphConvAgg,
    )

    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphRelGraphConv(CuGraphBaseConv):
    r"""An accelerated relational graph convolution layer from `Modeling
    Relational Data with Graph Convolutional Networks
    <https://arxiv.org/abs/1703.06103>`__ that leverages the highly-optimized
    aggregation primitives in cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.RelGraphConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops=23.04`.
    :code:`pylibcugraphops` 23.04 requires python 3.8.x or 3.10.x.

    .. note::
        This is an **experimental** feature.

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
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``.
    apply_norm : bool, optional
        True to normalize aggregation output by the in-degree of the destination
        node per edge type, i.e. :math:`|\mathcal{N}^r_i|`. Default: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import CuGraphRelGraphConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> feat = torch.ones(6, 10).to(device)
    >>> conv = CuGraphRelGraphConv(
    ...     10, 2, 3, regularizer='basis', num_bases=2).to(device)
    >>> etype = torch.tensor([0,1,2,0,1,2]).to(device)
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[-1.7774, -2.0184],
            [-1.4335, -2.3758],
            [-1.7774, -2.0184],
            [-0.4698, -3.0876],
            [-1.4335, -2.3758],
            [-1.4331, -2.3295]], device='cuda:0', grad_fn=<AddBackward0>)
    """
    MAX_IN_DEGREE_MFG = 500

    def __init__(
        self,
        in_feat,
        out_feat,
        num_rels,
        regularizer=None,
        num_bases=None,
        bias=True,
        self_loop=True,
        dropout=0.0,
        apply_norm=False,
    ):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(
                f"{self.__class__.__name__} requires pylibcugraphops=23.04. "
                f"Install via `conda install -c nvidia 'pylibcugraphops=23.04'`."
                f"pylibcugraphops requires Python 3.8 or 3.10."
            )
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.apply_norm = apply_norm
        self.dropout = nn.Dropout(dropout)

        dim_self_loop = 1 if self_loop else 0
        self.self_loop = self_loop
        if regularizer is None:
            self.W = nn.Parameter(
                torch.Tensor(num_rels + dim_self_loop, in_feat, out_feat)
            )
            self.coeff = None
        elif regularizer == "basis":
            if num_bases is None:
                raise ValueError(
                    'Missing "num_bases" for basis regularization.'
                )
            self.W = nn.Parameter(
                torch.Tensor(num_bases + dim_self_loop, in_feat, out_feat)
            )
            self.coeff = nn.Parameter(torch.Tensor(num_rels, num_bases))
            self.num_bases = num_bases
        else:
            raise ValueError(
                f"Supported regularizer options: 'basis' or None, but got "
                f"'{regularizer}'."
            )
        self.regularizer = regularizer

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        bound = 1 / math.sqrt(self.in_feat)
        end = -1 if self.self_loop else None
        nn.init.uniform_(self.W[:end], -bound, bound)
        if self.regularizer == "basis":
            nn.init.xavier_uniform_(
                self.coeff, gain=nn.init.calculate_gain("relu")
            )
        if self.self_loop:
            nn.init.xavier_uniform_(self.W[-1], nn.init.calculate_gain("relu"))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, g, feat, etypes, max_in_degree=None):
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
        max_in_degree : int, optional
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        offsets, indices, edge_ids = g.adj_tensors("csc")
        edge_types_perm = etypes[edge_ids.long()].int()

        if g.is_block:
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()

            if max_in_degree < self.MAX_IN_DEGREE_MFG:
                _graph = SampledHeteroCSC(
                    offsets,
                    indices,
                    edge_types_perm,
                    max_in_degree,
                    g.num_src_nodes(),
                    self.num_rels,
                )
            else:
                offsets_fg = self.pad_offsets(offsets, g.num_src_nodes() + 1)
                _graph = StaticHeteroCSC(
                    offsets_fg,
                    indices,
                    edge_types_perm,
                    self.num_rels,
                )
        else:
            _graph = StaticHeteroCSC(
                offsets,
                indices,
                edge_types_perm,
                self.num_rels,
            )

        h = RelGraphConvAgg(
            feat,
            self.coeff,
            _graph,
            concat_own=self.self_loop,
            norm_by_out_degree=self.apply_norm,
        )[: g.num_dst_nodes()]
        h = h @ self.W.view(-1, self.out_feat)
        if self.bias is not None:
            h = h + self.bias
        h = self.dropout(h)

        return h
