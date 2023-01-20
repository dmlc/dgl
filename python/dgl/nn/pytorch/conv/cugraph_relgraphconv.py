"""Torch Module for Relational graph convolution layer using the aggregation
primitives in cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments
import math

import torch
from torch import nn

try:
    from pylibcugraphops import make_fg_csr_hg, make_mfg_csr_hg
    from pylibcugraphops.torch.autograd import (
        agg_hg_basis_n2n_post as RelGraphConvAgg,
    )
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class CuGraphRelGraphConv(nn.Module):
    r"""An accelerated relational graph convolution layer from `Modeling
    Relational Data with Graph Convolutional Networks
    <https://arxiv.org/abs/1703.06103>`__ that leverages the highly-optimized
    aggregation primitives in cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.RelGraphConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.

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
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "dgl.nn.CuGraphRelGraphConv requires pylibcugraphops >= 23.02 "
                "to be installed."
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
        # Create csc-representation and cast etypes to int32.
        offsets, indices, edge_ids = g.adj_sparse("csc")
        edge_types_perm = etypes[edge_ids.long()].int()

        # Create cugraph-ops graph.
        if g.is_block:
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()
            _graph = make_mfg_csr_hg(
                g.dstnodes(),
                offsets,
                indices,
                max_in_degree,
                g.num_src_nodes(),
                n_node_types=0,
                n_edge_types=self.num_rels,
                out_node_types=None,
                in_node_types=None,
                edge_types=edge_types_perm,
            )
        else:
            _graph = make_fg_csr_hg(
                offsets,
                indices,
                n_node_types=0,
                n_edge_types=self.num_rels,
                node_types=None,
                edge_types=edge_types_perm,
            )

        h = RelGraphConvAgg(
            feat, self.coeff, _graph, not self.self_loop, self.apply_norm
        )
        h = h @ self.W.view(-1, self.out_feat)
        if self.bias is not None:
            h = h + self.bias
        h = self.dropout(h)

        return h
