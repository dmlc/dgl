"""Torch Module for GraphSAGE layer using the aggregation primitives in
cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments

from torch import nn

try:
    from pylibcugraphops import make_fg_csr, make_mfg_csr
    from pylibcugraphops.torch.autograd import agg_concat_n2n as SAGEConvAgg
except ImportError:
    has_pylibcugraphops = False
else:
    has_pylibcugraphops = True


class CuGraphSAGEConv(nn.Module):
    r"""An accelerated GraphSAGE layer from `Inductive Representation Learning
    on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__ that leverages the
    highly-optimized aggregation primitives in cugraph-ops.

    See :class:`dgl.nn.pytorch.conv.SAGEConv` for mathematical model.

    This module depends on :code:`pylibcugraphops` package, which can be
    installed via :code:`conda install -c nvidia pylibcugraphops>=23.02`.

    .. note::
        This is an **experimental** feature.

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    aggregator_type : str
        Aggregator type to use (``mean``, ``sum``, ``min``, ``max``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.
    norm : callable activation function/layer or None, optional
        If not None, applies normalization to the updated node features.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import CuGraphSAGEConv
    ...
    >>> device = 'cuda'
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3])).to(device)
    >>> g = dgl.add_self_loop(g)
    >>> feat = torch.ones(6, 10).to(device)
    >>> conv = CuGraphSAGEConv(10, 2, 'mean').to(device)
    >>> res = conv(g, feat)
    >>> res
    tensor([[-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952],
            [-1.1690,  0.1952]], device='cuda:0', grad_fn=<AddmmBackward0>)
    """

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type="mean",
        feat_drop=0.0,
        bias=True,
        norm=None,
    ):
        if has_pylibcugraphops is False:
            raise ModuleNotFoundError(
                "dgl.nn.CuGraphSAGEConv requires pylibcugraphops >= 23.02 "
                "to be installed."
            )
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        valid_aggr_types = {"max", "min", "mean", "sum"}
        if aggregator_type not in valid_aggr_types:
            raise ValueError(
                f"Invalid aggregator_type. Must be one of {valid_aggr_types}. "
                f"But got '{aggregator_type}' instead."
            )
        self.aggr = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.norm = norm

        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)

    def forward(self, g, feat, max_in_degree=None):
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Node features. Shape: :math:`(|V|, D_{in})`.
        max_in_degree : int
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            Output node features. Shape: :math:`(|V|, D_{out})`.
        """
        offsets, indices, _ = g.adj_sparse("csc")

        if g.is_block:
            if max_in_degree is None:
                max_in_degree = g.in_degrees().max().item()

            _graph = make_mfg_csr(
                g.dstnodes(), offsets, indices, max_in_degree, g.num_src_nodes()
            )
        else:
            _graph = make_fg_csr(offsets, indices)

        feat = self.feat_drop(feat)
        h = SAGEConvAgg(feat, _graph, self.aggr)
        h = self.linear(h)

        if self.norm is not None:
            h = self.norm(h)

        return h
