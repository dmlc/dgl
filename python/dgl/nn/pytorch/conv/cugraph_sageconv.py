"""Torch Module for GraphSAGE layer using the aggregation primitives in
cugraph-ops"""
# pylint: disable=no-member, arguments-differ, invalid-name, too-many-arguments

from torch import nn

from .cugraph_base import CuGraphBaseConv

try:
    from pylibcugraphops.pytorch import SampledCSC, StaticCSC
    from pylibcugraphops.pytorch.operators import agg_concat_n2n as SAGEConvAgg

    HAS_PYLIBCUGRAPHOPS = True
except ImportError:
    HAS_PYLIBCUGRAPHOPS = False


class CuGraphSAGEConv(CuGraphBaseConv):
    r"""An accelerated GraphSAGE layer from `Inductive Representation Learning
    on Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__ that leverages the
    highly-optimized aggregation primitives in cugraph-ops:

    .. math::
        h_{\mathcal{N}(i)}^{(l+1)} &= \mathrm{aggregate}
        \left(\{h_{j}^{l}, \forall j \in \mathcal{N}(i) \}\right)

        h_{i}^{(l+1)} &= W \cdot \mathrm{concat}
        (h_{i}^{l}, h_{\mathcal{N}(i)}^{(l+1)})

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
    aggregator_type : str
        Aggregator type to use (``mean``, ``sum``, ``min``, ``max``).
    feat_drop : float
        Dropout rate on features, default: ``0``.
    bias : bool
        If True, adds a learnable bias to the output. Default: ``True``.

    Examples
    --------
    >>> import dgl
    >>> import torch
    >>> from dgl.nn import CuGraphSAGEConv
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
    MAX_IN_DEGREE_MFG = 500

    def __init__(
        self,
        in_feats,
        out_feats,
        aggregator_type="mean",
        feat_drop=0.0,
        bias=True,
    ):
        if HAS_PYLIBCUGRAPHOPS is False:
            raise ModuleNotFoundError(
                f"{self.__class__.__name__} requires pylibcugraphops=23.04. "
                f"Install via `conda install -c nvidia 'pylibcugraphops=23.04'`."
                f"pylibcugraphops requires Python 3.8 or 3.10."
            )

        valid_aggr_types = {"max", "min", "mean", "sum"}
        if aggregator_type not in valid_aggr_types:
            raise ValueError(
                f"Invalid aggregator_type. Must be one of {valid_aggr_types}. "
                f"But got '{aggregator_type}' instead."
            )

        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggr = aggregator_type
        self.feat_drop = nn.Dropout(feat_drop)
        self.linear = nn.Linear(2 * in_feats, out_feats, bias=bias)

    def reset_parameters(self):
        r"""Reinitialize learnable parameters."""
        self.linear.reset_parameters()

    def forward(self, g, feat, max_in_degree=None):
        r"""Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Node features. Shape: :math:`(N, D_{in})`.
        max_in_degree : int
            Maximum in-degree of destination nodes. It is only effective when
            :attr:`g` is a :class:`DGLBlock`, i.e., bipartite graph. When
            :attr:`g` is generated from a neighbor sampler, the value should be
            set to the corresponding :attr:`fanout`. If not given,
            :attr:`max_in_degree` will be calculated on-the-fly.

        Returns
        -------
        torch.Tensor
            Output node features. Shape: :math:`(N, D_{out})`.
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
        h = SAGEConvAgg(feat, _graph, self.aggr)[: g.num_dst_nodes()]
        h = self.linear(h)

        return h
