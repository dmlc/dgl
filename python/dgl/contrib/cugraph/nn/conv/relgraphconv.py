"""Torch Module for Relational graph convolution layer using cugraph-ops"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import (agg_hg_basis_post_fwd_int32,
    agg_hg_basis_post_bwd_int32, agg_hg_basis_post_fwd_int64, agg_hg_basis_post_bwd_int64)
from pylibcugraphops.structure.graph_types import (message_flow_graph_hg_csr_int32,
    message_flow_graph_hg_csr_int64)

CUDA_SM_PER_BLOCK = 49152

class RelGraphConvAgg(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, fanout, num_rels, edge_types, feat, coeff):
        """
        Compute the forward pass of R-GCN aggregation layer.

        Parameters
        ----------
        g : dgl.heterograph.DGLHeteroGraph
            Heterogeneous graph.
        fanout : int
            Maximum in-degree of nodes.
        num_rels : int
            Number of edge types in this graph.
        edge_types : torch.Tensor
            Tensor of the edge types.
        coeff : torch.Tensor, dtype=torch.float32, requires_grad=True
            Coefficient matrix in basis-decomposition for regularization,
            shape: (num_rels, num_bases). It should be set to ``None`` when ``regularizer=None``.
        feat : torch.Tensor, dtype=torch.float32, requires_grad=True
            Input feature, shape: (num_src_nodes, in_feat).

        Returns
        -------
        agg_output : torch.Tensor, dtype=torch.float32
            Aggregation output, shape: (num_dst_nodes, num_rels * in_feat) when ``regularizer=None``,
            and (num_dst_nodes, num_bases * in_feat) when ``regularizer='basis'``.

        """
        if g.idtype == th.int32:
            mfg_csr_func = message_flow_graph_hg_csr_int32
            agg_fwd_func = agg_hg_basis_post_fwd_int32
        elif g.idtype == th.int64:
            mfg_csr_func = message_flow_graph_hg_csr_int64
            agg_fwd_func = agg_hg_basis_post_fwd_int64
        else:
            raise TypeError(
                f'Supported ID type: torch.int32 or torch.int64, but got {g.idtype}')
        ctx.graph_idtype = g.idtype

        _in_feat = feat.shape[-1]
        indptr, indices, edge_ids = g.adj_sparse('csc')
        # edge_ids is in a mixed order, need to permutate incoming etypes
        # and stash the result for backward propagation
        ctx.edge_types_int32 = edge_types[edge_ids.long()].int()
        # node_types are not needed by the post-variant rgcn aggregators
        _num_node_types = 0
        _out_node_types = _in_node_types = None

        mfg = mfg_csr_func(fanout, g.dstnodes(), g.srcnodes(), indptr, indices,
            _num_node_types, num_rels, _out_node_types, _in_node_types, ctx.edge_types_int32)

        if coeff is None:
            leading_dimension = num_rels * _in_feat
        else:
            _num_bases = coeff.shape[-1]
            leading_dimension = _num_bases * _in_feat

        agg_output = th.empty(g.num_dst_nodes(), leading_dimension, dtype=th.float32, device='cuda')
        if coeff is None:
            agg_fwd_func(agg_output, feat.detach(), mfg)
        else:
            agg_fwd_func(agg_output, feat.detach(), mfg, weights_combination=coeff.detach())

        ctx.backward_cache = mfg
        ctx.save_for_backward(feat, coeff)
        return agg_output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of R-GCN aggregation layer.

        Parameters
        ----------
        grad_output : torch.Tensor, dtype=torch.float32
            Gradient of loss function w.r.t output.

        """
        mfg = ctx.backward_cache
        feat, coeff = ctx.saved_tensors

        if ctx.graph_idtype == th.int32:
            agg_bwd_func = agg_hg_basis_post_bwd_int32
        elif ctx.graph_idtype == th.int64:
            agg_bwd_func = agg_hg_basis_post_bwd_int64
        else:
            raise TypeError(
                f'Supported ID type: torch.int32 or torch.int64, but got {ctx.graph_idtype}')

        grad_feat = th.empty_like(feat, dtype=th.float32, device='cuda')
        if coeff is None:
            grad_coeff = None
            agg_bwd_func(grad_feat, grad_output, feat.detach(), mfg)
        else:
            grad_coeff = th.empty_like(coeff, dtype=th.float32, device='cuda')
            agg_bwd_func(grad_feat, grad_output, feat.detach(), mfg,
                output_weight_gradient=grad_coeff, weights_combination=coeff.detach())

        return None, None, None, None, grad_feat, grad_coeff

class RelGraphConv(nn.Module):
    """ Relational graph convolution layer using the aggregation functions in cugraph-ops.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations.
    fanout : int
        Maximum number of sampled neighbors of an destination node;
        i.e, maximum in degree of destination nodes
    regularizer : str, optional
        Which weight regularizer to use ("basis" or ``None``):

         - "basis" is for basis-decomposition.
         - ``None`` applies no regularization.

        Default: ``None``.
    num_bases : int, optional
        Number of bases. It comes into effect when "basis" regularizer is applied.
        Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: bool, optional
        True to add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> from dgl.dataloading import NeighborSampler, DataLoader
    >>> from dgl.contrib.cugraph.nn import RelGraphConv
    >>> import torch.nn.functional as F
    >>>
    >>> device = 'cuda'
    >>> fanouts = [5, 6]
    >>> sampler = NeighborSampler(fanouts)
    >>> dataloader = DataLoader(g, train_nid, sampler, device=device, batch_size=1024)
    >>> conv1 = RelGraphConv(in_dim, h_dim, num_rels, fanouts[0],
    ...     regularizer='basis', num_bases=10)
    >>> conv2 = RelGraphConv(h_dim, out_dim, num_rels, fanouts[1],
    ...     regularizer='basis', num_bases=10)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     h = conv1(blocks[0], x, blocks[0].edata[dgl.ETYPE])
    ...     h = F.relu(h)
    ...     h = conv2(blocks[1], h, blocks[1].edata[dgl.ETYPE])
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 fanout,
                 regularizer=None,
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0,
                 layer_norm=False):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.fanout = fanout

        # regularizer (see dgl.nn.pytorch.linear.TypedLinear)
        if regularizer is None:
            self.W = nn.Parameter(th.Tensor(num_rels, in_feat, out_feat).cuda())
            self.coeff = None
        elif regularizer == 'basis':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(th.Tensor(num_bases, in_feat, out_feat).cuda())
            self.coeff = nn.Parameter(th.Tensor(num_rels, num_bases).cuda())
            self.num_bases = num_bases
        else:
            raise ValueError(
                f'Supported regularizer options: "basis", but got {regularizer}')
        self.regularizer = regularizer
        self.reset_parameters()

        # others
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat).cuda())
            nn.init.zeros_(self.h_bias)

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

        # weight for self_loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat).cuda())
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        with th.no_grad():
            if self.regularizer is None:
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_feat), 1/math.sqrt(self.in_feat))
            elif self.regularizer == 'basis':
                nn.init.uniform_(self.W, -1/math.sqrt(self.in_feat), 1/math.sqrt(self.in_feat))
                nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))
            else:
                raise ValueError(
                    f'Supported regularizer options: "basis", but got {self.regularizer}')

    def forward(self, g, feat, etypes, norm=None, *, presorted=False):
        """Forward computation.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            A 2D tensor of node features. Shape: :math:`(|V|, D_{in})`.
        etypes : torch.Tensor or list[int]
            An 1D integer tensor of edge types. Shape: :math:`(|E|,)`.
        norm : torch.Tensor, optional
            An 1D tensor of edge norm value.  Shape: :math:`(|E|,)`.
        presorted : bool, optional
            Whether the edges of the input graph have been sorted by their types.
            Forward on pre-sorted graph may be faster. Graphs created
            by :func:`~dgl.to_homogeneous` automatically satisfy the condition.
            Also see :func:`~dgl.reorder_graph` for sorting edges manually.

        Returns
        -------
        torch.Tensor
            New node features. Shape: :math:`(|V|, D_{out})`.
        """
        _sm_size = self.fanout * 2 + 3 + \
                   self.num_rels * self.num_bases if self.regularizer == 'basis' else 0
        _sm_size *= 8 if g.idtype == th.int64 else 4
        if _sm_size > CUDA_SM_PER_BLOCK:
            raise MemoryError(
                f"Failed to allocate {_sm_size} bytes shared memory on CUDA, "
                f"larger than the limit: {CUDA_SM_PER_BLOCK} bytes. "
                f"Try reducing fanout or num_bases.")
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            # message passing
            h = RelGraphConvAgg.apply(g, self.fanout, self.num_rels,
                                      etypes, feat, self.coeff)
            h = h @ self.W.view(-1, self.out_feat)
            # apply bias and activation
            if self.layer_norm:
                h = self.layer_norm_weight(h)
            if self.bias:
                h = h + self.h_bias
            if self.self_loop:
                h = h + feat[:g.num_dst_nodes()] @ self.loop_weight
            if self.activation:
                h = self.activation(h)
            h = self.dropout(h)
            return h
