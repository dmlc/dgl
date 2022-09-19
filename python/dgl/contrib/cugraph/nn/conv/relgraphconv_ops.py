"""Torch Module for Relational graph convolution layer using cugraph-ops"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import (agg_hg_basis_post_fwd_int32,
    agg_hg_basis_post_bwd_int32, agg_hg_basis_post_fwd_int64, agg_hg_basis_post_bwd_int64)
from pylibcugraphops.structure.graph_types import (message_flow_graph_hg_csr_int32,
    message_flow_graph_hg_csr_int64)

class RelGraphConvFunc(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, fanout, num_rels, out_node_types, in_node_types, edge_types,
                feat, W, coeff):
        """
        Compute the forward pass of R-GCN.

        Parameters
        ----------
        g : dgl.heterograph.DGLHeteroGraph
            Heterogeneous graph.

        fanout : int
            Maximum in-degree of nodes.

        num_rels : int
            Number of edge types in this graph.

        out_node_types : torch.Tensor, dtype=torch.int32
            Tensor of the node types of output nodes.

        in_node_types : torch.Tensor, dtype=torch.int32
            Tensor of the node types of input nodes.

        edge_types : torch.Tensor, dtype=torch.int32
            Tensor of the edge types.

        coeff : torch.Tensor, dtype=torch.float32, requires_grad=True
            Coefficient matrix in basis-decomposition for regularization,
            shape: (num_rels, num_bases). It should be set to ``None`` when ``regularizer=None``.

        feat : torch.Tensor, dtype=torch.float32, requires_grad=True
            Input feature, shape: (num_src_nodes, in_feat).

        W : torch.Tensor, dtype=torch.float32, requires_grad=True
            Weights tensor, shape: (num_bases, in_feat, out_feat) when ``regularizer='basis'``, or
            (num_rels, in_feat, out_feat) when ``regularizer=None``.

        Cached
        ------
        agg_out : torch.Tensor, dtype=torch.float32
            Aggregation output, shape: (num_dst_nodes, W.shape[0]*W.shape[1])

        Returns
        -------
        output : torch.Tensor, dtype=torch.float32
            Output feature, shape: (num_dst_nodes, out_feat)

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
        _out_feat = W.shape[-1]
        indptr, indices, _ = g.adj_sparse('csc')  # edge_ids not needed here

        # needed for creating MFG but not actually used in cugraph-ops aggregators
        # can be passed through graph class members if necessary in the future
        _n_node_types = 0

        mfg = mfg_csr_func(fanout, g.dstnodes(), g.srcnodes(), indptr, indices,
            _n_node_types, num_rels, out_node_types, in_node_types, edge_types)

        if coeff is None:
            leading_dimension = num_rels * _in_feat
        else:
            _num_bases = coeff.shape[-1]
            leading_dimension = _num_bases * _in_feat

        agg_out = th.empty(g.num_dst_nodes(), leading_dimension, dtype=th.float32, device='cuda')

        if coeff is None:
            agg_fwd_func(agg_out, feat.detach(), mfg)
        else:
            agg_fwd_func(agg_out, feat.detach(), mfg, weights_combination=coeff.detach())

        output = th.as_tensor(agg_out, device='cuda') @ W.view(leading_dimension, _out_feat)

        ctx.backward_cache = mfg
        ctx.save_for_backward(coeff, feat, W, output, agg_out)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of R-GCN.

        Parameters
        ----------
        grad_output : torch.Tensor, dtype=torch.float32
            Gradient of loss function w.r.t output.

        """
        mfg = ctx.backward_cache
        coeff, feat, W, output, agg_out = ctx.saved_tensors

        if ctx.graph_idtype == th.int32:
            agg_bwd_func = agg_hg_basis_post_bwd_int32
        elif ctx.graph_idtype == th.int64:
            agg_bwd_func = agg_hg_basis_post_bwd_int64
        else:
            raise TypeError(
                f'Supported ID type: torch.int32 or torch.int64, but got {ctx.graph_idtype}')

        # dense backward
        _out_feat = W.shape[-1]
        g_W = agg_out.t() @ grad_output
        agg_out = grad_output @ W.view(-1, _out_feat).t()   # gradient w.r.t input, reuse buffer

        # backward aggregation
        g_in = th.empty_like(feat, dtype=th.float32, device='cuda')

        if coeff is None:
            g_coeff = None
            agg_bwd_func(g_in, agg_out, feat.detach(), mfg)
        else:
            g_coeff = th.empty_like(coeff, dtype=th.float32, device='cuda')
            agg_bwd_func(g_in, agg_out, feat.detach(), mfg,
                output_weight_gradient=g_coeff, weights_combination=coeff.detach())

        return None, None, None, None, None, None, g_in, g_W.view_as(W), g_coeff

class RelGraphConvOps(nn.Module):
    """ Relational graph convolution layer. """
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
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            # message passing
            h = RelGraphConvFunc.apply(g, self.fanout, self.num_rels,
                                       g.dstdata['ntype'], g.srcdata['ntype'], etypes,
                                       feat, self.W, self.coeff)
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
