"""Torch Module for Relational graph convolution layer using cugraph"""
# pylint: disable= no-member, arguments-differ, invalid-name
import dgl
import math
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import agg_hg_basis_post_fwd_int64, agg_hg_basis_post_bwd_int64
from pylibcugraphops.structure.graph_types import message_flow_graph_hg_csr_int64

class RgcnFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, g, sample_size, n_node_types, n_edge_types, out_node_types, in_node_types, edge_types,
                coeff, feat, W):
        """
        Compute the forward pass of R-GCN.

        Parameters
        ----------
        g : dgl.heterograph.DGLHeteroGraph, device='cuda'
            Heterogeneous graph.

        sample_size : int64
        n_node_types : int64
        n_edge_types : int64
        out_node_types : ndarray or torch.Tensor, dtype=int32, device='cuda'
        in_node_types : ndarray or torch.Tensor, dtype=int32, device='cuda'
        edge_types : ndarray or torch.Tensor, dtype=int32, device='cuda'

        coeff : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            Coefficient matrix in basis-decomposition for regularization, shape = n_edge_types * n_bases

        feat : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            Input feature, shape = n_in_nodes * in_feat

        W : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            shape = (n_bases+1) * in_feat * out_feat_dim
            leading_dimension = (n_bases+1) * in_feat

        Cached
        ------
        agg_out : torch.Tensor, dtype=torch.float32, device='cuda'
            Aggregation output, shape = n_out_nodes * leading_dimension

        Returns
        -------
        output : torch.Tensor, dtype=torch.float32, device='cuda'
            Output feature, shape = n_out_nodes * out_feat_dim

        """
        _n_out_nodes = g.num_dst_nodes('_N')
        _n_in_nodes = g.num_src_nodes('_N')
        _sample_size = sample_size

        _out_nodes = g.dstnodes()
        _in_nodes = g.srcnodes()

        indptr, indices, edge_ids = g.adj_sparse('csc')

        mfg = message_flow_graph_hg_csr_int64(_sample_size, _out_nodes, _in_nodes, indptr, indices,
            n_node_types, n_edge_types, out_node_types, in_node_types, edge_types)

        _n_bases = coeff.shape[-1]
        leading_dimension = (_n_bases+1) * feat.shape[-1]

        agg_out = th.empty(_n_out_nodes, leading_dimension, dtype=th.float32, device='cuda')
        agg_hg_basis_post_fwd_int64(agg_out, feat.detach(), mfg, weights_combination=coeff.detach())

        out_feat_dim = W.shape[-1]
        output = th.as_tensor(agg_out, device='cuda') @ W.view(leading_dimension, out_feat_dim)

        ctx.backward_cache = mfg
        ctx.save_for_backward(coeff, feat, W, output, agg_out)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute the backward pass of R-GCN.

        Parameters
        ----------
        grad_output : torch.Tensor, dtype=torch.float32, device='cuda'
            Gradient of loss function w.r.t output.

        """
        mfg = ctx.backward_cache
        coeff, feat, W, output, agg_out = ctx.saved_tensors

        # dense backward
        _out_feat_dim = W.shape[-1]
        g_W = agg_out.t() @ grad_output
        agg_out = grad_output @ W.view(-1, _out_feat_dim).t()   # gradient w.r.t input, reuse buffer

        # backward aggregation
        g_in = th.empty_like(feat, dtype=th.float32, device='cuda')
        g_coeff = th.empty_like(coeff, dtype=th.float32, device='cuda')
        agg_hg_basis_post_bwd_int64(g_in, agg_out, feat.detach(), mfg,
            output_weight_gradient=g_coeff, weights_combination=coeff.detach())

        return None, None, None, None, None, None, None, g_coeff, g_in, g_W.view_as(W)

class RgcnConv(nn.Module):
    """ Relational graph convolution layer that provides same interface as `dgl.nn.pytorch.conv.relgraph`. """
    def __init__(self,
                 in_feat,
                 out_feat,
                 n_node_types,
                 num_rels,
                 sample_size,
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
        self.n_node_types = n_node_types
        self.num_rels = num_rels
        self.sample_size = sample_size    # fanout

        # regularizer (see dgl.nn.pytorch.linear.TypedLinear)
        if regularizer is None:
            raise NotImplementedError
        elif regularizer == 'basis':
            if num_bases is None:
                raise ValueError('Missing "num_bases" for basis regularization.')
            self.W = nn.Parameter(th.Tensor(num_bases+1, in_feat, out_feat).cuda())
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

        # self_loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat, out_feat).cuda())
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_val = dropout

        # TODO(tingyu66): only support basis regularization for now
        if num_bases is None or regularizer is None:
            raise NotImplementedError

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feat, elementwise_affine=True)

    def reset_parameters(self):
        with th.no_grad():
            nn.init.uniform_(self.W, -1/math.sqrt(self.in_feat), 1/math.sqrt(self.in_feat))
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))

    def forward(self, g, feat, etypes, norm=None, presorted=False):
        self.presorted = presorted
        with g.local_scope():
            g.srcdata['h'] = feat
            if norm is not None:
                g.edata['norm'] = norm
            _, _, L = g.adj_sparse("csc")
            etypes = etypes[L]
            # message passing
            output = RgcnFunction.apply(g, self.sample_size, self.n_node_types, self.num_rels,
                                        g.dstdata[dgl.NTYPE], g.srcdata[dgl.NTYPE], etypes,
                                        self.coeff, feat, self.W)
            g.dstdata['h'] = output
            h = g.dstdata['h']
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
