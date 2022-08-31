"""Torch Module for Relational graph convolution layer using cugraph"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import agg_hg_basis_post_fwd_int64, agg_hg_basis_post_bwd_int64
from pylibcugraphops.structure.graph_types import message_flow_graph_hg_csr_int64

class RgcnFunction(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, sample_size, n_node_types, n_edge_types, out_node_types, in_node_types, edge_types,
                coeff, in_feat, W):
        """
        Compute the forward pass of R-GCN.

        Parameters
        ----------
        graph : dgl.heterograph.DGLHeteroGraph, device='cuda'
            Heterogeneous graph.
        
        sample_size : int64
        n_node_types : int64
        n_edge_types : int64
        out_node_types : ndarray or torch.Tensor, dtype=int32, device='cuda'
        in_node_types : ndarray or torch.Tensor, dtype=int32, device='cuda'
        edge_types : ndarray or torch.Tensor, dtype=int32, device='cuda'

        coeff : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            Coefficient matrix in basis-decomposition for regularization, shape = n_edge_types * n_bases
        
        in_feat : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            Input feature, shape = n_in_nodes * in_feat_dim
        
        W : torch.Tensor, dtype=torch.float32, device='cuda', requires_grad=True
            shape = (n_bases+1) * in_feat_dim * out_feat_dim
            leading_dimension = (n_bases+1) * in_feat_dim

        Cached
        ------
        agg_out : torch.Tensor, dtype=torch.float32, device='cuda'
            Aggregation output, shape = n_out_nodes * leading_dimension

        Returns
        -------
        output : torch.Tensor, dtype=torch.float32, device='cuda'
            Output feature, shape = n_out_nodes * out_feat_dim

        """
        _n_out_nodes = graph.num_dst_nodes('_N')
        _n_in_nodes = graph.num_src_nodes('_N')
        _sample_size = sample_size

        _out_nodes = graph.dstnodes()
        _in_nodes = graph.srcnodes()

        indptr, indices, edge_ids = graph.adj_sparse('csc')

        mfg = message_flow_graph_hg_csr_int64(_sample_size, _out_nodes, _in_nodes, indptr, indices,
            n_node_types, n_edge_types, out_node_types, in_node_types, edge_types)
        
        _n_bases = coeff.shape[-1]
        leading_dimension = (_n_bases+1) * in_feat.shape[-1]

        agg_out = th.empty(_n_out_nodes, leading_dimension, dtype=th.float32, device='cuda')
        agg_hg_basis_post_fwd_int64(agg_out, in_feat.detach(), mfg, weights_combination=coeff.detach())

        out_feat_dim = W.shape[-1]
        output = th.as_tensor(agg_out, device='cuda') @ W.view(leading_dimension, out_feat_dim)

        ctx.backward_cache = mfg
        ctx.save_for_backward(coeff, in_feat, W, output, agg_out)
        
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
        coeff, in_feat, W, output, agg_out = ctx.saved_tensors

        # dense backward
        _out_feat_dim = W.shape[-1]
        g_W = agg_out.t() @ grad_output
        agg_out = grad_output @ W.view(-1, _out_feat_dim).t()   # gradient w.r.t input, reuse buffer
        
        # backward aggregation
        g_in = th.empty_like(in_feat, dtype=th.float32, device='cuda')
        g_coeff = th.empty_like(coeff, dtype=th.float32, device='cuda')
        agg_hg_basis_post_bwd_int64(g_in, agg_out, in_feat.detach(), mfg,
            output_weight_gradient=g_coeff, weights_combination=coeff.detach())

        return None, None, None, None, None, None, None, g_coeff, g_in, g_W.view_as(W)

class RgcnConv(nn.Module):
    def __init__(self,
                 in_feat_dim,
                 out_feat_dim,
                 n_node_types,
                 n_edge_types,
                 sample_size,
                 n_bases,
                 bias=True,
                 activation=None,
                 self_loop=True,
                 dropout=0.0):
        super().__init__()  
        self.in_feat_dim = in_feat_dim
        self.out_feat_dim = out_feat_dim
        self.n_node_types = n_node_types
        self.n_edge_types = n_edge_types
        self.sample_size = sample_size    # fanout
        self.n_bases = n_bases

        self.W = nn.Parameter(th.Tensor(self.n_bases+1, self.in_feat_dim, self.out_feat_dim).cuda())
        self.coeff = nn.Parameter(th.Tensor(self.n_edge_types, self.n_bases).cuda())

        self.reset_parameters()

        # others
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        
        # bias
        if self.bias:
            self.h_bias = nn.Parameter(th.Tensor(out_feat_dim).cuda())
            nn.init.zeros_(self.h_bias)
        
        # self_loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(in_feat_dim, out_feat_dim).cuda())
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        self.dropout_val = dropout

    def reset_parameters(self):
        with th.no_grad():
            nn.init.uniform_(self.W, -1/math.sqrt(self.in_feat_dim), 1/math.sqrt(self.in_feat_dim))
            nn.init.xavier_uniform_(self.coeff, gain=nn.init.calculate_gain('relu'))

    def forward(self, graph, feat, etypes, norm=None, presorted=False):
        self.presorted = presorted
        with graph.local_scope():
            graph.srcdata['h'] = feat
            if norm:
                graph.edata['norm'] = norm
            _, _, L = graph.adj_sparse("csc")
            etypes = etypes[L]

            T = RgcnFunction.apply(graph, self.sample_size, self.n_node_types, self.n_edge_types,
                                   graph.dstdata['ntype'], graph.srcdata['ntype'], etypes,
                                   self.coeff, feat, self.W)
            graph.dstdata['h'] = T
            h = graph.dstdata['h']

            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = h + feat[:graph.num_dst_nodes()] @ self.loop_weight

            h = self.dropout(h)

            return h