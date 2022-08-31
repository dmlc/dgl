"""Torch Module for Relational graph convolution layer using cugraph"""
# pylint: disable= no-member, arguments-differ, invalid-name
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

        coeff : torch.Tensor, dtype=torch.float32, device='cuda'
            Coefficient matrix in basis-decomposition for regularization, shape = n_edge_types * n_bases
        
        in_feat : torch.Tensor, dtype=torch.float32, device='cuda'
            Input feature, shape = n_in_nodes * in_feat_dim
        
        W : torch.Tensor, dtype=torch.float32, device='cuda'
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
        agg_hg_basis_post_fwd_int64(agg_out, in_feat, mfg, weights_combination=coeff)

        out_feat_dim = W.shape[-1]
        output = th.as_tensor(agg_out, device='cuda') @ W.view(leading_dimension, out_feat_dim)

        ctx.backward_cache = mfg
        ctx.save_for_backward(coeff, in_feat, W, output, agg_out)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output should have the same dimensionality as output in forward pass.
        """
        mfg = ctx.backward_cache
        coeff, in_feat, W, output, agg_out = ctx.saved_tensors

        # dense backward
        _out_feat_dim = W.shape[-1]
        g_W = agg_out.transpose(0,1) @ grad_output
        agg_out = grad_output @ W.view(-1, _out_feat_dim).transpose(0,1)   # gradient w.r.t input, reuse buffer
        
        # backward aggregation
        g_in = th.empty_like(in_feat, dtype=th.float32, device='cuda')
        g_coeff = th.empty_like(coeff, dtype=th.float32, device='cuda')
        agg_hg_basis_post_bwd_int64(g_in, agg_out, in_feat, mfg,
            output_weight_gradient=g_coeff, weights_combination=coeff)

        return None, None, None, None, None, None, None, g_coeff, g_in, g_W.view_as(W)
