"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import agg_hg_basis_post_fwd_int64
from pylibcugraphops.structure.graph_types import message_flow_graph_hg_csr_int64

def test_pylibcugraphops():
    import cupy
    cupy.random.seed(0)
    n_out_nodes = 5
    n_in_nodes = 15
    sample_size = 4
    mfg_dtype = cupy.int64
    offsets = cupy.arange(0, sample_size*(1+n_out_nodes), sample_size, dtype=mfg_dtype)
    indices = cupy.random.randint(0, n_in_nodes, sample_size*n_out_nodes, dtype=mfg_dtype)
    out_nodes = cupy.arange(0, n_out_nodes, dtype=mfg_dtype)
    in_nodes = cupy.arange(0, n_in_nodes, dtype=mfg_dtype)

    n_node_types = 6
    n_edge_types = 3
    out_node_types = cupy.random.randint(0, high=n_node_types, size=n_out_nodes, dtype=cupy.int32)
    in_node_types = cupy.random.randint(0, high=n_node_types, size=n_in_nodes, dtype=cupy.int32)
    edge_types = cupy.random.randint(0, high=n_edge_types, size=indices.shape[0], dtype=cupy.int32)

    mfg = message_flow_graph_hg_csr_int64(sample_size, out_nodes, in_nodes, offsets, indices,
        n_node_types, n_edge_types, out_node_types, in_node_types, edge_types)

    dim = 2
    leading_dimension = mfg.get_num_edge_types()*dim
    input_embedding = cupy.random.ranf((mfg.get_num_in_nodes(), dim), dtype=cupy.float32)
    output_embedding = cupy.empty((mfg.get_num_out_nodes(), leading_dimension), dtype=cupy.float32)
    agg_hg_basis_post_fwd_int64(output_embedding=output_embedding, input_embedding=input_embedding, mfg=mfg)

    print(f"{output_embedding = }")

class OPSRGCN(th.autograd.Function):
    @staticmethod
    def forward(ctx, graph, sample_size, n_nodes_types, n_edge_types, out_node_types, in_node_types, edge_types,
                coeff, in_feat, W):
        """ graph has to be on device,
        [node/edge]_types are ndarrays with a type of numpy.int32,
        coeff, in_feat and W are torch.tensors on device

        Some terms are used interchangeably here: in_feat vs in_embedding, num_rels vs n_edge_types
        DGL uses in_feat to denote input feature size

        coeff   (weight_combs)     : n_edge_types * n_bases
        in_feat (input_embedding)  : n_in_nodes   * in_dim (in_dim is "dimension" in cugraph-ops)
        agg_out (output_embedding) : n_out_nodes  * leading_dimension
        leading dimension          : (n_bases+1)  * in_dim
        """
        import cupy as _cupy

        _n_out_nodes = graph.num_dst_nodes('_N')
        _n_in_nodes = graph.num_src_nodes('_N')
        _sample_size = sample_size

        _out_nodes = graph.dstnodes()
        _in_nodes = graph.srcnodes()

        indptr, indices, edge_ids = graph.adj_sparse('csc')

        mfg = message_flow_graph_hg_csr_int64(_sample_size, _out_nodes, _in_nodes, indptr, indices,
            n_nodes_types, n_edge_types, out_node_types, in_node_types, edge_types)
        
        _n_bases = coeff.shape[-1]
        leading_dimension = (_n_bases+1) * in_feat.shape[-1]
        
        agg_out = _cupy.empty((_n_out_nodes, leading_dimension), dtype=_cupy.float32)
        agg_hg_basis_post_fwd_int64(agg_out, in_feat, mfg, weights_combination=coeff)

        out_feat_dim = W.shape[-1]
        h = th.as_tensor(agg_out, device='cuda') @ W.view(leading_dimension, out_feat_dim)
        ctx.save_for_backward(coeff, in_feat, W, agg_out)
        
        return h

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError
        return super().backward(ctx, *grad_outputs)
