"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from pylibcugraphops.aggregators.node_level import agg_hg_basis_post_fwd_int32
from pylibcugraphops.structure.graph_types import message_flow_graph_hg_csr_int32

def test_pylibcugraphops():
    import cupy
    cupy.random.seed(0)
    n_out_nodes = 5
    n_in_nodes = 15
    sample_size = 4
    mfg_dtype = cupy.int32
    offsets = cupy.arange(0, sample_size*(1+n_out_nodes), sample_size, dtype=mfg_dtype)
    indices = cupy.random.randint(0, n_in_nodes, sample_size*n_out_nodes, dtype=mfg_dtype)
    out_nodes = cupy.arange(0, n_out_nodes, dtype=mfg_dtype)
    in_nodes = cupy.arange(0, n_in_nodes, dtype=mfg_dtype)

    n_node_types = 6
    n_edge_types = 3
    out_node_types = cupy.random.randint(0, high=n_node_types, size=n_out_nodes, dtype=mfg_dtype)
    in_node_types = cupy.random.randint(0, high=n_node_types, size=n_in_nodes, dtype=mfg_dtype)
    edge_types = cupy.random.randint(0, high=n_edge_types, size=indices.shape[0], dtype=mfg_dtype)

    mfg = message_flow_graph_hg_csr_int32(sample_size, out_nodes, in_nodes, offsets, indices,
        n_node_types, n_edge_types, out_node_types, in_node_types, edge_types)

    dim = 2
    leading_dimension = mfg.get_num_edge_types()*dim
    input_embedding = cupy.random.ranf((mfg.get_num_in_nodes(), dim), dtype=cupy.float32)
    output_embedding = cupy.empty((mfg.get_num_out_nodes(), leading_dimension), dtype=cupy.float32)
    agg_hg_basis_post_fwd_int32(output_embedding=output_embedding, input_embedding=input_embedding, mfg=mfg)

    print(f"{output_embedding = }")