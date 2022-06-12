from torch.fx import Node

import torch
from dgl import DGLHeteroGraph
from dgl.utils import gather_pinned_tensor_rows
from .ssd import SSDGraph

def arg_trace(a):
    ret = set()
    if isinstance(a, Node):
        ret.add(a)
    if isinstance(a, dict):
        for _, v in a.items():
            ret = ret.union(arg_trace(v))
    if isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            ret = ret.union(arg_trace(v))
    elif isinstance(a, slice):
        ret = ret.union(arg_trace((a.start, a.step, a.stop)))
    return ret

def get_new_arg_input(inputs, data_map, input_nodes, inference_graph, device, use_uva=False):
    new_args = ()
    for arg_node in inputs:
        if isinstance(data_map[arg_node], torch.Tensor):
            if data_map[arg_node].device == device:
                new_args += (data_map[arg_node][input_nodes],)
            elif use_uva:
                new_args += (gather_pinned_tensor_rows(data_map[arg_node], input_nodes),)
            else:
                new_args += (data_map[arg_node][input_nodes].to(device),)
        elif isinstance(data_map[arg_node], DGLHeteroGraph) or isinstance(data_map[arg_node], SSDGraph):
            new_args += (inference_graph.to(device),)
        elif hasattr(data_map[arg_node], "to"):
            new_args += (data_map[arg_node].to(device),)
        else:
            new_args += (data_map[arg_node],)
    return new_args

def update_ret_output(output_vals, rets, input_nodes, output_nodes, blocks):
    if not isinstance(output_vals, tuple):
        output_vals = (output_vals,)
    for output_val, ret in zip(output_vals, rets):
        if isinstance(output_val, torch.Tensor):
            if ret is None:
                raise RuntimeError("Can't determine return's type.")
            if output_val.size()[0] == blocks[0].num_dst_nodes():
                update_out_in_chunks(ret, output_nodes, output_val)
            elif output_val.size()[0] == blocks[0].num_src_nodes():
                update_out_in_chunks(ret, input_nodes, output_val)
            else:
                raise RuntimeError("Can't determine return's type.")
        else:
            ret = output_val
    return rets

def update_out_in_chunks(ret, idx, val):
    memory_comsuption = 4 # float, TODO
    for dim in range(1, len(val.shape)):
        memory_comsuption *= val.shape[dim]
    num_nodes = val.shape[0]
    num_node_in_chunks = 33000000 // memory_comsuption
    start, end = 0, 0
    while start < num_nodes:
        end = min(start + num_node_in_chunks, num_nodes)
        ret[idx[start:end]] = val[start:end].cpu()
        start = end
