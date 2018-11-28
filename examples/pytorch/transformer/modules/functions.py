import torch as th
import dgl.function as fn

def src_dot_dst(src_field, dst_field, out_field):
    """
    This function serves as a surrogate for `src_dot_dst` built-in apply_edge function.
    """
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}
    return func

def scaled_exp(field, c):
    def func(edges):
        return {field: th.exp((edges.data[field] / c).clamp(-5, 5))}
    return func
